
import numpy as np
import pandas as pd
import sklearn
import re
import os
from pynvml import *
import torch
print(torch.__version__)
import os
from collections import defaultdict
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union
import torch.utils.checkpoint
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import os
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
import torch.nn as nn
from peft import LoraConfig, get_peft_model, inject_adapter_in_model, TaskType, prepare_model_for_kbit_training
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
import torch
from transformers import LlamaConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention,
    LlamaSdpaAttention,
    LlamaFlashAttention2,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_flash_attention_utils import _upad_input, pad_input, flash_attn_varlen_func
from salmon import SALMONHead


"""
Package versions used:
! pip install transformers==4.46.1
! pip install datasets
! pip install accelerate nvidia-ml-py3
! pip install sentencepiece
! pip install peft
! pip install bitsandbytes
! pip install -U flash-attn
! pip install pandas numpy scikit-learn matplotlib statsmodels cython transformers==4.46.1 datasets accelerate nvidia-ml-py3 sentencepiece peft bitsandbytes -U flash-attn
"""


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}

logger = logging.get_logger(__name__)


class NewsDS(Dataset):
    def __init__(self, df, tokenizer, model, params, ts_features=None):
        if params is None:
            params = dict()
        self.df = df
        self.tokenizer = tokenizer
        self.model = model
        self.params = params
        self.inputs = defaultdict(list)        
        self.num_hist = self.params.get('num_text_hist', 10)
        self.text_hist_col = self.params.get('text_hist_col', 'Text')
        self.max_date_diff = self.params.get('max_hist_days')
        
        logger.info(f'Prefilter: {len(self.df)}')
        
        # Process time series features
        if ts_features is not None and self.params.get('include_ts', False):
            assert len(self.df) == len(ts_features)
            self.ts_features = ts_features
            self.ts_max_lags = self.params['ts_max_lags']
            
            if self.params.get('ts_clip', None) is not None:
                self.ts_clip = self.params.get('ts_clip')
                logger.info(f'>> Clipping Historical TS at [{-self.ts_clip}, {self.ts_clip}]')
                self.ts_features = np.clip(self.ts_features, -self.ts_clip, self.ts_clip)
            
            self.inputs['ts_features'] = torch.FloatTensor(self.ts_features[:, -self.ts_max_lags:])
        
        # Labels
        self.inputs['labels'] = torch.LongTensor(self.df['labels'].tolist())
        
        # Tokenization parameters
        self.max_length_hist = self.params.get('max_length_hist', 512)
        self.batch_size = 10000
        
        # Padding IDs for detecting empty articles
        self.hist_pad_ids = self.tokenizer([''], max_length=self.max_length_hist,
                                           padding='max_length', truncation=True,
                                           return_tensors='pt')['input_ids'][0]
        
        # Process in batches
        for batch_index in range(0, len(self.df), self.batch_size):
            batch_end = batch_index + self.batch_size
            
            # Process each historical article (hist_0 is the most recent/"main" article)
            for i in range(self.num_hist):
                # Determine column name
                text_col = self.text_hist_col if i == 0 else f'{self.text_hist_col}_{i}'
                
                # Tokenize
                text_batch = self.df[text_col].iloc[batch_index:batch_end].tolist()
                d = self.tokenizer(text_batch, max_length=self.max_length_hist,
                                 padding='max_length', truncation=True,
                                 return_tensors='pt')
                
                # Detect empty articles
                hist_mask = (d['input_ids'] == self.hist_pad_ids).all(dim=-1).bool()
                
                # Store tokenized outputs
                for k in d.keys():
                    if 'token_type_ids' not in k:
                        self.inputs[f'{k}_hist_{i}'].append(d[k].type(torch.IntTensor))
                
                self.inputs[f'hist_mask_{i}'].append(hist_mask)
                
                # Date difference (0 for hist_0, actual values for others)
                if i == 0:
                    hist_time = torch.zeros(batch_end - batch_index, dtype=torch.int)
                else:
                    hist_time = torch.IntTensor(
                        self.df[f'date_diff_lag_{i}'].iloc[batch_index:batch_end].astype(int).tolist()
                    )
                self.inputs[f'hist_time_{i}'].append(hist_time)
        
        # Concatenate batches
        for k in self.inputs.keys():
            if isinstance(self.inputs[k], list):
                self.inputs[k] = torch.cat(self.inputs[k], dim=0)
        
        # Combine masks and times
        self.inputs['hist_mask'] = torch.cat(
            [self.inputs[f'hist_mask_{i}'][:, None] for i in range(self.num_hist)],
            dim=-1
        )
        self.inputs['hist_time'] = torch.cat(
            [self.inputs[f'hist_time_{i}'][:, None] for i in range(self.num_hist)],
            dim=-1
        )
    
    def __len__(self):
        return len(self.inputs['labels'])
    
    def __getitem__(self, idx):
        batch = {}
        for k in self.inputs:
            if 'hist_mask' in k:
                batch[k] = self.inputs[k][idx].type(torch.BoolTensor)
            elif k in ['ts_features']:
                batch[k] = self.inputs[k][idx].type(torch.FloatTensor)
            else:
                batch[k] = self.inputs[k][idx].type(torch.LongTensor)
        return batch


def init_new_token_embedding(model, config):
    """Initialize a single new token embedding from LLM token embedding distribution"""

    # Get model token embedding distribution
    old_embeddings = model.get_input_embeddings()
    old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
    old_num_tokens = old_embeddings_weight.size(0)

    # Compute mean and covariance
    mean_embeddings = torch.mean(old_embeddings_weight, dim=0)
    old_centered_embeddings = old_embeddings_weight - mean_embeddings
    covariance = old_centered_embeddings.T @ old_centered_embeddings / old_num_tokens

    # Check if the covariance is positive definite
    eigenvalues = torch.linalg.eigvals(covariance)
    is_covariance_psd = bool(
        (covariance == covariance.T).all() and not torch.is_complex(eigenvalues) and (eigenvalues.real > 0).all()
    )

    if is_covariance_psd:
        logger.info('Covariance matrix is positive definite, sampling from distribution')
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            mean_embeddings, covariance_matrix=1e-9 * covariance
        )
        # Sample a single embedding vector
        new_embedding_data = distribution.sample().to(old_embeddings.weight.dtype)
    else:
        logger.info('Covariance matrix is not positive definite, initializing with mean')
        new_embedding_data = mean_embeddings.to(old_embeddings.weight.dtype)

    # Create Parameter from the sampled/mean embedding
    new_embedding = nn.Parameter(new_embedding_data.clone(), requires_grad=True)

    return new_embedding


def initialize_embeddings_from_token_embedding_dist(new_embeddings, model, config):
    old_embeddings = model.get_input_embeddings()
    old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
    old_num_tokens = old_embeddings_weight.size(0)
    added_num_tokens = new_embeddings.weight.data.size(0)
    mean_embeddings = torch.mean(old_embeddings_weight, dim=0)
    old_centered_embeddings = old_embeddings_weight - mean_embeddings
    covariance = old_centered_embeddings.T @ old_centered_embeddings / old_num_tokens

    # Check if the covariance is positive definite.
    eigenvalues = torch.linalg.eigvals(covariance)
    is_covariance_psd = bool(
        (covariance == covariance.T).all() and not torch.is_complex(eigenvalues) and (eigenvalues > 0).all()
    )
    if is_covariance_psd:
        logger.info(
            'covariance matrixc is positive definite, a distribution can be created. and we can sample new weights from it')
        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            mean_embeddings, covariance_matrix=1e-9 * covariance
        )
        new_embeddings.weight.data = distribution.sample(
            sample_shape=(added_num_tokens,)
        ).to(old_embeddings.weight.dtype)
    else:
        logger.info(
            'covariance matrix is not positive definite, resort to initialize with the mean because distribtion is not well defined created.')
        new_embeddings.weight.data = (
            mean_embeddings[None, :].repeat(added_num_tokens, 1).to(old_embeddings.weight.dtype)
        )

    return new_embeddings


class TSEncoder(nn.Module):
    """Time Series Encoder that discretizes continuous values with quantile binning based on training distribution"""

    def __init__(self, train_ts, params, model=None, config=None, vocab_size=None):
        super().__init__()
        self.params = params
        self.config = config
        if config is not None:
            self.hidden_dim = self.config.hidden_size
        else:
            self.hidden_dim = self.params['ts_hidden_size']
        if vocab_size is None:
            self.vocab_size = self.params['ts_vocab_size']
        else:
            self.vocab_size = vocab_size
        self.epsilon = 1e-6

        flat_train_ts = torch.from_numpy(train_ts).reshape(-1)
        flat_train_ts = flat_train_ts[flat_train_ts != 0.00]
        max_samples = 10 ** 6
        if flat_train_ts.numel() > max_samples:
            perm = torch.randperm(flat_train_ts.numel())[:max_samples]
            flat_train_ts = flat_train_ts[perm]
        quantiles = torch.linspace(0, 1, self.vocab_size + 1, dtype=flat_train_ts.dtype, device=flat_train_ts.device)
        self.bin_edges = torch.quantile(flat_train_ts, quantiles)

        if len(torch.unique(self.bin_edges)) < len(self.bin_edges):
            data_range = flat_train_ts.max() - flat_train_ts.min()
            noise = torch.arange(len(self.bin_edges), device=flat_train_ts.device) * self.epsilon * data_range
            self.bin_edges = self.bin_edges + noise
            self.bin_edges[0] = flat_train_ts.min()
            self.bin_edges[-1] = flat_train_ts.max()

        self.bin_edges = self.bin_edges.detach()
        logger.info('>> Using Discrete Time Series Embeddings (Pointwise) >>')
        self.ts_embedding = nn.Embedding(self.vocab_size, self.hidden_dim, max_norm=1.0, norm_type=2)

        if model is not None and config is not None:
            logger.info('>> Initializing Time Embeddings from Multivariate Normal sampled from LLM Token Embeddings >>')
            self.ts_embedding = initialize_embeddings_from_token_embedding_dist(self.ts_embedding, model, config)


    def forward(self, ts_values):
        """
        Args:
            ts_values: Tensor of shape [num_points] containing continuous TS values
        Returns:
            ts_embeddings: Tensor of shape [num_points, hidden_dim]
            ts_tokens: Tensor of shape [num_points] containing discrete token IDs
        """
        bin_edges = self.bin_edges.to(ts_values.device)
        ts_tokens = torch.bucketize(ts_values.contiguous(), bin_edges) - 1
        ts_tokens = torch.clamp(ts_tokens, 0, self.vocab_size - 1)
        ts_embeddings = self.ts_embedding(ts_tokens)
        return ts_embeddings, ts_tokens


class Classifier(torch.nn.Module):
    """
    Simple classification head for prediction from pooled hidden states
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = torch.nn.Linear(self.config.class_input, self.config.class_input // 2)
        self.dropout = torch.nn.Dropout(0.10)
        self.activation = torch.nn.ReLU()
        self.out_proj = torch.nn.Linear(self.config.class_input // 2, self.config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MSE_ITT_Layer(LlamaDecoderLayer):

    """A Llama decoder layer modified for the MSE-ITT architecture.

    This layer extends the standard LlamaDecoderLayer to process sequences
    containing both text and time-series data. It uses a `modality_ids` tensor
    to route inputs through either the original pretrained text parameters or
    newly added, modality-specific parameters for time-series. This allows for
    specialized processing while preserving the model's language capabilities.

    The layer supports selective cross-modal attention, enabling attention to be
    computed either within each modality (selected) or globally across all tokens,
    controlled by the cross_mod_attention flag.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        if hasattr(self.config, 'cross_mod_attention_layers') and self.config.cross_mod_attention_layers is not None:
            self.cross_mod_attention = layer_idx in self.config.cross_mod_attention_layers
        else:
            self.cross_mod_attention = layer_idx >= (self.config.num_hidden_layers // 2)

        if hasattr(self.config, 'separate_ts_params') and self.config.separate_ts_params is not None:
            self.separate_ts_params = True
        else:
            self.separate_ts_params = False

        # Text (original) modules -----
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.separate_ts_params:
            # Add dedicated TS-specific parameters
            if config.separate_ts_attn_params and self.separate_ts_params:
                self.ts_self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

            if config.separate_ts_ln_params and self.separate_ts_params:
                self.ts_input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.ts_post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            if config.separate_ts_mlp_params and self.separate_ts_params:
                self.ts_mlp = LlamaMLP(config)


    def init_ts_weights(self):

        """Initializes TS-specific weights from pretrained language weights.

        This method copies the state dictionaries from the original text-based
        modules (self_attn, mlp, layernorms) to their newly created time-series
        counterparts. This ensures that the new parameters start with a
        reasonable initialization.
        """

        if hasattr(self, 'ts_self_attn'):
            logger.info(f'Initializing Added TS Attn Params from Text in Layer: {self.layer_idx}')
            self.ts_self_attn.load_state_dict(self.self_attn.state_dict())
        if hasattr(self, 'ts_mlp'):
            logger.info(f'Initializing Added TS MLP Params from Text in Layer: {self.layer_idx}')
            self.ts_mlp.load_state_dict(self.mlp.state_dict())
        if hasattr(self, 'ts_input_layernorm'):
            logger.info(f'Initializing Added TS Input Layernorm Params from Text in Layer: {self.layer_idx}')
            self.ts_input_layernorm.load_state_dict(self.input_layernorm.state_dict())
        if hasattr(self, 'ts_post_attention_layernorm'):
            logger.info(f'Initializing Added TS Input Layernorm Params from Text in Layer: {self.layer_idx}')
            self.ts_post_attention_layernorm.load_state_dict(self.post_attention_layernorm.state_dict())


    def apply_qkv_proj(self, hidden_states, mod='text'):

        """Applies the QKV projection for a specified modality.

        This is a helper method that selects the appropriate QKV projection
        weights (either for text or time-series) and applies them to the
        input hidden states.

        Args:
            hidden_states (torch.Tensor): The input hidden states.
            mod (str, optional): The modality to use, either 'text' or 'ts'.
                Defaults to 'text'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                the query, key, and value tensors.

        Raises:
            ValueError: If an invalid modality is provided.
        """

        if mod == 'text':
            q = self.self_attn.q_proj(hidden_states)
            k = self.self_attn.k_proj(hidden_states)
            v = self.self_attn.v_proj(hidden_states)
        elif mod == 'ts':
            q = self.ts_self_attn.q_proj(hidden_states)
            k = self.ts_self_attn.k_proj(hidden_states)
            v = self.ts_self_attn.v_proj(hidden_states)
        else:
            raise ValueError(f'Invalid Modality: {mod}')

        return q, k, v


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        modality_ids: torch.Tensor = None,  # Expected shape: [batch, seq_len]; 0=text, 1=time series
        past_key_value: torch.Tensor = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor = None,
        position_embeddings: tuple = None,  # (cos, sin)
        **kwargs
    ):

        """Performs a forward pass through the interleaved decoder layer.

        The forward pass routes inputs based on `modality_ids`. It applies
        modality-specific LayerNorm, attention, and MLP transformations.
        Attention can be either isolated within modalities or fully
        cross-modal, controlled by the `self.cross_mod_attention` attribute.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape
                `(batch, seq_len, embed_dim)`.
            attention_mask (torch.Tensor, optional): Attention mask of shape
                `(batch, 1, seq_len, seq_len)`. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs for rotary
                embeddings. Defaults to None.
            modality_ids (torch.Tensor, optional): Tensor of shape
                `(batch, seq_len)` where 0 indicates text and 1 indicates
                time-series. If None, all tokens are treated as text.
            past_key_value (torch.Tensor, optional): Cached past key and value
                states. Defaults to None.
            output_attentions (bool, optional): Whether to return attention
                weights. Defaults to False.
            use_cache (bool, optional): Whether to use caching for faster
                decoding. Defaults to False.
            cache_position (torch.Tensor, optional): The position of the tokens
                in the cache. Defaults to None.
            position_embeddings (tuple, optional): A tuple containing the
                precomputed cosine and sine for rotary position embeddings.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the output hidden states.
        """

        if modality_ids is None:
            modality_ids = torch.zeros(hidden_states.size(0), hidden_states.size(1), dtype=torch.long, device=hidden_states.device)

        assert modality_ids is not None, '>> Modality IDs must be defined >>'

        text_idx = ((modality_ids == 0) & (attention_mask == 1)).nonzero(as_tuple=True)
        ts_idx = ((modality_ids == 1) & (attention_mask == 1)).nonzero(as_tuple=True)

        # Input LN Routing
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.config.separate_ts_ln_params and self.separate_ts_params:
            hidden_states[ts_idx] = self.ts_input_layernorm(residual[ts_idx])

        # Self-Attention Block
        batch, seq_len, _ = hidden_states.size()

        # Compute Q, K, V from both text and time series branches
        q, k, v = self.apply_qkv_proj(hidden_states, 'text')

        if self.config.separate_ts_attn_params and self.separate_ts_params:
            q_ts, k_ts, v_ts = self.apply_qkv_proj(hidden_states[ts_idx], 'ts')
            # Route based on modality
            q[ts_idx] = q_ts
            k[ts_idx] = k_ts
            v[ts_idx] = v_ts

        # Reshape Q, K, V to [batch, num_heads, seq_len, head_dim].
        q = q.view(batch, seq_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        dropout_rate = self.self_attn.attention_dropout if self.training else 0.0

        input_dtype = q.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.self_attn.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)


        if not self.cross_mod_attention:

            attn_output = torch.zeros_like(q, device=q.device, dtype=q.dtype)  # [B, S, nH, hDim]

            # Loop over modalities (e.g. 0=text, 1=time series)
            for mod in modality_ids.unique():
                # Create a boolean mask for the current modality; shape [B, S]
                mod_mask = (modality_ids == mod) & (attention_mask == 1)

                if not mod_mask.any():
                    continue

                # Count tokens per batch
                batch_size = hidden_states.size(0)
                mod_counts = [mod_mask[b].sum().item() for b in range(batch_size)]
                total_mod_tokens = sum(mod_counts)
                max_mod_len = max(mod_counts) if mod_counts else 0

                if max_mod_len == 0:
                    continue

                # Extract batch indices and sequence positions
                batch_indices, seq_indices = mod_mask.nonzero(as_tuple=True)
                # Extract q, k, v for this modality
                q_mod = q[batch_indices, seq_indices]
                k_mod = k[batch_indices, seq_indices]
                v_mod = v[batch_indices, seq_indices]

                mod_counts_tensor = torch.tensor(mod_counts, device=q.device, dtype=torch.int32)
                cu_seq_lens = torch.cat([
                    torch.zeros(1, device=q.device, dtype=torch.int32),
                    torch.cumsum(mod_counts_tensor, dim=0).to(torch.int32)
                    # Explicitly cast back to int32
                ])

                # Call flash attention varlen on the unpadded (flattened) tokens.
                out_flat = flash_attn_varlen_func(
                    q_mod,
                    k_mod,
                    v_mod,
                    cu_seqlens_q=cu_seq_lens,
                    cu_seqlens_k=cu_seq_lens,
                    max_seqlen_q=max_mod_len,
                    max_seqlen_k=max_mod_len,
                    dropout_p=dropout_rate,
                    causal=self.self_attn.is_causal,
                )
                attn_output[batch_indices, seq_indices] = out_flat
        else:
            attn_output = _flash_attention_forward(
                q,
                k,
                v,
                attention_mask,
                seq_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=self.self_attn._flash_attn_uses_top_left_mask,
                is_causal=self.self_attn.is_causal,
            )

        attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()

        # Apply modality-specific output projections
        hidden_states = self.self_attn.o_proj(attn_output)
        if self.config.separate_ts_attn_params and self.separate_ts_params:
            out_ts = self.ts_self_attn.o_proj(attn_output[ts_idx])
            hidden_states[ts_idx] = out_ts

        hidden_states = residual + hidden_states

        # MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.config.separate_ts_ln_params and self.separate_ts_params:
            norm_ts = self.ts_post_attention_layernorm(residual[ts_idx])
            hidden_states[ts_idx] = norm_ts

        mlp_output = self.mlp(hidden_states)
        if self.config.separate_ts_mlp_params and self.separate_ts_params:
            mlp_ts = self.ts_mlp(hidden_states[ts_idx])
            mlp_output[ts_idx] = mlp_ts

        hidden_states = residual + mlp_output

        return (hidden_states,)


class MSE_ITT_Model(LlamaModel):

    """A Llama model adapted for MSE-ITT architecture.

    This model replaces the standard LlamaDecoderLayer with the custom
    MSE_ITT_Layer. It is designed to process sequences
    containing mixed modalities (e.g., text and time-series) by leveraging
    modality-specific parameters within each layer.

    Args:
        config (LlamaConfig): The model configuration object.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MSE_ITT_Layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def init_ts_weights(self):
        for layer in self.layers:
            layer.init_ts_weights()

    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        modality_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        """Performs a forward pass through the MSE-ITT model.

        This method takes standard transformer inputs along with a `modality_ids`
        tensor, which is passsed to each LlamaInterleavedDecoderLayer
        to enable modality-specific routing.

        Args:
            input_ids (torch.LongTensor, optional): Indices of input sequence
                tokens in the vocabulary. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Mask to avoid
                performing attention on padding token indices. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): Indices of positions
                of each input sequence token in the position embeddings.
            modality_ids (Optional[torch.LongTensor], optional): Tensor indicating
                the modality of each token (0 for text, 1 for time-series).
            past_key_values (Optional[Cache], optional): Pre-computed key and
                value hidden states. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): Optionally,
                pass embedded input instead of `input_ids`. Defaults to None.
            use_cache (Optional[bool], optional): If True, past key values are
                returned and can be used to speed up decoding. Defaults to None.
            output_attentions (Optional[bool], optional): Whether or not to return
                the attentions tensors of all attention layers. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether or not to return
                the hidden states of all layers. Defaults to None.
            return_dict (Optional[bool], optional): Whether or not to return a
                `BaseModelOutputWithPast` instead of a plain tuple.
            cache_position (Optional[torch.LongTensor], optional): The position of
                the tokens in the cache.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The model's output, which is
                either a tuple or a `BaseModelOutputWithPast` object containing
                the last hidden state, past key values, hidden states, and attentions.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    modality_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    modality_ids=modality_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


def get_lm_weights(model_path):

    config = AutoConfig.from_pretrained(model_path, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN, config=config)
    lm_weights = model.lm_head.weight.data
    return lm_weights


def create_peft_config(params, model_config):

    """Create LoRA configuration for finetuning TS-specifc parameters in MSE-ITT model"""

    target_modules = []
    modules_to_save = []

    ts_attn_modules = ["ts_self_attn.q_proj", "ts_self_attn.k_proj", "ts_self_attn.v_proj", "ts_self_attn.o_proj"]
    ts_mlp_modules = ["ts_mlp.gate_proj", "ts_mlp.up_proj", "ts_mlp.down_proj"]
    ts_ln_modules = ['ts_input_layernorm', 'ts_post_attention_layernorm']

    for layer_idx in range(model_config.num_hidden_layers):
        if params['separate_ts_attn_params']:
            for module in ts_attn_modules:
                target_modules.append(f"layers.{layer_idx}.{module}")

        if params['separate_ts_mlp_params']:
            for module in ts_mlp_modules:
                target_modules.append(f"layers.{layer_idx}.{module}")

        if params['separate_ts_ln_params']:
            for module in ts_ln_modules:
                modules_to_save.append(f"layers.{layer_idx}.{module}")

    peft_config = LoraConfig(
        use_rslora=params.get('use_rslora', True),
        use_dora=params.get('use_dora', False),
        r=params.get('lora_rank', 16),
        lora_alpha=params.get('lora_alpha', 32),
        target_modules=target_modules if target_modules else None,
        bias="none",
        modules_to_save=modules_to_save if modules_to_save else None,
        lora_dropout=0.05)

    return peft_config


class ModelClassifierLM(transformers.PreTrainedModel):

    def __init__(self, model_name, config, params, tokenizer, hist_tokenizer, torch_dtype,
                 base_path, train_ts):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.config = config
        self.tokenizer = tokenizer
        self.hist_tokenizer = hist_tokenizer
        self.base_path = base_path
        self.from_checkpoint = self.params.get('from_checkpoint', False)
        self.checkpoint_name = self.params.get('checkpoint_name', None)
        self.checkpoint_path = self.params.get('checkpoint_path', None)
        self.pretrained_path = self.params.get('pretrained_path', self.checkpoint_path)
        self.config.num_labels = self.params['num_labels']
        self.task_type = self.params.get('task_type', 'CLASS')
        self.torch_dtype = torch_dtype
        self.pooling_method = self.params.get('pooling_method', 'MEAN')
        self.include_hist = self.params.get('include_text_hist', False)
        self.num_hist = self.params.get('num_text_hist', 10)
        self.train_ts = train_ts
        self.output_path = self.params['output_path']

        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"PyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        if self.task_type == 'SALMON':
            self.pooling_method = None
            self.params['val_metric'] = 'loss'

        logger.info(f'>> Using {self.pooling_method} pooling method for document representation >>')
        self.config = transformers.AutoConfig.from_pretrained(model_name, token=HF_TOKEN)
        if self.params.get('attn_implementation', None) in ['sdpa', 'flash_attention_2', 'eager']:
            logger.info(f'>> Using {self.params.get("attn_implementation")} implementation for attention computation >>')
            self.config._attn_implementation = self.params.get('attn_implementation')

        if self.params.get('quantization', True):
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype
            )
            self.torch_dtype = None

        elif self.torch_dtype in [torch.float16,
                                      torch.bfloat16]:
            self.torch_dtype = None
            self.bnb_config = None
        else:
            self.bnb_config = None

        logger.info(f'BNB Config: {self.bnb_config}')
        logger.info(f'Torch dtype: {self.torch_dtype}')

        self.config.separate_ts_params = self.params['separate_ts_params']
        self.config.separate_ts_ln_params = self.params['separate_ts_ln_params']
        self.config.separate_ts_attn_params = self.params['separate_ts_attn_params']
        self.config.separate_ts_mlp_params = self.params['separate_ts_mlp_params']

        if self.params.get('cross_mod_attention_layers') is None:
            self.config.cross_mod_attention_layers = list(range(self.config.num_hidden_layers // 2,
                                                                self.config.num_hidden_layers))
        else:
            self.config.cross_mod_attention_layers = self.params['cross_mod_attention_layers']
        logger.info(f'Cross-modal attention in layers: {self.config.cross_mod_attention_layers}')

        self.bert = MSE_ITT_Model.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=None,
            trust_remote_code=True,
            config=self.config,
            quantization_config=self.bnb_config)

        self.bert.init_ts_weights()

        if self.params.get('gradient_checkpointing', False):
            logger.info('>> Enabling gradient checkpointing >>')
            self.bert.gradient_checkpointing_enable()

        if self.params.get('quantization', False):
            logger.info('>> Enabling quantization >>')
            self.bert = prepare_model_for_kbit_training(self.bert)

        if self.params.get('peft', False):
            logger.info('>> Using PEFT with LORA >>')
            self.peft_config = create_peft_config(self.params, self.config)
            self.enable_peft()
            self.bert.print_trainable_parameters()

        self.config.d_model = self.config.hidden_size
        logger.info('>> Resizing token embedding matrix >>')
        self.bert.resize_token_embeddings(len(self.tokenizer))

        if self.task_type == 'SALMON':
            logger.info(f'>> Adding {self.task_type} head >>')
            self.salmon_head = SALMONHead(self.params, self.bert, self.config, self.tokenizer)
            logger.info('Initializing LM Head from Pretrained Checkpoint')
            self.salmon_head.lm_head.weight.data = get_lm_weights(model_name).to(torch.float32)

        if self.params.get('include_ts', False):
            self.ts_model = TSEncoder(self.train_ts, self.params)
            self.max_hist_days = self.params['max_hist_days']
            logger.info(f'Max Hist Days: {self.max_hist_days}')

            if self.params['article_sep_tokens']:
                self.article_start_emb = init_new_token_embedding(self.bert, self.config)
                self.article_end_emb = init_new_token_embedding(self.bert, self.config)

            if self.params['include_mod_sep_token']:
                self.mod_sep_emb = init_new_token_embedding(self.bert, self.config)

            if self.params['learnable_eos_token']:
                self.eos_token_emb = init_new_token_embedding(self.bert, self.config)

        self.config.class_input = self.config.hidden_size
        self.config.num_labels = self.params['num_labels']
        self.classifier = Classifier(self.config)


    def enable_peft(self):

        self.peft_config = create_peft_config(self.params, self.config)
        self.bert = get_peft_model(self.bert, self.peft_config)


    def mod2id(self, mod_name):
        if mod_name == 'TEXT':
            return 0
        elif mod_name == 'TS':
            return 1
        else:
            raise ValueError(f'Invalid Modality Name: {mod_name}')


    def print_tensor(self, tensor_name, tensor_value):
        print(f'{tensor_name}: shape = {tensor_value.size()}, value = {tensor_value}')


    def get_ts_inputs(self, embeddings, modality_ids, positions):
        """Extract time series values and positions"""
        ts_values = []
        ts_positions = []
        for (embedding, modality_id, position) in zip(embeddings, modality_ids, positions):
            if modality_id == self.mod2id('TS'):
                ts_values.append(embedding)
                ts_positions.append(position)
        return ts_values, ts_positions


    def extend_tensor(self, t1, t2, dim=0):
        return torch.cat([t1, t2], dim=dim)


    def concat_token_embeddings(self, batch, return_token_ids=False):
        token_es = []
        ams = []
        times = []
        token_ids = []

        # Iterate from oldest to newest (hist_{num_hist-1} down to hist_0)
        for j in range(self.num_hist - 1, -1, -1):
            input_ids = batch[f'input_ids_hist_{j}']
            time_ids = batch[f'hist_time'][:, j]  # Extract j-th column from hist_time tensor
            attention_mask = batch[f'attention_mask_hist_{j}']
            token_embeddings = self.bert.get_input_embeddings()(input_ids)

            token_es.append(token_embeddings)
            ams.append(attention_mask)
            times.append(time_ids)
            token_ids.append(input_ids)

        if return_token_ids:
            return token_es, ams, times, token_ids
        else:
            return token_es, ams, times


    def create_interleaved_sequence(self, batch, ts_embeddings=None):

        """Create interleaved sequence of text and TS token embeddings with pointwise TS tokenization"""

        token_es, ams, times, token_ids = self.concat_token_embeddings(batch, return_token_ids=True)

        all_embeddings = []
        modality_ids = []
        all_token_inputs = []
        all_ts_inputs = []
        all_ts_tokens = []

        for i in range(batch['input_ids'].size(0)):
            embeddings = None
            token_inputs = None
            ts_inputs = None
            mods = []
            positions = []

            ts = batch['ts_features'][i, :]
            T = self.params['max_hist_days']
            if T > len(ts):
                padding = torch.zeros(T - len(ts), dtype=ts.dtype, device=ts.device)
                ts = torch.cat([padding, ts], dim=0)
            assert T <= len(ts), f"Time series length {len(ts)} < required {T}"

            prev_t = 0
            prev_mod = None

            for j in range(self.num_hist):
                token_embeddings = token_es[j][i]
                attention_mask = ams[j][i]
                time_id = times[j][i]
                tokens = token_ids[j][i]

                if embeddings is None:
                    embeddings = torch.empty((0, token_embeddings.size(-1)), device=token_embeddings.device,
                                             dtype=token_embeddings.dtype)
                if token_inputs is None:
                    token_inputs = torch.empty((0), device=tokens.device, dtype=tokens.dtype)
                if ts_inputs is None:
                    ts_inputs = torch.empty((0), device=ts.device, dtype=ts.dtype)

                seq_len = attention_mask.sum()
                article_day = (T - time_id).item()

                if article_day < 0 or time_id < 0:
                    continue

                if self.tokenizer.padding_side == 'left':
                    token_embeddings = token_embeddings[-seq_len:, :]
                    tokens = tokens[-seq_len:]
                elif self.tokenizer.padding_side == 'right':
                    token_embeddings = token_embeddings[:seq_len, :]
                    tokens = tokens[:seq_len]

                if self.params.get('article_sep_tokens', True):
                    s1 = self.article_start_emb[None, :]
                    s2 = self.article_end_emb[None, :]
                    token_embeddings = torch.cat([s1, token_embeddings, s2], dim=0)
                    st = torch.full((1,), -100, dtype=tokens.dtype, device=tokens.device)
                    tokens = torch.cat([st, tokens, st], dim=0)

                # Add pointwise TS tokens between articles
                if article_day > prev_t:
                    num_ts_points = article_day - prev_t
                    # Create placeholder embeddings (will be replaced after encoding)
                    ts_placeholder = torch.zeros(num_ts_points, token_embeddings.size(-1),
                                                 device=token_embeddings.device, dtype=token_embeddings.dtype)
                    embeddings = self.extend_tensor(embeddings, ts_placeholder)

                    # Create token inputs (all -100 for TS)
                    token_inputs = self.extend_tensor(token_inputs,
                                                      torch.full((num_ts_points,), -100,
                                                                 dtype=token_inputs.dtype,
                                                                 device=token_inputs.device))

                    # Store raw TS values
                    ts_inputs = self.extend_tensor(ts_inputs, ts[prev_t:article_day])

                    # Modality and position tracking
                    mods.extend([self.mod2id('TS')] * num_ts_points)
                    positions.extend(range(prev_t, article_day))

                    prev_mod = 'TS'

                prev_t = article_day

                # Add modality separator before text if switching from TS
                if prev_mod != 'TEXT':
                    z = self.mod_sep_emb[None, :]
                    st = torch.full((1,), -100, dtype=tokens.dtype, device=tokens.device)
                    token_embeddings = torch.cat([z, token_embeddings, z], dim=0)
                    tokens = torch.cat([st, tokens, st], dim=0)
                    prev_mod = 'TEXT'

                # Add EOS token at the end
                if self.params['learnable_eos_token'] and j == (self.num_hist-1):
                    z = self.eos_token_emb[None, :]
                    st = torch.full((1,), -100, dtype=tokens.dtype, device=tokens.device)
                    token_embeddings = torch.cat([token_embeddings, z], dim=0)
                    tokens = torch.cat([tokens, st], dim=0)

                # Add text tokens
                embeddings = self.extend_tensor(embeddings, token_embeddings)
                token_inputs = self.extend_tensor(token_inputs, tokens)
                ts_inputs = self.extend_tensor(ts_inputs,
                                               torch.full((tokens.size(0),), -100,
                                                          dtype=ts_inputs.dtype,
                                                          device=ts_inputs.device))
                mods.extend([self.mod2id('TEXT')] * len(token_embeddings))
                positions.extend([article_day] * len(token_embeddings))

            # Convert to tensors
            positions = torch.tensor(positions, dtype=torch.long, device=self.device)
            mods = torch.tensor(mods, dtype=torch.long, device=self.device)

            assert len(embeddings) == len(positions) == len(mods)
            assert len(token_inputs) == len(ts_inputs) == len(embeddings)
            assert (positions[1:] >= positions[:-1]).all(), "Time positions not in ascending order"

            # Encode TS values and replace placeholder embeddings
            ts_mask = (mods == self.mod2id('TS'))
            if ts_mask.sum() > 0:
                ts_values = embeddings[ts_mask]  # Get placeholder values (which are the raw TS values)
                # Get actual raw values from ts_inputs
                ts_raw = ts_inputs[ts_mask]
                ts_embeddings, ts_tokens = self.ts_model(ts_raw)
                embeddings[ts_mask] = ts_embeddings

                all_ts_tokens.append(ts_tokens)
            else:
                all_ts_tokens.append(torch.tensor([], dtype=torch.long, device=self.device))

            all_embeddings.append(embeddings)
            modality_ids.append(mods)
            all_token_inputs.append(token_inputs)
            all_ts_inputs.append(ts_inputs)

        inputs_embeds, attention_mask = self.pad_var_seq(all_embeddings)
        modality_ids, _ = self.pad_var_seq(modality_ids)
        token_ids, _ = self.pad_var_seq(all_token_inputs, padding_value=-100)
        ts_inputs, _ = self.pad_var_seq(all_ts_inputs, padding_value=0.00)
        ts_tokens, _ = self.pad_var_seq(all_ts_tokens, padding_value=-100)

        outputs = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask, modality_ids=modality_ids,
                       token_ids=token_ids, ts_inputs=ts_inputs, ts_tokens=ts_tokens)

        return outputs


    def pad_var_seq(self, embeddings, padding_value=0.00):

            """
            embeddings [List][Tensor = [L, D]]
            Note: Embeddings do not have a first batch dimension, the pad_sequence function will insert one
            """

            seq_lens = [e.size(0) for e in embeddings]
            padded_embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True,
                                                                padding_value=padding_value)
            max_len = padded_embeddings.size(1)
            attention_mask = [[1] * sl + [0] * (max_len - sl) for sl in seq_lens]

            attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=padded_embeddings.device)

            assert padded_embeddings.size(0) == attention_mask.size(0)
            assert padded_embeddings.size(1) == attention_mask.size(1)

            return padded_embeddings, attention_mask


    def token_pooling(self, h, attention_mask):

        """Extract the (last) [EOS] Token for prediction head"""

        is_padding_right = (attention_mask[:, -1].sum().item() != attention_mask.size(0))
        if is_padding_right:
            last_token_index = attention_mask.sum(dim=1) - 1
            bi = torch.arange(0, h.size(0))
            h = h[bi, last_token_index, :]
        else:
            h = h[:, -1, :]

        return h


    def forward(self, batch):

        inputs = self.create_interleaved_sequence(batch)
        with torch.set_grad_enabled(self.training):
            h = self.bert(inputs_embeds=inputs['inputs_embeds'], attention_mask=inputs['attention_mask'],
                          modality_ids=inputs['modality_ids'], use_cache=False)[0]

        if self.task_type == 'CLASS':
            h = self.token_pooling(h, inputs['attention_mask'])
            logits = self.classifier(h)
            labels = batch['labels']
            loss = F.cross_entropy(logits, labels, reduction='mean')
        elif self.task_type == 'SALMON':
            loss, text_loss, ts_loss = self.salmon_head(hidden_states=h, attention_mask=inputs['attention_mask'],
                                                    modality_ids=inputs['modality_ids'],
                                                    input_ids=inputs['token_ids'], ts_tokens=inputs['ts_tokens'],
                                                    inputs_embeds=inputs['inputs_embeds'], ts_inputs=inputs['ts_inputs'])

        return loss

