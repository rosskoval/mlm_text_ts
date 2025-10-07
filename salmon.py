import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import math
from collections import defaultdict
from mse_itt import TSEncoder, MSE_ITT_Model

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

class SALMONHead(nn.Module):
    def __init__(self, params, model, config, tokenizer):
        """
        SALMON Head for computing unified CE losses for next token predictions from interleaved sequences of text and discretized (ts) series.
        """
        super().__init__()
        self.params = params
        self.model = model
        self.text_modality_id = 0
        self.ts_modality_id = 1
        self.IGNORE_INDEX = -100
        self.ts_vocab_size = params.get('ts_vocab_size', 32)
        self.ts_loss_weight = params.get('ts_loss_weight', 1.0)
        self.mod_switch_loss = params.get('mod_switch_loss', True)
        self.min_weight = params.get('min_token_weight_ratio', 0.1)
        self.max_weight = params.get('max_token_weight_ratio', 10.0)
        self.use_stw_loss = params.get('use_stw_loss', True)

        total_steps = int(params.get('total_train_steps', 0))
        warm_frac = float(params.get('stw_warmup_fraction', 0.20))
        self.stw_warmup_steps = int(round(total_steps * warm_frac)) if total_steps > 0 else 0
        self.global_step = 0

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.ts_head = nn.Linear(config.hidden_size, self.ts_vocab_size)

        self.tokenizer = tokenizer
        self.special_token_ids = self.tokenizer.all_special_ids
        self.text_loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=self.IGNORE_INDEX)
        self.ts_loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=self.IGNORE_INDEX)


    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state['global_step'] = self.global_step
        return state

    def load_state_dict(self, state_dict, strict=True):
        if 'global_step' in state_dict:
            self.global_step = state_dict.pop('global_step')
        return super().load_state_dict(state_dict, strict)


    def compute_stw(self, text_mask, inputs_embeds, attention_mask, modality_ids, text_states,
                                          text_ids, text_loss):
        """
        Compute STW weights to identify salient tokens that benefit most from TS-context by performing two model forward passes:
        (1) with full multimodal inputs
        (2) with TS-context masked out for the baseline contrast
        """

        with torch.no_grad():
            text_only_attention_mask = attention_mask.clone()
            ts_token_mask = (modality_ids == self.ts_modality_id)
            text_only_attention_mask[ts_token_mask] = 0

            # Run model with text-only inputs to get contrast predictions by masking out TS context but maintaining the same positions
            contrast_hidden_states = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=text_only_attention_mask,
                modality_ids=modality_ids  # Keep original modality_ids for position reference
            )[0]

            # Extract predictions for text tokens
            contrast_flat_states = contrast_hidden_states[:, :-1].reshape(-1, contrast_hidden_states.size(-1))
            contrast_text_states = contrast_flat_states[text_mask]
            contrast_text_logits = self.lm_head(contrast_text_states)
            contrast_loss = self.text_loss_fn(contrast_text_logits, text_ids)

        # Compute likelihood ratio from negative logprobs: P(x)/P(y) = exp( -log(P(y)) - -log(P(x)) )
        loss_diff = torch.exp(contrast_loss - text_loss)
        weights = torch.clamp(loss_diff, min=self.min_weight, max=self.max_weight)
        # Normalize weights such that each sequence has mean one
        norm_weights = torch.zeros_like(weights)
        B, S = attention_mask.shape
        flat_batch_idx = torch.arange(B, device=attention_mask.device).unsqueeze(1).expand(B, S-1).reshape(-1)
        text_batch_idx = flat_batch_idx[text_mask]
        for b in text_batch_idx.unique():
            sel = (text_batch_idx == b)
            if sel.any():
                z = weights[sel]
                norm_weights[sel] = z / (z.mean() + 1e-8)

        weighted_loss = torch.mean(text_loss * norm_weights)

        return weighted_loss, weights


    def forward(self, hidden_states, attention_mask, inputs_embeds, modality_ids, input_ids, ts_inputs, ts_tokens):
        """
        Compute SALMON losses for interleaved text and TS token sequences
        """
        batch_size, seq_length, hidden_size = hidden_states.size()

        mod_switch_mask = (modality_ids[:, :-1] != modality_ids[:, 1:])
        # Flatten the hidden states, modality IDs, and attention mask to combine the batch and sequence dimensions.
        flat_states = hidden_states[:, :-1].reshape(-1, hidden_size)  # [batch_size * seq_length, hidden_size]
        # Shift all output sequences by 1 since these are what we are predicting from the inputs
        flat_modality_ids = modality_ids[:, 1:].reshape(-1)  # [batch_size * seq_length]
        flat_attention_mask = attention_mask[:, 1:].reshape(-1).bool()  # [batch_size * seq_length]
        flat_input_ids = input_ids[:, 1:].reshape(-1)
        flat_ts_inputs = ts_inputs[:, 1:].reshape(-1)
        flat_ts_tokens = ts_tokens[:, 1:].reshape(-1)
        flat_mod_switch_mask = mod_switch_mask.reshape(-1)

        losses = {}

        # Process Text Tokens - create a mask that selects tokens that are both valid (non-padding) and of text modality.
        non_special_mask = ~torch.isin(flat_input_ids,
                                       torch.tensor(self.special_token_ids, device=flat_input_ids.device))
        valid_tokens_mask = (flat_input_ids != self.IGNORE_INDEX) & non_special_mask
        text_mask = (flat_modality_ids == self.text_modality_id) & flat_attention_mask & valid_tokens_mask
        if not self.mod_switch_loss:
            text_mask = (text_mask & ~flat_mod_switch_mask)
        if text_mask.sum() > 0:
            text_states = flat_states[text_mask] # [num_text_tokens, hidden_size]
            text_ids = flat_input_ids[text_mask]
            text_logits = self.lm_head(text_states) # [num_text_tokens, vocab_size]
            text_loss = self.text_loss_fn(text_logits, text_ids)

            if self.global_step < self.stw_warmup_steps or not self.use_stw_loss:
                text_loss = torch.mean(text_loss)
            elif self.use_stw_loss and self.global_step >= self.stw_warmup_steps:
                text_loss, weights = self.compute_stw(text_mask, inputs_embeds, attention_mask, modality_ids, text_states, text_ids, text_loss)
            losses['text_loss'] = text_loss
        else:
            losses['text_loss'] = torch.tensor(0.0, device=flat_input_ids.device)

        # Process TS Tokens - select tokens that are valid and belong to the TS modality
        ts_valid_mask = (flat_ts_inputs != self.IGNORE_INDEX)
        ts_mask = (flat_modality_ids == self.ts_modality_id) & flat_attention_mask & ts_valid_mask
        if not self.mod_switch_loss:
            ts_mask = (ts_mask & ~flat_mod_switch_mask)
        if ts_mask.sum() > 0:
            ts_states = flat_states[ts_mask]
            ts_logits = self.ts_head(ts_states)
            ts_ids = flat_ts_tokens[ts_mask]
            ts_loss = self.ts_loss_fn(ts_logits, ts_ids).mean()
            losses[f'ts_loss'] = ts_loss
        else:
            losses[f'ts_loss']= torch.tensor(0.0, device=flat_ts_inputs.device)

        total_loss = losses['text_loss'] + self.ts_loss_weight * losses['ts_loss']

        if self.training:
            self.global_step += 1

        return total_loss, losses['text_loss'], losses['ts_loss']

