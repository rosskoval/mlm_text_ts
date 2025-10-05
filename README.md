# Multimodal Language Models with Modality-Specific Experts for Financial Forecasting from Interleaved Sequences of Text and Time Series

Please cite our paper if you use the code.

## Introduction
This repository contains the main code used in our work: "Multimodal Language Models with Modality-Specific Experts for Financial Forecasting from Interleaved Sequences of Text and Time Seriess" by Ross Koval, Nicholas Andrews, and Xifeng Yan. 

## Citation
If you find this work or code useful, please cite the paper:

```bibtex
@misc{koval2025multimodallanguagemodelsmodalityspecific,
      title={Multimodal Language Models with Modality-Specific Experts for Financial Forecasting from Interleaved Sequences of Text and Time Series}, 
      author={Ross Koval and Nicholas Andrews and Xifeng Yan},
      year={2025},
      eprint={2509.19628},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2509.19628}, 
}
```

## Abstract
Text and time series data offer complementary views of financial markets: news articles provide narrative context about company events, while stock prices reflect how markets react to those events. However, despite their complementary nature, effectively integrating these interleaved modalities for improved forecasting remains challenging. In this work, we propose a unified neural architecture that models these interleaved sequences using modality-specific experts, allowing the model to learn unique time series patterns, while still enabling joint reasoning across modalities and preserving pretrained language understanding capabilities. To further improve multimodal understanding, we introduce a cross-modal alignment framework with a salient token weighting mechanism that learns to align representations across modalities with a focus on the most informative tokens. We demonstrate the effectiveness of our approach on a large-scale financial forecasting task, achieving state-of-the-art performance across a wide variety of strong unimodal and multimodal baselines. We develop an interpretability method that reveals insights into the value of time series-context and reinforces the design of our cross-modal alignment objective. Finally, we demonstrate that these improvements translate to meaningful economic gains in investment simulations.

## Model Architecture - Modality-Specific Experts for Interleaved Sequences of Text and Time-Series (MSE-ITT)

<p align="center">
<img width="600" height="600" alt="model_diagram" src="https://github.com/user-attachments/assets/5ccee1a6-f579-45c5-b34e-6b6a81ee1d8a" />
</p>

*Figure 1:Overview of our multimodal forecasting task and proposed model (MSE-ITT), which processes interleaved sequences of tokens of news articles (Text) and quantized stock returns (TS). MSE-ITT incorporates modality-specific experts to capture distinct patterns in text and time series, while enabling joint reasoning across modalities through selective cross-modal attention.*

## Cross-Modal Alignment - SALMON

<p align="center">
<img width="600" height="600" alt="cma_diagram" src="https://github.com/user-attachments/assets/207e4f20-69d2-41f0-b2ea-ef59e78b968c" />
</p>

*Figure 2: Illustration of our cross-modal alignment framework, SALMON, which learns to align historical stock price behavior and news articles with a unified objective. The Salient Token Weighting (STW) mechanism dynamically assigns higher weight to tokens that benefit most from time-series context, improving cross-modal alignment.*

## Case Study 

<p align="center">
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/8afb7794-18d4-4a26-aa04-4adee6673f1a" />
</p>

*Figure 3: Example sequence of news events and market reactions about the potential Pfizer-Allergan merger. The news articles contain persuasive language, yet the accompanying stock returns reveal the market’s true perception: optimism on early rumors, a sharp sell-off once the deal terms are set, and a relief rally when the takeover is cancelled. Jointly modeling these inputs can more effectively interpret the outcome of such events.*
