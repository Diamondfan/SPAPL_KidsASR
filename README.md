# Benchmarking Children's ASR with Supervised and Self-supervised Speech Foundation Models

## Overview

This github repository contains models, scripts and data splits from our paper accepted at Interspeech 2024, which can be found [here](https://arxiv.org/abs/2406.10507)

## Folder Structure

Source code for training different Supervised and Self Supervised models can be found under /src

/egs contains bash scripts to train models on the MyST and CSLU OGI Kids' datasets, as well as scripts to filter these datasets and obtain train test splits

## Getting Started

1. **Install Dependencies**: transformers==4.32.1 torch evaluate datasets
2. On older versions of transformers it might be necessary to make minor edits to trainer.py to allow hotloading of Iterable datasets (if streaming is set to True). Follow the instructions in /egs/MyST/README.txt to make the necessary edits.
3. For training Nemo based models, it is necessary to clone [the github repo](https://github.com/NVIDIA/NeMo)
4. To train/evaluate a model a particular dataset, edit the corresponding yaml file stored in the /egs/dataset/config directory, and edit the necessary bash script.

# Trained Models

## MyST Models - Fully Finetuned

|      Model      | MyST test WER |                             Huggingface Link                             |
| :--------------: | :-----------: | :-----------------------------------------------------------------------: |
|   Whisper tiny   |     11.6     |  [model](https://huggingface.co/balaji1312/whisper-tiny-myst-fullfinetune)  |
|   Whisper base   |     10.4     |  [model](https://huggingface.co/balaji1312/whisper-base-myst-fullfinetune)  |
|  Whisper small  |      9.3      |  [model](https://huggingface.co/balaji1312/whisper-small-myst-fullfinetune)  |
|  Whisper Medium  |      8.9      | [model](https://huggingface.co/balaji1312/whisper-medium-myst-fullfinetune) |
|  Whisper Large  |     13.0     |  [model](https://huggingface.co/balaji1312/whisper-large-myst-fullfinetune)  |
| Whisper Large v3 |      9.1      | [model](https://huggingface.co/balaji1312/whisper-largev3-myst-fullfinetune) |
|      Canary      |      9.2      |     [model](https://huggingface.co/balaji1312/canary-myst-fullfinetune)      |
|     Parakeet     | **8.5** |    [model](https://huggingface.co/balaji1312/parakeet-myst-fullfinetune)     |
| Wav2vec2.0 Large |     11.1     | [model](https://huggingface.co/balaji1312/wav2vec2-large-myst-fullfinetune) |
|   HuBERT Large   |     11.3     |  [model](https://huggingface.co/balaji1312/hubert-large-myst-fullfinetune)  |
|   WavLM Large   |     10.4     |   [model](https://huggingface.co/balaji1312/wavlm-large-myst-fullfinetune)   |

## OGI Models - Fully Finetuned

|      Model      | OGI test WER |                             Huggingface Link                             |
| :--------------: | :-----------: | :----------------------------------------------------------------------: |
|   Whisper tiny   |      3.0      |  [model](https://huggingface.co/balaji1312/whisper-tiny-ogi-fullfinetune)  |
|   Whisper base   |      2.3      |  [model](https://huggingface.co/balaji1312/whisper-base-ogi-fullfinetune)  |
|  Whisper small  |      1.8      |  [model](https://huggingface.co/balaji1312/whisper-small-ogi-fullfinetune)  |
|  Whisper Medium  |      1.5      | [model](https://huggingface.co/balaji1312/whisper-medium-ogi-fullfinetune) |
|  Whisper Large  |      1.7      |  [model](https://huggingface.co/balaji1312/whisper-large-ogi-fullfinetune)  |
| Whisper Large v3 | **1.4** | [model](https://huggingface.co/balaji1312/whisper-largev3-ogi-fullfinetune) |
|      Canary      |      1.5      |     [model](https://huggingface.co/balaji1312/canary-ogi-fullfinetune)      |
|     Parakeet     |      1.8      |    [model](https://huggingface.co/balaji1312/parakeet-ogi-fullfinetune)     |
| Wav2vec2.0 Large |      2.5      | [model](https://huggingface.co/balaji1312/wav2vec2-large-ogi-fullfinetune) |
|   HuBERT Large   |      2.5      |  [model](https://huggingface.co/balaji1312/hubert-large-ogi-fullfinetune)  |
|   WavLM Large   |      1.8      |   [model](https://huggingface.co/balaji1312/wavlm-large-ogi-fullfinetune)   |

## MyST Models - Whisper small with Data Augmentations

| Data Augmentation | Myst test WER |                               Huggingface Link                               |
| :---------------: | :-----------: | :--------------------------------------------------------------------------: |
|        PP        | **8.8** |  [model](https://huggingface.co/balaji1312/whisper-small-myst-fullfinetune-pp)  |
|       VTLP       |      9.0      | [model](https://huggingface.co/balaji1312/whisper-small-myst-fullfinetune-vtlp) |
|        SP        |      8.9      |  [model](https://huggingface.co/balaji1312/whisper-small-myst-fullfinetune-sp)  |
|        SA        |      9.0      |  [model](https://huggingface.co/balaji1312/whisper-small-myst-fullfinetune-sa)  |

## MyST Models - Whisper small with PEFT

| PEFT Method | MyST test WER |                          Huggingface Link                          |
| :---------: | :-----------: | :----------------------------------------------------------------: |
|     Enc     | **9.2** |   [model](https://huggingface.co/balaji1312/whisper-small-myst-enc)   |
|     Dec     |      9.5      |   [model](https://huggingface.co/balaji1312/whisper-small-myst-dec)   |
|    LoRA    |      9.6      |  [model](https://huggingface.co/balaji1312/whisper-small-myst-lora)  |
|   Prompt   |     10.4     | [model](https://huggingface.co/balaji1312/whisper-small-myst-prompt) |
|   Prefix   |     10.2     | [model](https://huggingface.co/balaji1312/whisper-small-myst-prefix) |
|   Adapter   |      9.3      | [model](https://huggingface.co/balaji1312/whisper-small-myst-adapter) |

## Citation

If you use this code in your research, please cite it as follows:

```{bibtex}
@article{fan2024benchmarking,   
title={Benchmarking Children's ASR with Supervised and Self-supervised Speech Foundation Models},   
author={Fan, Ruchao and Shankar, Natarajan Balaji and Alwan, Abeer},   
journal={arXiv preprint arXiv:2406.10507},   
year={2024} 
}
```
