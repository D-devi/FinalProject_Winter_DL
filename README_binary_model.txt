# VQA-RAD Multimodal Pipeline README

## Overview
This project is a multimodal Visual Question Answering (VQA) pipeline built on the VQA-RAD dataset. It uses pretrained CLIP or DenseNet for image encoding and BERT or BiLSTM for question encoding, with a co-attention mechanism to fuse modalities. The goal is to predict short medical answers (from a fixed vocabulary) to radiology-related questions based on image-text pairs.



## System Requirements
Python 3.8+
PyTorch (with CUDA support recommended)
Transformers (Hugging Face) library
OpenCLIP for image encoding
GPU with at least 16GB VRAM (24GB+ preferred)
CUDA-capable GPU with compute capability 7.0+ (e.g., NVIDIA V100, A100)


## Model Components

- Image Encoder: BioClip (pretrained, optionally fine-tuned) or Dense12BilSTM
- Text Encoder: BERT-base (pretrained, optionally fine-tuned) or BiLSTM


## Setup Instructions

1. Install required packages : 
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
	pip install transformers open_clip_torch pandas datasets scikit-learn tqdm

2. Download VQA_RAD image Folder.zip and VQA_RAD Datatset Public.json
3. Run each file that ends with BinaryModel


## Evaluation Metrics

- **F1, acc, auroc


## Hyper-parameter Details

- Tune dropout, LR, image/text learning rate multipliers
- For BioClip + BERT model:
	Batch Size:        64
	Learning Rate:     1e-4
	Dropout Rate:      0.1
	Image Multiplier:  3.0
	Text Multiplier:   0.5
	LR Mult. for Heads: 64

- For Dense121 + BiLSTM model:
	Batch Size:        64
	Learning Rate:     1e-4
	Dropout Rate:      0.3
	Image Multiplier:  2.0
	Text Multiplier:   1.0
	LR Mult. for Heads: 64






