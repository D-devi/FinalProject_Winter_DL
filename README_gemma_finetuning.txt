# Gemma 3 Fine-Tuning for Medical VQA

This code fine-tunes the Gemma 3 4B model on a medical image question-answering dataset (VQA-RAD) for improved performance on medical image analysis tasks.

## Overview

The script performs the following:
- Loads and quantizes Gemma 3 4B model using 4-bit precision
- Prepares a medical VQA dataset with images and descriptive answers
- Applies LoRA (Low-Rank Adaptation) fine-tuning to adapt the model
- Evaluates model performance before and after fine-tuning
- Saves the fine-tuned model for later use

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- PEFT library for parameter-efficient fine-tuning
- TRL library for SFT training
- BitsAndBytes for quantization
- Hugging Face account with access to Gemma 3 models
- GPU with at least 16GB VRAM (preferably 24GB+)
- CUDA-capable GPU with compute capability 7.0+

## Setup

1. Install required packages:
   ```
   pip install transformers peft trl bitsandbytes accelerate evaluate rouge_score
   ```

2. Authentication:
   - You'll need a Hugging Face token with access to Gemma 3 models
   - Replace "hf_token" with your own token

3. Data:
   - Make sure your enhanced dataset files are available:
     - vqa_rad_gemini_train.json
     - vqa_rad_gemini_test.json
   - These should contain image paths and QA pairs for medical images
   - Make sure the image folder 'md5_images' with all the images is available

## Running the Code

1. Basic execution:
   ```
   Run the notebook
   ```

2. Adjusting parameters:
   - Edit these variables at the top of the script if needed:
     - Number of epochs
     - Batch size
     - Learning rate
     - Output directory

## Model Training Process

1. The script first evaluates the base model to establish a baseline
2. Model is trained with the chosen parameters and saved to output directory
3. The model is re-evaluated by printing evaluation metrics after each training run to show improvement

## Output

- Training logs showing progress
- Evaluation metrics before/after fine-tuning:
  - BLEU
  - ROUGE-1/2/L
  - BERTScore
  - Exact match accuracy
- Saved model checkpoints in directories:
  - gemma-md5-full_run/

## Fine-Tuning Details

- Both the embedding matrix and LM head are fully fine-tuned
- LoRA is applied to linear layers for efficient fine-tuning
- Training hyperparameters:
  - LoRA rank: 16
  - LoRA alpha: 16
  - Learning rates: 1e-4
  - Gradient accumulation steps: 32

## Notes and Limitations

- The code uses BF16 precision which requires compatible GPU hardware
- Memory errors may occur on GPUs with less than 16GB VRAM
- Training time depends on dataset size and GPU capability
- Gradient checkpointing is enabled to reduce memory usage
- For best results, run on an A100, H100, or RTX 4090 GPU 