# VQA-RAD Gemini Enhancement

This repository contains code to enhance the VQA-RAD (Visual Question Answering for Radiology) dataset by generating descriptive answers using the Google Gemini AI model.

## Requirements

- Python 3.8+
- Required Python packages:
  ```
  datasets
  pillow
  numpy
  google-generativeai
  tqdm
  ```

## Setup

1. **Install required packages:**
   ```bash
   pip install datasets pillow numpy google-generativeai tqdm
   ```

2. **Google Gemini API Keys:**
   - Obtain Gemini API keys from the [Google AI Studio](https://aistudio.google.com/)
   - Add your API keys to the `api_keys` list in the script:
     ```python
     api_keys = [
         "YOUR_API_KEY_1",
         "YOUR_API_KEY_2",
         # Add more keys for better throughput
     ]
     ```

3. **Directory Structure:**
   - The script will automatically create:
     - `md5_images/` directory for storing images with MD5 hash filenames
     - `md5_train/` directory for storing output JSON files

## Running the Code

1. **Basic Execution:**
   ```bash
   python vqa_rad_gemini_batch1.py
   ```

2. **What the Script Does:**
   - Loads the VQA-RAD dataset from HuggingFace
   - For the chosen 'split': Train or Test dataset (Line 36)
   - Processes images and assigns MD5 hashes
   - Saves images to disk with hash-based filenames
   - Generates descriptive answers for each Q&A pair using Gemini
   - Saves the enhanced dataset as a JSON file

3. **Output:**
   - Main output: `md5_train/vqa_rad_gemini_batch1_train.json`
   - Temporary backup: `md5_train/vqa_rad_gemini_batch1_train_temp.json`
   - Saved images: `md5_images/image_md5_[HASH].png`

## Rate Limiting and API Key Rotation

The script implements API key rotation and rate limit handling to efficiently use multiple Gemini API keys:

- Automatically switches between keys based on usage
- Respects the 60 requests per minute rate limit
- Implements exponential backoff for retries
- Tracks errors per key and rotates if needed

## Output Structure

The enhanced dataset JSON has the following structure:

```json
{
  "image_hash": {
    "image_info": {
      "size": [width, height],
      "split": "train"
    },
    "image_path": "md5_images/image_md5_[HASH].png",
    "qa_pairs": [
      {
        "question": "Original question",
        "original_answer": "Original short answer",
        "descriptive_answer": "Enhanced answer with explanation",
        "split": "train"
      },
      ...
    ]
  },
  ...
}
```

## Troubleshooting

- **API Rate Limits**: If you encounter frequent rate limiting, add more API keys
- **Memory Issues**: Process the dataset in smaller batches by adjusting `BATCH_SIZE`
- **Interrupted Execution**: The script saves progress regularly, so you can continue by modifying the code to load the last saved JSON file

## Notes

- Processing time will vary based on your number of API keys and dataset size
- The script has built-in cooldown periods between images for API stability
- Progress updates and ETAs are displayed during execution 