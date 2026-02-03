# Beetles Drought Prediction Challenge

A machine learning pipeline for predicting drought conditions (SPEI metrics) from carabid beetle specimen images.

## What is this project?

This project predicts drought conditions from beetle images for the **HDR Scientific-Mood ML Challenge**. The key insight is that beetle populations respond to environmental conditions, so their appearance may contain signals about local climate.

We predict three drought metrics:
- **SPEI_30d**: 30-day Standardized Precipitation Evapotranspiration Index
- **SPEI_1y**: 1-year SPEI
- **SPEI_2y**: 2-year SPEI

The model outputs both predictions AND uncertainty estimates, which are evaluated using the Continuous Ranked Probability Score (CRPS).

## Architecture Overview

```
[Beetle Images] → [DINOv3 Backbone] → [Embeddings] → [Mean Pooling] → [MLP Head] → [μ, σ predictions]
     (frozen)                                                         (trained)
```

**Why this approach?**
1. **Frozen backbone**: DINOv3 is pretrained on millions of images - we leverage this knowledge
2. **Pre-computed embeddings**: Extract features once, then training is super fast
3. **Simple MLP head**: Only ~200K parameters to train - less prone to overfitting
4. **Uncertainty estimation**: Using Gaussian NLL loss, the model learns when to be confident

## Quick Start

### 1. Set up the environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the data

You'll need a HuggingFace account and API token. Get one at: https://huggingface.co/settings/tokens

```bash
# Set your token as an environment variable
set HF_TOKEN=your_token_here  # Windows
# export HF_TOKEN=your_token_here  # Linux/Mac

# Download the dataset
python scripts/download_data.py
```

### 3. Extract embeddings

This step pre-computes image features using DINOv3. It only needs to run once!

```bash
python scripts/extract_embeddings.py
```

This will take some time depending on your hardware, but it makes subsequent training very fast.

### 4. Train the model

```bash
python scripts/train.py
```

Default training runs for up to 200 epochs with early stopping.

### 5. Evaluate

```bash
python scripts/evaluate.py
```

## Monitoring Training Progress

We use TensorBoard for experiment tracking:

```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir logs

# Open http://localhost:6006 in your browser
```

You'll see:
- Training and validation loss curves
- CRPS score over time
- Per-target metrics (SPEI_30d, SPEI_1y, SPEI_2y)
- Learning rate schedule

## Project Structure

```
SMood-ML-Challenge/
├── config.py                 # All hyperparameters and settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Downloaded data (gitignored)
├── embeddings/               # Pre-computed features (gitignored)
├── checkpoints/              # Saved model weights
├── logs/                     # TensorBoard logs
├── src/
│   ├── data_loader.py       # Dataset classes
│   ├── feature_extractor.py # DINOv3 embedding extraction
│   ├── model.py             # MLP regression head
│   ├── trainer.py           # Training loop
│   ├── loss.py              # Gaussian NLL loss
│   ├── metrics.py           # CRPS calculation
│   └── utils.py             # Helper functions
├── scripts/
│   ├── download_data.py     # Download from HuggingFace
│   ├── extract_embeddings.py # Pre-compute features
│   ├── train.py             # Main training script
│   └── evaluate.py          # Evaluation script
└── submission/
    ├── model.py             # Competition submission format
    └── requirements.txt     # Submission dependencies
```

## Key Hyperparameters

Edit `config.py` to change these settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 1e-3 | Learning rate for Adam optimizer |
| `BATCH_SIZE` | 64 | Training batch size |
| `NUM_EPOCHS` | 200 | Maximum training epochs |
| `HIDDEN_DIM` | 256 | Hidden layer size in MLP |
| `EARLY_STOPPING_PATIENCE` | 20 | Stop if no improvement for N epochs |
| `VALIDATION_DOMAIN_IDS` | [32, 99] | Sites held out for validation |

**Tuning tips:**
- If validation loss is much higher than training loss → model is overfitting → try smaller `HIDDEN_DIM` or more dropout
- If both losses are high → model is underfitting → try larger `HIDDEN_DIM` or deeper model
- If training is unstable → try smaller `LEARNING_RATE`

## How the Model Works

### 1. Feature Extraction (DINOv3)

DINOv3 is a self-supervised vision model from Meta AI that learns powerful image representations without labels. We use the ViT-B/16 variant which:
- Processes 224x224 images
- Outputs 768-dimensional embedding vectors
- Was trained on 1.7 billion images

### 2. Aggregation (Mean Pooling)

Each sampling event has multiple beetle images. We average all embeddings for an event:

```
event_embedding = mean([embed_1, embed_2, ..., embed_n])
```

### 3. Regression Head (MLP)

A simple 2-layer neural network:

```
[768] → Linear → ReLU → Dropout → [256] → Linear → [6 outputs]
                                            ↓
                                    [3 mu, 3 sigma]
```

The sigma outputs use softplus activation to ensure they're positive.

### 4. Loss Function (Gaussian NLL)

We train using Gaussian Negative Log-Likelihood:

```
loss = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
```

This encourages the model to:
- Predict accurate values (minimize the squared error term)
- Know when it's uncertain (larger σ when predictions are hard)

### 5. Evaluation (CRPS)

CRPS measures both accuracy and uncertainty quality:

```
CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
where z = (y - μ) / σ
```

Lower is better. The competition uses RMS of CRPS across all predictions.

## Creating a Competition Submission

After training, create your submission:

```bash
# 1. Copy the trained weights
copy checkpoints\best_model.pt submission\model.pth

# 2. The submission folder should contain:
#    - model.py (already there)
#    - model.pth (your trained weights)
#    - requirements.txt (already there)

# 3. Create tarball for submission
cd submission
tar -cvf submission.tar model.py model.pth requirements.txt
```

Upload `submission.tar` to Codabench.

## Troubleshooting

### "CUDA out of memory"

Reduce batch size:
```bash
python scripts/extract_embeddings.py --batch_size 8
python scripts/train.py --batch_size 32
```

### "DINOv3 not available"

The model will automatically fall back to DINOv2. This is fine - DINOv2 is also excellent.

### "No HuggingFace token found"

Set your token:
```bash
set HF_TOKEN=your_token_here  # Windows
export HF_TOKEN=your_token_here  # Linux/Mac
```

### Training loss not decreasing

- Try a smaller learning rate: `--lr 0.0001`
- Check your data was loaded correctly

## Further Improvements to Try

1. **Different aggregation**: Try `--aggregation max` or attention-based pooling
2. **Deeper model**: Use `--model_type deep` for more capacity
3. **Different backbone**: Edit `config.py` to use DINOv2-large
4. **Data augmentation**: Add augmentations during embedding extraction
5. **Ensemble**: Train multiple models and average predictions

## References

- [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [HDR-SMood Challenge](https://github.com/Imageomics/HDR-SMood-challenge)
- [CRPS Score](https://en.wikipedia.org/wiki/Continuous_ranked_probability_score)

## License

This project is for educational/competition purposes. See the competition rules for data usage terms.
