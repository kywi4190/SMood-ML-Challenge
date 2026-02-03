"""
Competition submission model for the Beetles Drought Prediction Challenge.

This file follows the required Model class interface for the competition.
It uses DINOv3 for feature extraction and a trained regression head for predictions.

To submit:
1. Copy this file to your submission folder
2. Include model.pth (trained regression head weights)
3. Include requirements.txt
4. Create tarball and submit to Codabench
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class RegressionHead(nn.Module):
    """
    Regression head that predicts SPEI values with uncertainty.

    This is a copy of the model architecture from src/model.py.
    We include it here to make the submission self-contained.
    """

    def __init__(
        self,
        embedding_dim=768,
        hidden_dim=256,
        num_targets=3,
        min_sigma=0.01,
        dropout=0.1
    ):
        super().__init__()
        self.num_targets = num_targets
        self.min_sigma = min_sigma

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_targets * 2)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)

        mu = output[:, :self.num_targets]
        sigma_raw = output[:, self.num_targets:]
        sigma = F.softplus(sigma_raw) + self.min_sigma

        return mu, sigma


class Model:
    """
    Main model class for the competition.

    This class implements the required interface:
    - load(): Load model weights and preprocessing
    - predict(datapoints): Make predictions on a batch of images

    The model uses:
    1. DINOv3 backbone for feature extraction (frozen)
    2. Trained regression head for SPEI prediction
    """

    def __init__(self):
        """Initialize model attributes."""
        self.model = None
        self.backbone = None
        self.processor = None
        self.device = None

    def load(self):
        """
        Load model weights and preprocessing transforms.

        This method is called once at startup to initialize everything.
        """
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load DINOv3 backbone from HuggingFace
        print("Loading DINOv3 backbone...")
        self._load_backbone()

        # Load trained regression head
        print("Loading regression head...")
        self._load_regressor()

        print("Model loaded successfully!")

    def _load_backbone(self):
        """Load the DINOv3 feature extraction backbone."""
        from transformers import AutoModel, AutoImageProcessor

        # Model name - DINOv3 base variant
        model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
        except Exception as e:
            # Fallback to DINOv2 if DINOv3 not available
            print(f"DINOv3 not available, falling back to DINOv2: {e}")
            model_name = "facebook/dinov2-base"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)

        # Set to evaluation mode and move to device
        self.backbone.eval()
        self.backbone.to(self.device)

        # Freeze backbone (no gradient computation needed)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _load_regressor(self):
        """Load the trained regression head."""
        # Create the regression head architecture
        self.model = RegressionHead(
            embedding_dim=768,
            hidden_dim=256,
            num_targets=3,
            min_sigma=0.01
        )

        # Load trained weights
        weights_path = Path(__file__).parent / "model.pth"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {weights_path}")
        else:
            print(f"WARNING: No weights found at {weights_path}")
            print("Model will use random weights!")

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, datapoints):
        """
        Make predictions on a batch of images.

        This method is called by the competition framework for each
        sampling event. All images in datapoints belong to the same event.

        Args:
            datapoints: List of dicts, each containing:
                - "relative_img": PIL Image object
                - "domainID": Integer domain identifier
                - Other metadata fields

        Returns:
            dict: Predictions in the required format:
            {
                "SPEI_30d": {"mu": float, "sigma": float},
                "SPEI_1y": {"mu": float, "sigma": float},
                "SPEI_2y": {"mu": float, "sigma": float},
            }
        """
        # Extract images from datapoints
        images = [entry["relative_img"] for entry in datapoints]

        # Convert images to RGB if needed
        rgb_images = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            rgb_images.append(img)

        # Preprocess images
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        # Extract features using backbone
        # Process in smaller batches if needed
        batch_size = 8
        all_embeddings = []

        for i in range(0, len(pixel_values), batch_size):
            batch = pixel_values[i:i + batch_size]
            outputs = self.backbone(batch)

            # Get the [CLS] token embedding
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0]

            all_embeddings.append(embeddings)

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Aggregate embeddings (mean pooling across all images in the event)
        aggregated_embedding = all_embeddings.mean(dim=0, keepdim=True)

        # Get predictions from regression head
        mu, sigma = self.model(aggregated_embedding)

        # Convert to numpy for output
        mu = mu.cpu().numpy()[0]
        sigma = sigma.cpu().numpy()[0]

        # Format output
        predictions = {
            "SPEI_30d": {"mu": float(mu[0]), "sigma": float(sigma[0])},
            "SPEI_1y": {"mu": float(mu[1]), "sigma": float(sigma[1])},
            "SPEI_2y": {"mu": float(mu[2]), "sigma": float(sigma[2])},
        }

        return predictions


# For testing
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    print("Testing Model class...")

    # Create model
    model = Model()
    model.load()

    # Create dummy datapoints (simulating competition input)
    dummy_images = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(5)
    ]
    datapoints = [{"relative_img": img, "domainID": 1} for img in dummy_images]

    # Make prediction
    predictions = model.predict(datapoints)

    print("\nPredictions:")
    for name, values in predictions.items():
        print(f"  {name}: mu={values['mu']:.4f}, sigma={values['sigma']:.4f}")

    # Verify format
    assert "SPEI_30d" in predictions
    assert "SPEI_1y" in predictions
    assert "SPEI_2y" in predictions
    for v in predictions.values():
        assert "mu" in v
        assert "sigma" in v
        assert v["sigma"] > 0

    print("\nAll tests passed!")
