"""
Color calibration utilities for normalizing beetle images.

This module provides tools to detect color checker patches in reference images,
compute color correction matrices (CCMs), and apply those corrections to
beetle images before embedding extraction.

The goal is to normalize images so beetle coloration differences reflect actual
morphology rather than camera/lighting artifacts.

Key components:
- ColorCheckerDetector: Detects color patches using OpenCV MCC or grid-based fallback
- ColorCorrectionMatrix: Computes and applies 3x3 color correction matrices
- ColorCalibrationCache: Manages cached correction matrices for efficiency
"""

import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import warnings
from typing import Optional, Tuple, Dict, Any

# Standard X-Rite ColorChecker Classic reference values (sRGB, 4x6=24 patches)
# Values are in sRGB color space, range 0-255
COLORCHECKER_24_SRGB = np.array([
    # Row 1: Brown tones
    [115, 82, 68],    # Dark Skin
    [194, 150, 130],  # Light Skin
    [98, 122, 157],   # Blue Sky
    [87, 108, 67],    # Foliage
    [133, 128, 177],  # Blue Flower
    [103, 189, 170],  # Bluish Green
    # Row 2: Primary colors
    [214, 126, 44],   # Orange
    [80, 91, 166],    # Purplish Blue
    [193, 90, 99],    # Moderate Red
    [94, 60, 108],    # Purple
    [157, 188, 64],   # Yellow Green
    [224, 163, 46],   # Orange Yellow
    # Row 3: Secondary colors
    [56, 61, 150],    # Blue
    [70, 148, 73],    # Green
    [175, 54, 60],    # Red
    [231, 199, 31],   # Yellow
    [187, 86, 149],   # Magenta
    [8, 133, 161],    # Cyan
    # Row 4: Grayscale
    [243, 243, 242],  # White
    [200, 200, 200],  # Neutral 8
    [160, 160, 160],  # Neutral 6.5
    [122, 122, 121],  # Neutral 5
    [85, 85, 85],     # Neutral 3.5
    [52, 52, 52],     # Black
], dtype=np.float32)

# 5x6 ColorChecker variant (30 patches) - extended version
# First 24 patches match Classic, then 6 additional patches
COLORCHECKER_30_SRGB = np.concatenate([
    COLORCHECKER_24_SRGB,
    np.array([
        # Row 5: Additional reference patches
        [170, 85, 97],    # Additional red
        [40, 70, 130],    # Additional blue
        [190, 165, 75],   # Additional yellow
        [110, 130, 85],   # Additional green
        [140, 95, 145],   # Additional purple
        [185, 145, 125],  # Additional skin tone
    ], dtype=np.float32)
], axis=0)


class ColorCheckerDetector:
    """
    Detect color patches in ColorChecker reference images.

    Uses OpenCV's MCC (Macbeth ColorChecker) module when available,
    with a grid-based sampling fallback for robustness.
    """

    def __init__(self, use_opencv_mcc: bool = True):
        """
        Initialize the detector.

        Args:
            use_opencv_mcc: Whether to try OpenCV MCC detection first
        """
        self.use_opencv_mcc = use_opencv_mcc
        self._has_opencv_mcc = None

    def _check_opencv_mcc(self) -> bool:
        """Check if OpenCV MCC module is available."""
        if self._has_opencv_mcc is None:
            try:
                import cv2
                # Check for mcc module (requires opencv-contrib)
                self._has_opencv_mcc = hasattr(cv2, 'mcc') and hasattr(cv2.mcc, 'CCheckerDetector_create')
            except ImportError:
                self._has_opencv_mcc = False
        return self._has_opencv_mcc

    def detect_patches(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect color patches in a ColorChecker image.

        Args:
            image: RGB image as numpy array (H, W, 3), values 0-255

        Returns:
            Array of detected patch colors, shape (num_patches, 3), or None if detection fails
        """
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Try OpenCV MCC first if available and enabled
        if self.use_opencv_mcc and self._check_opencv_mcc():
            patches = self._detect_with_opencv_mcc(image)
            if patches is not None:
                return patches

        # Fall back to grid-based detection
        return self._detect_with_grid_sampling(image)

    def _detect_with_opencv_mcc(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect patches using OpenCV's MCC module.

        Args:
            image: RGB image as numpy array

        Returns:
            Array of patch colors or None
        """
        try:
            import cv2

            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Create detector
            detector = cv2.mcc.CCheckerDetector_create()

            # Try to detect ColorChecker
            if not detector.process(bgr_image, cv2.mcc.MCC24):
                return None

            # Get detected checkers
            checkers = detector.getListColorChecker()
            if not checkers:
                return None

            # Get the best detected checker
            checker = checkers[0]

            # Get charts corners and extract patch colors
            charts_rgb = checker.getChartsRGB()

            # charts_rgb contains average colors for each patch
            # Shape is (num_patches, 4) where 4th column is a count
            patches = charts_rgb[:, :3].copy()

            # Convert BGR to RGB
            patches = patches[:, ::-1]

            return patches.astype(np.float32)

        except Exception as e:
            warnings.warn(f"OpenCV MCC detection failed: {e}")
            return None

    def _detect_with_grid_sampling(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect patches using grid-based sampling.

        This fallback method assumes the image is primarily a ColorChecker,
        divides it into a grid, and samples the center of each cell.

        Args:
            image: RGB image as numpy array

        Returns:
            Array of patch colors
        """
        try:
            h, w = image.shape[:2]

            # Infer grid layout from aspect ratio
            rows, cols = self._infer_grid_layout(image)

            # Calculate cell dimensions
            cell_h = h // rows
            cell_w = w // cols

            # Sample center 40% of each cell to avoid edges
            margin_h = int(cell_h * 0.3)
            margin_w = int(cell_w * 0.3)

            patches = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate cell bounds
                    y1 = row * cell_h + margin_h
                    y2 = (row + 1) * cell_h - margin_h
                    x1 = col * cell_w + margin_w
                    x2 = (col + 1) * cell_w - margin_w

                    # Ensure valid bounds
                    y1, y2 = max(0, y1), min(h, y2)
                    x1, x2 = max(0, x1), min(w, x2)

                    if y2 <= y1 or x2 <= x1:
                        continue

                    # Sample cell and compute mean color
                    cell = image[y1:y2, x1:x2]
                    mean_color = cell.mean(axis=(0, 1))
                    patches.append(mean_color)

            if len(patches) == 0:
                return None

            return np.array(patches, dtype=np.float32)

        except Exception as e:
            warnings.warn(f"Grid sampling failed: {e}")
            return None

    def _infer_grid_layout(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Infer ColorChecker grid layout from image aspect ratio.

        Args:
            image: RGB image as numpy array

        Returns:
            Tuple of (rows, cols)
        """
        h, w = image.shape[:2]
        aspect_ratio = w / h

        # X-Rite Classic (4x6): aspect ratio ~1.5
        # 5x6 variant: aspect ratio ~1.2
        if aspect_ratio > 1.35:
            return (4, 6)  # 24 patches
        else:
            return (5, 6)  # 30 patches


class ColorCorrectionMatrix:
    """
    Compute and apply 3x3 color correction matrices.

    Uses least squares to find the optimal linear transform that maps
    observed patch colors to reference values.
    """

    def __init__(self, matrix: np.ndarray, offset: Optional[np.ndarray] = None,
                 rmse: float = 0.0, num_patches: int = 0):
        """
        Initialize with a pre-computed matrix.

        Args:
            matrix: 3x3 color correction matrix
            offset: Optional 3-element offset vector (for affine transform)
            rmse: Root mean squared error from fitting
            num_patches: Number of patches used for fitting
        """
        self.matrix = matrix.astype(np.float32)
        self.offset = offset.astype(np.float32) if offset is not None else np.zeros(3, dtype=np.float32)
        self.rmse = rmse
        self.num_patches = num_patches

    @classmethod
    def from_patches(cls, observed: np.ndarray, reference: np.ndarray) -> 'ColorCorrectionMatrix':
        """
        Compute CCM from observed and reference patch colors.

        Uses least squares: reference = observed @ matrix.T + offset

        Args:
            observed: Observed patch colors, shape (N, 3)
            reference: Reference patch colors, shape (N, 3)

        Returns:
            ColorCorrectionMatrix instance
        """
        assert observed.shape == reference.shape
        assert observed.shape[1] == 3

        n = observed.shape[0]

        # Normalize to 0-1 range for better numerical stability
        obs = observed / 255.0
        ref = reference / 255.0

        # Add column of ones for affine transform (offset)
        obs_aug = np.column_stack([obs, np.ones(n)])

        # Solve least squares: ref = obs_aug @ coeffs.T
        # coeffs shape: (3, 4) where last column is offset
        coeffs, residuals, rank, s = np.linalg.lstsq(obs_aug, ref, rcond=None)

        # Extract matrix and offset
        matrix = coeffs[:3, :].T  # (3, 3)
        offset = coeffs[3, :]     # (3,)

        # Compute RMSE
        pred = obs @ matrix.T + offset
        rmse = np.sqrt(np.mean((pred - ref) ** 2)) * 255.0

        # Store matrix scaled for 0-255 input, offset scaled
        return cls(matrix, offset * 255.0, rmse, n)

    @classmethod
    def identity(cls) -> 'ColorCorrectionMatrix':
        """Create an identity (no-op) correction matrix."""
        return cls(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0.0, 0)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color correction to an image.

        Args:
            image: RGB image as numpy array, shape (H, W, 3), values 0-255

        Returns:
            Corrected image with same shape and dtype
        """
        original_dtype = image.dtype

        # Convert to float for computation
        img_float = image.astype(np.float32) / 255.0

        # Apply affine transform: corrected = img @ matrix.T + offset/255
        h, w = img_float.shape[:2]
        flat = img_float.reshape(-1, 3)
        corrected_flat = flat @ self.matrix.T + self.offset / 255.0
        corrected = corrected_flat.reshape(h, w, 3)

        # Clip to valid range and convert back
        corrected = np.clip(corrected * 255.0, 0, 255)

        if original_dtype == np.uint8:
            return corrected.astype(np.uint8)
        return corrected.astype(original_dtype)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'matrix': self.matrix.tolist(),
            'offset': self.offset.tolist(),
            'rmse': float(self.rmse),
            'num_patches': int(self.num_patches)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorCorrectionMatrix':
        """Deserialize from dictionary."""
        return cls(
            matrix=np.array(data['matrix'], dtype=np.float32),
            offset=np.array(data['offset'], dtype=np.float32),
            rmse=data.get('rmse', 0.0),
            num_patches=data.get('num_patches', 0)
        )

    def is_identity(self) -> bool:
        """Check if this is an identity matrix (no correction)."""
        return np.allclose(self.matrix, np.eye(3)) and np.allclose(self.offset, 0)


class ColorCalibrationCache:
    """
    Manage cached color correction matrices.

    Stores CCMs keyed by colorpicker image path for fast lookup.
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[str, ColorCorrectionMatrix] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'failures': 0
        }

    def load(self, path: Path) -> bool:
        """
        Load cache from disk.

        Args:
            path: Path to pickle file

        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Reconstruct CCM objects from dicts
            self._cache = {
                k: ColorCorrectionMatrix.from_dict(v)
                for k, v in data.get('matrices', {}).items()
            }
            self._stats = data.get('stats', self._stats)

            print(f"Loaded {len(self._cache)} color correction matrices from {path}")
            return True

        except Exception as e:
            warnings.warn(f"Failed to load color calibration cache: {e}")
            return False

    def save(self, path: Path) -> None:
        """
        Save cache to disk.

        Args:
            path: Path to pickle file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'matrices': {k: v.to_dict() for k, v in self._cache.items()},
            'stats': self._stats
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved {len(self._cache)} color correction matrices to {path}")

    def get(self, colorpicker_path: str) -> Optional[ColorCorrectionMatrix]:
        """
        Get CCM for a colorpicker image.

        Args:
            colorpicker_path: Path/identifier for the colorpicker image

        Returns:
            ColorCorrectionMatrix if found, None otherwise
        """
        if colorpicker_path in self._cache:
            self._stats['hits'] += 1
            return self._cache[colorpicker_path]
        else:
            self._stats['misses'] += 1
            return None

    def add(self, colorpicker_path: str, ccm: ColorCorrectionMatrix) -> None:
        """
        Add a CCM to the cache.

        Args:
            colorpicker_path: Path/identifier for the colorpicker image
            ccm: ColorCorrectionMatrix to cache
        """
        self._cache[colorpicker_path] = ccm

    def mark_failure(self, colorpicker_path: str) -> None:
        """Mark a colorpicker as having failed detection."""
        # Store identity matrix for failed detections
        self._cache[colorpicker_path] = ColorCorrectionMatrix.identity()
        self._stats['failures'] += 1

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            **self._stats,
            'total_cached': len(self._cache),
            'identity_count': sum(1 for ccm in self._cache.values() if ccm.is_identity())
        }


def apply_color_correction(image: Image.Image, ccm: ColorCorrectionMatrix) -> Image.Image:
    """
    Apply color correction to a PIL Image.

    This is the main entry point for applying color normalization to beetle images
    before feature extraction.

    Args:
        image: PIL Image object
        ccm: ColorCorrectionMatrix to apply

    Returns:
        Color-corrected PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        alpha = img_array[:, :, 3:4]
        img_array = img_array[:, :, :3]
        has_alpha = True
    else:
        has_alpha = False

    # Apply correction
    corrected = ccm.apply(img_array)

    # Handle alpha channel
    if has_alpha:
        corrected = np.concatenate([corrected, alpha], axis=-1)
        return Image.fromarray(corrected.astype(np.uint8), mode='RGBA')

    return Image.fromarray(corrected.astype(np.uint8), mode='RGB')


def get_reference_colors(num_patches: int) -> np.ndarray:
    """
    Get reference ColorChecker colors for a given number of patches.

    Args:
        num_patches: Number of patches (24 or 30)

    Returns:
        Reference colors array, shape (num_patches, 3)
    """
    if num_patches <= 24:
        return COLORCHECKER_24_SRGB[:num_patches].copy()
    else:
        return COLORCHECKER_30_SRGB[:num_patches].copy()
