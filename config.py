import os
import torch

# Dataset directory paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Model checkpoints and outputs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_ultrasound_model.pth")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image hyperparameters
IMG_SIZE = 224  # Standard input resolution for ResNet / EfficientNet
CHANNELS = 3    # RGB (Grayscale ultrasound scans are replicated to 3 channels for transfer learning)

# Classification configuration
# 0: Benign / Normal, 1: Malignant
NUM_CLASSES = 2
CLASS_NAMES = ["benign", "malignant"]

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5

# Preprocessing flag: Enable CLAHE (Contrast Limited Adaptive Histogram Equalization)
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Device configuration (auto-detect Apple Silicon MPS, NVIDIA CUDA, or CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
