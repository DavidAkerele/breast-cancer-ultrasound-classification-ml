import os
import torch

# Base project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset directory paths
DATA_DIR = os.path.join(BASE_DIR, "data")
ACTIVE_DATASET = os.getenv("ACTIVE_DATASET", "busi")
TRAIN_DIR = os.path.join(DATA_DIR, ACTIVE_DATASET, "train")
VAL_DIR = os.path.join(DATA_DIR, ACTIVE_DATASET, "val")
TEST_DIR = os.path.join(DATA_DIR, ACTIVE_DATASET, "test")

# Web static frontend directory
WEB_DIR = os.path.join(BASE_DIR, "web")

# Model checkpoints and outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_ultrasound_model.pth")
MODEL_CHECKPOINT_PATHS = {
    "resnet50": os.path.join(OUTPUT_DIR, "resnet50_model.pth"),
    "efficientnet_b0": os.path.join(OUTPUT_DIR, "efficientnet_b0_model.pth"),
    "custom_cnn": os.path.join(OUTPUT_DIR, "custom_cnn_model.pth"),
}

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
CLAHE_TILE_GRID_SIZE = (16, 16)

# Cropping & Resizing Strategy: "roi_crop" (Padded ROI), "center_crop" (Default PyTorch), or "direct_resize" (Anamorphic)
CROP_STRATEGY = os.getenv("CROP_STRATEGY", "roi_crop")
ROI_MARGIN_RATIO = 0.20  # 20% margin around localized lesion to preserve acoustic shadowing

# Device configuration (auto-detect Apple Silicon MPS, NVIDIA CUDA, or CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
