import os
import torch

# Base project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset directory paths
DATA_DIR = os.path.join(BASE_DIR, "data")
ACTIVE_DATASET = os.getenv("ACTIVE_DATASET", "busi")
# Validated experiments default to datasets whose local subject identifiers are
# recoverable. BUSI remains available for exploratory work until its source
# identifier manifest is restored.
TRAIN_DATASETS = tuple(
    name.strip().lower()
    for name in os.getenv("TRAIN_DATASETS", "breast,oasbud").split(",")
    if name.strip()
)
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
# The dissertation's audited evaluation record uses EfficientNet-B0. Keep the
# generic training checkpoint above for training compatibility, but make the
# audited checkpoint the default for evaluation, prediction, and the dashboard.
EVALUATION_CHECKPOINT_PATH = os.getenv(
    "EVALUATION_CHECKPOINT_PATH", MODEL_CHECKPOINT_PATHS["efficientnet_b0"]
)

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
# Low-confidence predictions are surfaced as UNCERTAIN by the API instead of
# being presented as a forced benign/malignant decision.
ABSTAIN_CONFIDENCE = float(os.getenv("ABSTAIN_CONFIDENCE", "0.60"))

# Device configuration (auto-detect Apple Silicon MPS, NVIDIA CUDA, or CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
