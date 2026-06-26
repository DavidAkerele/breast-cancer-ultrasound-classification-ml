import os
import argparse
import cv2
from PIL import Image
import torch
from torchvision import transforms

import config
from dataset import apply_clahe
from models import get_model

@torch.no_grad()
def predict_image(image_path, model, transform, device, use_clahe=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Read image
    img_np = cv2.imread(image_path)
    if img_np is None:
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    if use_clahe and config.USE_CLAHE:
        img_np = apply_clahe(img_np)

    img_pil = Image.fromarray(img_np)
    tensor = transform(img_pil).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).squeeze(0)
    pred_idx = torch.argmax(probs).item()
    pred_class = config.CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    return pred_class, confidence, probs.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Predict Breast Cancer Mammogram Class")
    parser.add_argument("--image", type=str, required=True, help="Path to mammogram image file or folder")
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH, help="Path to model checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Please train the model first.")

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model_name = checkpoint.get("model_name", "resnet50")

    model = get_model(model_name=model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if os.path.isdir(args.image):
        print(f"\nRunning predictions on directory: {args.image}")
        for fname in sorted(os.listdir(args.image)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.tif')):
                fpath = os.path.join(args.image, fname)
                pred_class, conf, _ = predict_image(fpath, model, val_transform, config.DEVICE)
                print(f"File: {fname:<25} -> Prediction: {pred_class.upper():<10} (Confidence: {conf*100:.2f}%)")
    else:
        pred_class, conf, probs = predict_image(args.image, model, val_transform, config.DEVICE)
        print("\n--- Prediction Result ---")
        print(f"Image Path:  {args.image}")
        print(f"Prediction:  {pred_class.upper()}")
        print(f"Confidence:  {conf*100:.2f}%")
        for idx, class_name in enumerate(config.CLASS_NAMES):
            print(f"  {class_name.capitalize():<10} probability: {probs[idx]*100:.2f}%")

if __name__ == "__main__":
    main()
