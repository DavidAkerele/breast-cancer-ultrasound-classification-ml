import os
import cv2
import numpy as np
import config

def generate_synthetic_mammogram(is_malignant=False, size=224):
    """
    Generates a synthetic grayscale mammogram image for pipeline testing.
    - Simulated dense breast tissue gradient background.
    - If malignant: inserts spiculated/irregular high-contrast masses and cluster microcalcifications.
    - If benign: inserts smooth oval lesions or normal background tissue.
    """
    # Base background: simulated breast profile curve with perlin/gaussian noise
    img = np.zeros((size, size), dtype=np.float32)

    # Create a smooth radial gradient simulating breast tissue from left to right
    for x in range(size):
        for y in range(size):
            dist = np.sqrt((x - size * 0.1)**2 + (y - size * 0.5)**2)
            intensity = max(0, 180 - dist * 0.8)
            img[y, x] = intensity

    # Add realistic structured noise (simulating fibroglandular tissue)
    noise = np.random.normal(loc=0.0, scale=12.0, size=(size, size))
    img = np.clip(img + noise, 0, 255)

    if is_malignant:
        # Simulate irregular mass (high density center with spiculated edges)
        center_x = np.random.randint(int(size * 0.3), int(size * 0.7))
        center_y = np.random.randint(int(size * 0.3), int(size * 0.7))
        radius = np.random.randint(15, 30)

        for x in range(max(0, center_x - radius*2), min(size, center_x + radius*2)):
            for y in range(max(0, center_y - radius*2), min(size, center_y + radius*2)):
                d = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if d < radius:
                    # High intensity core
                    img[y, x] = min(255, img[y, x] + (radius - d) * 3.5)
                elif d < radius * 1.5:
                    # Spiculated/irregular extensions
                    angle = np.arctan2(y - center_y, x - center_x)
                    spicule = np.sin(angle * 8) * 10
                    if d < radius + spicule:
                        img[y, x] = min(255, img[y, x] + 40)

        # Simulate microcalcification clusters (tiny bright dots)
        num_calcifications = np.random.randint(5, 15)
        for _ in range(num_calcifications):
            cx = np.clip(center_x + np.random.randint(-20, 20), 0, size - 1)
            cy = np.clip(center_y + np.random.randint(-20, 20), 0, size - 1)
            img[cy, cx] = 255
            if cy + 1 < size:
                img[cy + 1, cx] = 240
    else:
        # Optional benign finding: smooth circumscribed oval mass or cyst
        if np.random.rand() > 0.5:
            center_x = np.random.randint(int(size * 0.3), int(size * 0.7))
            center_y = np.random.randint(int(size * 0.3), int(size * 0.7))
            radius = np.random.randint(10, 22)
            cv2.circle(img, (center_x, center_y), radius, (160,), -1)
            # Smooth blur around benign lesion
            img = cv2.GaussianBlur(img, (5, 5), 1.5)

    img = np.clip(img, 0, 255).astype(np.uint8)
    # Convert grayscale to RGB 3-channel
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb

def create_dummy_dataset(num_train=30, num_val=10):
    print("Generating synthetic dummy dataset for immediate pipeline testing...")
    for split, count in [("train", num_train), ("val", num_val), ("test", num_val)]:
        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            class_dir = os.path.join(config.DATA_DIR, split, class_name)
            os.makedirs(class_dir, exist_ok=True)

            is_malignant = (class_name == "malignant")
            for i in range(count):
                img = generate_synthetic_mammogram(is_malignant=is_malignant, size=config.IMG_SIZE)
                filepath = os.path.join(class_dir, f"sample_{i}.png")
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Dummy dataset generated successfully at: {config.DATA_DIR}")

if __name__ == "__main__":
    create_dummy_dataset()
