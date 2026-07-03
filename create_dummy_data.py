import os
import cv2
import numpy as np
import config

def generate_synthetic_ultrasound(is_malignant=False, size=224):
    """
    Generates a synthetic grayscale breast ultrasound image for pipeline testing.
    - Speclked background simulating tissue echogenicity.
    - Horizontal bands representing skin (bright top), fat (darker), and glandular tissue.
    - Benign: circumscribed, oval hypoechoic (darker) mass, clean boundaries, posterior enhancement.
    - Malignant: irregular, microlobulated markedly hypoechoic mass with posterior acoustic shadowing (dark band below).
    """
    # Base background: dark speckle noise representing background echogenicity
    img = np.random.normal(loc=50.0, scale=8.0, size=(size, size)).astype(np.float32)
    img = np.clip(img, 0, 255)

    # 1. Simulate skin line (bright thin layer at the top)
    img[0:6, :] = np.clip(img[0:6, :] + np.random.normal(loc=140.0, scale=10.0, size=(6, size)), 0, 255)
    img[6:15, :] = np.clip(img[6:15, :] - 15, 0, 255) # Subcutaneous fat layer (hypoechoic)

    # 2. Add some horizontal tissue texture (wavy structures)
    for y in range(20, size, 12):
        wave = np.sin(np.linspace(0, 4 * np.pi, size)) * 6.0
        for x in range(size):
            target_y = int(y + wave[x] + np.random.randint(-1, 2))
            if 0 <= target_y < size:
                img[target_y : min(size, target_y + 3), x] = np.clip(
                    img[target_y : min(size, target_y + 3), x] + 25.0, 0, 255
                )

    # 3. Insert Lesion (mass)
    center_x = np.random.randint(int(size * 0.35), int(size * 0.65))
    center_y = np.random.randint(int(size * 0.4), int(size * 0.65))

    if is_malignant:
        # Malignant: irregular shape
        radius_x = np.random.randint(18, 28)
        radius_y = np.random.randint(14, 24)
        
        # Markedly hypoechoic lesion (very dark center)
        for y in range(max(0, center_y - radius_y * 2), min(size, center_y + radius_y * 2)):
            for x in range(max(0, center_x - radius_x * 2), min(size, center_x + radius_x * 2)):
                # Calculate distance normalized by radius (ellipsoid) with angular perturbations
                dx = x - center_x
                dy = y - center_y
                angle = np.arctan2(dy, dx)
                perturbation = np.sin(angle * 7) * 4.0 + np.cos(angle * 3) * 2.0
                dist = np.sqrt(dx**2 + (dy * (radius_x / radius_y))**2)
                
                limit = radius_x + perturbation
                if dist < limit:
                    # MARKEDLY hypoechoic (value goes down to ~15-25)
                    factor = (limit - dist) / limit
                    img[y, x] = max(10, img[y, x] - factor * 45.0)

        # 4. Posterior Acoustic Shadowing (dark vertical band directly under the malignant lesion)
        shadow_width = int(radius_x * 1.5)
        for y in range(center_y + 5, size):
            for x in range(max(0, center_x - shadow_width // 2), min(size, center_x + shadow_width // 2)):
                # Shadow gets diffuse/fades towards the bottom
                dist_from_lesion = y - center_y
                shadow_factor = max(0.2, 1.0 - (dist_from_lesion / (size - center_y)) * 0.5)
                img[y, x] = max(8, img[y, x] - 32.0 * shadow_factor)
    else:
        # Benign: smooth circumscribed oval mass
        radius_x = np.random.randint(16, 26)
        radius_y = np.random.randint(10, 16)
        
        # Moderately hypoechoic (value goes down to ~35-45)
        for y in range(max(0, center_y - radius_y * 2), min(size, center_y + radius_y * 2)):
            for x in range(max(0, center_x - radius_x * 2), min(size, center_x + radius_x * 2)):
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx**2 + (dy * (radius_x / radius_y))**2)
                
                if dist < radius_x:
                    factor = (radius_x - dist) / radius_x
                    img[y, x] = max(20, img[y, x] - factor * 30.0)
                    
        # 5. Posterior Acoustic Enhancement (light vertical band directly under benign lesion)
        enhance_width = int(radius_x * 1.1)
        for y in range(center_y + radius_y, size):
            for x in range(max(0, center_x - enhance_width // 2), min(size, center_x + enhance_width // 2)):
                dist_from_lesion = y - center_y
                enhance_factor = max(0.1, 1.0 - (dist_from_lesion / (size - center_y)) * 0.7)
                img[y, x] = min(255, img[y, x] + 20.0 * enhance_factor)

    # Apply Gaussian blur to blend features and speckle nicely
    img = cv2.GaussianBlur(img, (3, 3), 1.0)
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
                img = generate_synthetic_ultrasound(is_malignant=is_malignant, size=config.IMG_SIZE)
                filepath = os.path.join(class_dir, f"sample_{i}.png")
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Dummy dataset generated successfully at: {config.DATA_DIR}")

if __name__ == "__main__":
    create_dummy_dataset(num_train=50, num_val=25)
