import torch
import cv2
import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def histogram_equalization(img_tensor):
    # Convert to numpy for histogram equalization
    img_np = img_tensor.squeeze().cpu().numpy()
    if img_np.ndim == 3:
        # For RGB images, apply HE to each channel
        img_eq = np.zeros_like(img_np)
        for c in range(3):
            img_eq[c] = cv2.equalizeHist((img_np[c] * 255).astype(np.uint8)) / 255.0
    else:
        img_eq = cv2.equalizeHist((img_np * 255).astype(np.uint8)) / 255.0
    return torch.from_numpy(img_eq).float().unsqueeze(0)

def morphological_operation(img_tensor, operation='dilate', kernel_size=3):
    # Convert to numpy for morphological operations
    img_np = img_tensor.squeeze().cpu().numpy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'dilate':
        img_morph = cv2.dilate((img_np * 255).astype(np.uint8), kernel, iterations=1)
    elif operation == 'erode':
        img_morph = cv2.erode((img_np * 255).astype(np.uint8), kernel, iterations=1)
    else:
        raise ValueError("Operation must be 'dilate' or 'erode'")
    img_morph = img_morph / 255.0
    return torch.from_numpy(img_morph).float().unsqueeze(0)

def load_image(path, grayscale=False):
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img_tensor = TF.to_tensor(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension

def show_image(img_tensor, title='Image'):
    img = img_tensor.squeeze().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Example usage
    img_path = 'DeepLearning-S7-AI-ML-KTU-Lab\For_28Jul\input.jpg'
    img_tensor = load_image(img_path, grayscale=True)

    # Histogram Equalization
    img_he = histogram_equalization(img_tensor)
    show_image(img_tensor, 'Original')
    show_image(img_he, 'Histogram Equalized')

    # Morphological Operations
    img_dilated = morphological_operation(img_tensor, operation='dilate', kernel_size=5)
    img_eroded = morphological_operation(img_tensor, operation='erode', kernel_size=5)
    show_image(img_dilated, 'Dilated')
    show_image(img_eroded, 'Eroded')