import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def histogram_equalization(img_tensor):
    # Convert to numpy
    img_np = img_tensor.squeeze().cpu().numpy()
    if img_np.ndim == 3:
        # For RGB images, apply HE to each channel
        img_eq = np.zeros_like(img_np)
        for c in range(3):
            img_eq[c] = _equalize_channel(img_np[c])
    else:
        img_eq = _equalize_channel(img_np)
    return torch.from_numpy(img_eq).float().unsqueeze(0)

def _equalize_channel(channel):
    # Compute histogram
    hist, bins = np.histogram(channel.flatten(), bins=256, range=[0, 1])
    cdf = hist.cumsum()  # Cumulative distribution function
    cdf_normalized = cdf / cdf[-1]  # Normalize to [0, 1]
    equalized = np.interp(channel.flatten(), bins[:-1], cdf_normalized)
    return equalized.reshape(channel.shape)

def morphological_operation(img_tensor, operation='dilate', kernel_size=3):
    # Convert to numpy
    img_np = img_tensor.squeeze().cpu().numpy()
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    if operation == 'open':
        img_morph = _dilate(_erode(img_np, kernel), kernel)
    elif operation == 'close':
        img_morph = _erode(_dilate(img_np, kernel), kernel)
    else:
        raise ValueError("Operation must be 'open' or 'close'")
    return torch.from_numpy(img_morph).float().unsqueeze(0)

def _dilate(image, kernel):
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    dilated = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dilated[i, j] = np.max(padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return dilated

def _erode(image, kernel):
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=1)
    eroded = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            eroded[i, j] = np.min(padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return eroded

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
    img_path = '/home/cs-ai-21/Prince/DeepLearning-S7-AI-ML-KTU-Lab/code/exp3/input.jpg'
    img_tensor = load_image(img_path, grayscale=True)

    # Histogram Equalization
    img_he = histogram_equalization(img_tensor)
    img_open = morphological_operation(img_tensor, operation='open', kernel_size=5)
    img_close = morphological_operation(img_tensor, operation='close', kernel_size=5)

    # Display all four images together
    images = [img_tensor, img_he, img_open, img_close]
    titles = ['Original', 'Histogram Equalized', 'Opened', 'Closed']
    plt.figure(figsize=(12, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i+1)
        arr = img.squeeze().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        plt.imshow(arr, cmap='gray' if arr.ndim == 2 else None)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
