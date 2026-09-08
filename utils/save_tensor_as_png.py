# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import torch
from PIL import Image
import numpy as np


def save_tensor_as_png(tensor: torch.Tensor, save_path: str, normalize: bool = True):
    """
    Save a PyTorch tensor as a PNG image.

    Args:
        tensor (torch.Tensor): Input tensor. Can be:
            - Grayscale: shape (H, W) or (1, H, W)
            - RGB: shape (3, H, W) or (H, W, 3)
            - Batch: shape (1, H, W) or (1, 3, H, W)
        save_path (str): Path to save the PNG file
        normalize (bool): Whether to normalize tensor values to [0, 255] range
    """
    # Convert to CPU and detach from computation graph
    tensor = tensor.detach().cpu()

    # Handle different tensor shapes
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor.squeeze(0)  # Remove batch dimension

    if tensor.dim() == 3:
        if tensor.shape[0] == 1:  # Grayscale with channel dimension
            tensor = tensor.squeeze(0)
        elif tensor.shape[0] == 3:  # RGB format (C, H, W)
            tensor = tensor.permute(1, 2, 0)  # Convert to (H, W, C)

    # Convert to numpy
    numpy_array = tensor.numpy()

    # Normalize if needed
    if normalize:
        if numpy_array.dtype != np.uint8:
            # Normalize to [0, 1] then scale to [0, 255]
            numpy_array = (numpy_array - numpy_array.min()) / (
                numpy_array.max() - numpy_array.min()
            )
            numpy_array = (numpy_array * 255).astype(np.uint8)
    else:
        # Ensure values are in valid range [0, 255]
        numpy_array = np.clip(numpy_array, 0, 255).astype(np.uint8)

    # Create PIL Image and save
    if numpy_array.ndim == 2:  # Grayscale
        image = Image.fromarray(numpy_array, mode="L")
    else:  # RGB
        image = Image.fromarray(numpy_array, mode="RGB")

    image.save(save_path)
    print(f"Tensor saved as PNG to: {save_path}")


# Example usage
if __name__ == "__main__":
    # Example 1: Random grayscale tensor
    grayscale_tensor = torch.rand(100, 100)
    save_tensor_as_png(grayscale_tensor, "grayscale_example.png")

    # Example 2: Random RGB tensor
    rgb_tensor = torch.rand(3, 100, 100)
    save_tensor_as_png(rgb_tensor, "rgb_example.png")

    # Example 3: Binary mask (values 0 and 1)
    binary_mask = torch.randint(0, 2, (100, 100)).float()
    save_tensor_as_png(binary_mask, "binary_mask.png", normalize=False)
