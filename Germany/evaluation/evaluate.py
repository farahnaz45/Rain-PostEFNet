"""
Evaluation module. Refer to `notebooks/evaluation_example.ipynb` on examples.
"""

from collections import defaultdict
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm

from data.dataset import StandardDataset
from evaluation.metrics import compile_metrics,compile_metrics_Germany
from einops import rearrange


def evaluate_model(model: nn.Module, data_loader, thresholds, criterion, device,
                   normalization=None) -> Tuple[float, float, np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param model:
    :param data_loader:
    :param thresholds:
    :param criterion:
    :param device:
    :param normalization:
    :return: confusion, binary_metrics_by_threshold
    """
    dataset = data_loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, StandardDataset):
        raise ValueError('`data_loader` must contain a (subset of) StandardDataset')

    n_thresholds = len(thresholds)
    n_classes = n_thresholds + 1
    total_loss = 0
    total_samples = 0

    model.eval()

    # Run inference on single epoch
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    metrics_by_threshold = defaultdict(list)
    for i, (images, target, t) in enumerate(tqdm(data_loader)):
        # Note, StandardDataset retrieves timestamps in Tensor format due to collation issue, for now
        timestamps = []
        for e in t:
            origin = datetime(year=e[0], month=e[1], day=e[2], hour=e[3])
            lead_time = e[4].item()
            timestamps.append((origin, lead_time))

        if normalization:
            with torch.no_grad():
                for i, (max_val, min_val) in enumerate(zip(normalization['max_values'], normalization['min_values'])):
                    if min_val < 0:
                        images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                    else:
                        images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

        images = images.float().to(device)
        target = target.long().to(device)
        output = model(images, t)
        loss, _, _ = criterion(output, target, timestamps, mode="train")
        if loss is None:  # hotfix for None return values from losses.CrossEntropyLoss
            continue

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]

        predictions = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)[1]  # (batch_size, height, width)
        predictions = predictions.numpy()
        step_confusion, step_metrics_by_threshold = compile_metrics(data_loader.dataset, predictions, timestamps,
                                                                    thresholds)
        confusion += step_confusion
        for threshold, metrics in step_metrics_by_threshold.items():
            metrics_by_threshold[threshold].append(metrics)

    metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
    correct = (confusion[np.diag_indices_from(confusion)]).sum()
    accuracy = correct / confusion.sum()
    loss = total_loss / total_samples

    return accuracy, loss, confusion, metrics_by_threshold

def evaluate_model_Germany(model: nn.Module, data_loader, thresholds, criterion, device,
                   normalization=None) -> Tuple[float, float, np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param model:
    :param data_loader:
    :param thresholds:
    :param criterion:
    :param device:
    :param normalization:
    :return: confusion, binary_metrics_by_threshold
    """
    # dataset = data_loader.dataset
    # if isinstance(dataset, Subset):
    #     dataset = dataset.dataset
    # if not isinstance(dataset, StandardDataset):
    #     raise ValueError('`data_loader` must contain a (subset of) StandardDataset')

    n_thresholds = len(thresholds)
    n_classes = n_thresholds + 1
    total_loss = 0
    total_samples = 0

    model.eval()

    # Run inference on single epoch
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    metrics_by_threshold = defaultdict(list)
    for i, (images, target, t) in enumerate(tqdm(data_loader)):
        # Note, StandardDataset retrieves timestamps in Tensor format due to collation issue, for now
        # timestamps = []
        # for e in t:
        #     origin = datetime(year=e[0], month=e[1], day=e[2], hour=e[3])
        #     lead_time = e[4].item()
        #     timestamps.append((origin, lead_time))

        if normalization:
            with torch.no_grad():
                for i, (max_val, min_val) in enumerate(zip(normalization['max_values'], normalization['min_values'])):
                    if min_val < 0:
                        images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                    else:
                        images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

        images = images.float().to(device)
        target = target.long().to(device)
        output = model(images, t)
        timestamps = None
        loss, _, _ = criterion(output, target, timestamps, mode="train")
        if loss is None:  # hotfix for None return values from losses.CrossEntropyLoss
            continue

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]

        predictions = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)[1]  # (batch_size, height, width)
        predictions = predictions.numpy()
        step_confusion, step_metrics_by_threshold = compile_metrics_Germany(data_loader.dataset, predictions, target.detach().cpu().numpy(), thresholds)
        confusion += step_confusion
        for threshold, metrics in step_metrics_by_threshold.items():
            metrics_by_threshold[threshold].append(metrics)

    metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
    correct = (confusion[np.diag_indices_from(confusion)]).sum()
    accuracy = correct / confusion.sum()
    loss = total_loss / total_samples

    return accuracy, loss, confusion, metrics_by_threshold

#     return accuracy, loss, confusion, metrics_by_threshold


import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from data.dataset import StandardDataset
from evaluation.metrics import compile_metrics,compile_metrics_Germany
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from data.dataset import StandardDataset
from evaluation.metrics import compile_metrics,compile_metrics_Germany
from einops import rearrange
# from utils import rgb_to_grayscale  # Assuming this utility function exists
import matplotlib.colors as mcolors

# from utils import rgb_to_grayscale # Assuming this utility function exists


def rgb_to_grayscale(image):
    """Convert an RGB image to grayscale using standard luminance method."""
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# def visualize_predictions(predictions, ground_truth, num_samples=5, save_path="visualizations/visualization.png", index=0):
#     """
#     Visualizes the best epoch's predictions and corresponding ground truth for 3-class classification.

#     :param predictions: Torch tensor of model predictions (shape: batch_size, 3, height, width)
#     :param ground_truth: Torch tensor of actual labels (shape: batch_size, 1, height, width)
#     :param num_samples: Number of images to visualize
#     :param save_path: Path to save the visualization
#     """
#     num_samples = min(num_samples, len(predictions))

#     fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples)) # Adjusted figsize

#     # Ensure axes is always iterable correctly
#     if num_samples == 1:
#         axes = [axes]  # Convert single pair of axes into a list

   
#     save_path = f"visualizations/visualization-{index}.png"


#     # Define a colormap for 3 classes (e.g., using ListedColormap)
#     colors = ['green', 'blue', 'red']  # Example: class 0=black, class 1=gray, class 2=white
#     cmap_3classes = mcolors.ListedColormap(colors)
#     class_labels = ["Class 0", "Class 1", "Class 2"] # Optional class labels for legend/colorbar

#     for i in range(num_samples):
#         # --- Ground Truth ---
#         gt = ground_truth[i].cpu().squeeze(0).numpy()  # (height, width) - already correct shape for grayscale
#         ax_gt = axes[i][0] # Correctly index axes for 2 columns
#         im_gt = ax_gt.imshow(gt, cmap=cmap_3classes, interpolation="nearest", vmin=0, vmax=2) # Use 3-class colormap, adjust vmax to max class index
#         ax_gt.set_title(f"Ground Truth") # Removed sample index from title for cleaner look
#         ax_gt.axis("off")
#         fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04, ticks=[0, 1, 2]) # Add ticks for classes in colorbar


#         # --- Prediction ---
#         pred_tensor = predictions[i].detach().cpu() # (3, height, width)
#         predicted_classes = torch.argmax(pred_tensor, dim=0).numpy() # (height, width) - Class index map
#         ax_pred = axes[i][1] # Correctly index axes for 2 columns
#         im_pred = ax_pred.imshow(predicted_classes, cmap=cmap_3classes, interpolation="nearest", vmin=0, vmax=2) # Use 3-class colormap, adjust vmax
#         ax_pred.set_title(f"Prediction") # Removed sample index from title for cleaner look
#         ax_pred.axis("off")
#         # print(im_pred)
#         fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04, ticks=[0, 1, 2]) # Add ticks for classes in colorbar

   

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()



# regression one but blurred

# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
# import torch

# def visualize_predictions(predictions, ground_truth, num_samples=5, save_path="visualizations/visualization.png", index=0):
#     """
#     Visualizes the best epoch's predictions and corresponding ground truth for a regression task.

#     :param predictions: Torch tensor of model predictions (shape: batch_size, 1, height, width)
#     :param ground_truth: Torch tensor of actual values (shape: batch_size, 1, height, width)
#     :param num_samples: Number of images to visualize
#     :param save_path: Path to save the visualization
#     """
#     num_samples = min(num_samples, len(predictions))

#     fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples)) # Adjusted figsize

#     # Ensure axes is always iterable correctly
#     if num_samples == 1:
#         axes = [axes]  # Convert single pair of axes into a list

#     save_path = f"visualizations/visualization-{index}.png"

#     # Define a continuous colormap for regression (e.g., viridis)
#     cmap_continuous = plt.cm.viridis  # You can replace 'viridis' with other continuous colormaps like 'plasma' or 'inferno'

#     for i in range(num_samples):
#         # --- Ground Truth ---
#         gt = ground_truth[i].cpu().squeeze(0).numpy()  # (height, width) - already correct shape for grayscale
#         ax_gt = axes[i][0]  # Correctly index axes for 2 columns
#         im_gt = ax_gt.imshow(gt, cmap=cmap_continuous, interpolation="nearest")  # Use continuous colormap
#         ax_gt.set_title(f"Ground Truth")
#         ax_gt.axis("off")
#         fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)  # Add colorbar without ticks

#         # --- Prediction ---
#         pred_tensor = predictions[i].detach().cpu().squeeze().numpy()[0]
#         print(pred_tensor)
#         # (height, width) for regression values
#         ax_pred = axes[i][1]  # Correctly index axes for 2 columns
#         im_pred = ax_pred.imshow(pred_tensor, cmap=cmap_continuous, interpolation="nearest")  # Use continuous colormap
#         ax_pred.set_title(f"Prediction")
#         ax_pred.axis("off")
#         fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)  # Add colorbar without ticks

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import torch
import random

# def visualize_predictions(predictions, ground_truth, num_samples=5, save_path="visualizations/visualization.png", index=0):
#     """
#     Visualizes the best epoch's predictions and corresponding ground truth for a regression task or classification task
#     with RGB continuous color mapping.

#     :param predictions: Torch tensor of model predictions (shape: batch_size, 3, height, width)
#     :param ground_truth: Torch tensor of actual values (shape: batch_size, 1, height, width)
#     :param num_samples: Number of images to visualize
#     :param save_path: Path to save the visualization
#     """
#     num_samples = min(num_samples, len(predictions))

#     fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))  # Adjusted figsize

#     # Ensure axes is always iterable correctly
#     if num_samples == 1:
#         axes = [axes]  # Convert single pair of axes into a list

#     save_path = f"visualizations/visualization-{index}.png"

#     # Normalize the classification output to the [0, 1] range for RGB
#     cmap_continuous = plt.cm.viridis  # You can replace 'viridis' with other continuous colormaps like 'plasma' or 'inferno'

#     for i in range(num_samples):
#         # --- Ground Truth ---
#         gt = ground_truth[i].cpu().squeeze(0).numpy()  # (height, width) - already correct shape for grayscale
#         ax_gt = axes[i][0]  # Correctly index axes for 2 columns
#         im_gt = ax_gt.imshow(gt, cmap=cmap_continuous, interpolation="nearest")  # Use continuous colormap
#         ax_gt.set_title(f"Ground Truth")
#         ax_gt.axis("off")
#         fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)  # Add colorbar without ticks

#         # --- Prediction ---
#         # Normalize the classification output to create a continuous RGB representation
        
#         pred_tensor = predictions[i].detach().cpu().numpy()  # (3, 64, 64) or (1, 64, 64)

#     # 1. Get Predicted Class Labels (Essential!)
#         pred_labels = np.argmax(pred_tensor, axis=0)  # Shape: (64, 64) for multi-class

#     # 2. Create a ListedColormap (Important!)
#         num_classes = pred_tensor.shape[0] if pred_tensor.ndim == 3 else np.max(pred_labels) + 1 # Dynamic num_classes
#         colors = ["black", "gray", "white", "red", "green", "blue", "yellow", "cyan", "magenta"]  # Add more colors as needed
#         if num_classes > len(colors):
            
#             extra_colors = [tuple(random.random() for i in range(3)) for _ in range(num_classes - len(colors))]
#             colors.extend(extra_colors)
#         cmap = ListedColormap(colors[:num_classes])

#         ax_pred = axes[i][1]
#         im_pred = ax_pred.imshow(pred_labels, cmap=cmap, interpolation="nearest", vmin=0, vmax=num_classes - 1)  # vmin/vmax are CRUCIAL # Normalize to [0, 1] range

#         # Convert to RGB by stacking the 3 channels (this works as RGB, where each channel represents a class prediction)
#         # rgb_image = np.transpose(pred_normalized, (1, 2, 0))
#         num_classes = pred_tensor.shape[0]
#         # cmap = ListedColormap(["black", "gray", "white", "red", "green", "blue", ...])  # Add enough colors!  # Shape: (height, width, 3) for RGB

#         ax_pred = axes[i][1]
#         im_pred = ax_pred.imshow(pred_labels, cmap=cmap, interpolation="nearest", vmin=0, vmax=num_classes - 1)  # Use RGB continuous color
#         ax_pred.set_title(f"Prediction")
#         ax_pred.axis("off")
#         fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)  # Add colorbar without ticks

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()


import matplotlib.pyplot as plt
import torch

def visualize_predictions(predictions, ground_truth, num_samples=5, save_path="visualizations/visualization.png", index=0):
    """
    Visualizes pixel-by-pixel comparisons of predictions and ground truth.

    :param predictions: Torch tensor of model predictions (shape: batch_size, 3, height, width) or (batch_size, 1, height, width)
    :param ground_truth: Torch tensor of actual values (shape: batch_size, 1, height, width)
    :param num_samples: Number of images to visualize
    :param save_path: Path to save the visualization
    """
    num_samples = min(num_samples, len(predictions))

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))  # Two columns: GT | Prediction

    if num_samples == 1:
        axes = [axes]  # Convert single pair of axes into a list

    save_path = f"visualizations/visualization-{index}.png"

    cmap_continuous = plt.cm.viridis  # Color map for visualization

    for i in range(num_samples):
        # --- Ground Truth ---
        gt = ground_truth[i].cpu().squeeze().numpy()  # Convert to (H, W)
        ax_gt = axes[i][0]
        im_gt = ax_gt.imshow(gt, cmap=cmap_continuous, interpolation="nearest")
        ax_gt.set_title(f"Ground Truth (Pixel-wise)")
        ax_gt.axis("off")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        # --- Prediction ---
        pred = predictions[i].detach().cpu().squeeze().numpy() 
         # Convert to (C, H, W) or (H, W)

        # If prediction has 3 channels, take mean or choose a single channel
        if pred.ndim == 3 and pred.shape[0] == 3:
            pred = pred.mean(axis=0) 
            
        pred = (pred - pred.min()) / (pred.max() - pred.min()) 
        pred= pred*255# Convert to grayscale (H, W)

        ax_pred = axes[i][1]
        im_pred = ax_pred.imshow(pred, cmap=cmap_continuous, interpolation="nearest")
        ax_pred.set_title(f"Prediction (Pixel-wise)")
        ax_pred.axis("off")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()








def evaluate_model_Germany_Two(model: nn.Module, data_loader, thresholds, criterion, device,
                                normalization=None) -> Tuple[float, float, np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param model:
    :param data_loader:
    :param thresholds:
    :param criterion:
    :param device:
    :param normalization:
    :return: confusion, binary_metrics_by_threshold
    """

    n_thresholds = len(thresholds)
    n_classes = n_thresholds + 1
    total_loss = 0
    total_samples = 0

    model.eval()

    # Run inference on single epoch
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    metrics_by_threshold = defaultdict(list)
    for i, (images, target, t) in enumerate(tqdm(data_loader)):


        if normalization:
            with torch.no_grad():
                for i, (max_val, min_val) in enumerate(zip(normalization['max_values'], normalization['min_values'])):
                    if min_val < 0:
                        images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                    else:
                        images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

        images = images.float().to(device)
        target = target.long().to(device)
        output,output2 = model(images, t)
        timestamps = None
        loss, _, _ = criterion(output, output2, target, timestamps, mode="train")
        # visualize_predictions(output, target, index=i) # Visualization is called here, using output2 and target!
        if loss is None:  # hotfix for None return values from losses.CrossEntropyLoss
            continue

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]

        predictions = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)[1]  # (batch_size, height, width)
        step_confusion, step_metrics_by_threshold = compile_metrics_Germany(data_loader.dataset, predictions.numpy(),
                                                                            target.detach().cpu().numpy(), thresholds)

        confusion += step_confusion
        for threshold, metrics in step_metrics_by_threshold.items():
            metrics_by_threshold[threshold].append(metrics)

    metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
    correct = (confusion[np.diag_indices_from(confusion)]).sum()
    accuracy = correct / confusion.sum()
    loss = total_loss / total_samples

    return accuracy, loss, confusion, metrics_by_threshold


def evaluate_model_Two(model: nn.Module, data_loader, thresholds, criterion, device,
                                normalization=None) -> Tuple[float, float, np.ndarray, Dict[float, pd.DataFrame]]:
    """
    :param model:
    :param data_loader:
    :param thresholds:
    :param criterion:
    :param device:
    :param normalization:
    :return: confusion, binary_metrics_by_threshold
    """
    dataset = data_loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, StandardDataset):
        raise ValueError('data_loader must contain a (subset of) StandardDataset')

    n_thresholds = len(thresholds)
    n_classes = n_thresholds + 1
    total_loss = 0
    total_samples = 0

    model.eval()

    # Run inference on single epoch
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    metrics_by_threshold = defaultdict(list)
    for i, (images, target, t) in enumerate(tqdm(data_loader)):
        # Note, StandardDataset retrieves timestamps in Tensor format due to collation issue, for now
        timestamps = []
        for e in t:
            origin = datetime(year=e[0], month=e[1], day=e[2], hour=e[3])
            lead_time = e[4].item()
            timestamps.append((origin, lead_time))

        if normalization:
            with torch.no_grad():
                for i, (max_val, min_val) in enumerate(zip(normalization['max_values'], normalization['min_values'])):
                    if min_val < 0:
                        images[:, :, i, :, :] = images[:, :, i, :, :] / max(-min_val, max_val)
                    else:
                        images[:, :, i, :, :] = (images[:, :, i, :, :] - min_val) / (max_val - min_val)

        images = images.float().to(device)
        target = target.long().to(device)
        output,output2 = model(images, t)
        loss, _, _ = criterion(output, output2,  target, timestamps, mode="train")

        if loss is None:  # hotfix for None return values from losses.CrossEntropyLoss
            continue

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
        predictions = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)[1]  # (batch_size, height, width)
        predictions = predictions.numpy()
        step_confusion, step_metrics_by_threshold = compile_metrics(data_loader.dataset, predictions, timestamps,
                                                                    thresholds)
        confusion += step_confusion
        for threshold, metrics in step_metrics_by_threshold.items():
            metrics_by_threshold[threshold].append(metrics)

    metrics_by_threshold = {t: pd.concat(metrics) for t, metrics in metrics_by_threshold.items()}
    correct = (confusion[np.diag_indices_from(confusion)]).sum()
    accuracy = correct / confusion.sum()
    loss = total_loss / total_samples

    return accuracy, loss, confusion, metrics_by_threshold