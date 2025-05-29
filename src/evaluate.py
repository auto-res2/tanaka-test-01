import torch
import numpy as np
import matplotlib.pyplot as plt
from .preprocess import split_image_regions, GatedRegionMerger

def region_based_pipeline(image, model, patch_size=256, overlap=32, device="cpu"):
    """
    Process a high-resolution image in overlapping regions:
     • Splits image.
     • Processes each patch through the model.
     • Merges results with a gated region merger.
    """
    print("\n[Experiment 1] Starting region-based pipeline...")
    regions = split_image_regions(image, patch_size, overlap)
    region_results = []
    for reg in regions:
        patch = reg['patch'].to(device)  # (C, patch_size, patch_size)
        patch = patch.unsqueeze(0)       # add batch dimension -> (1, C, patch_size, patch_size)
        t = torch.tensor([0.5], device=device)
        y = torch.tensor([1], device=device, dtype=torch.long)
        grid = torch.zeros(1, 2, 1, device=device)
        mask = torch.ones(1, 1, device=device)
        out = model.forward(patch, t, y, grid, mask, size=patch.shape[-2:])
        region_results.append((out, reg['coords']))
        print(f"Processed patch at coords {reg['coords']}")
    merger = GatedRegionMerger(in_channels=region_results[0][0].shape[1], hidden_channels=64).to(device)
    full_result = merger(region_results, output_size=image.shape[-2:])
    print("[Experiment 1] Completed merging of regions.")
    return full_result

def training_free_pipeline(model, image, device="cpu"):
    """
    Training-free pipeline for ultra-high-resolution image processing.
    Uses region-based pipeline without fine-tuning.
    """
    print("\n[Experiment 3] Running training-free pipeline...")
    result = region_based_pipeline(image, model, patch_size=256, overlap=32, device=device)
    return result

def evaluate_pipeline(model, dataloader, pipeline_fn, device="cpu"):
    """
    Evaluate a given pipeline on a dataset and return a dummy score.
    (Here we simply average random scores to mimic an evaluation metric.)
    """
    print("\n[Evaluation] Running pipeline evaluation...")
    scores = []
    for image, label in dataloader:
        image = image.to(device)
        with torch.no_grad():
            pred = pipeline_fn(model, image, device=device)
        scores.append(np.random.rand())
    score_mean = np.mean(scores)
    print(f"[Evaluation] Dummy evaluation score: {score_mean:.4f}")
    return score_mean

def plot_comparison(metric_values, labels, title, filename):
    """Plot comparison metrics and save as PDF."""
    plt.figure()
    for val, lab in zip(metric_values, labels):
        plt.plot(val, marker="o", label=lab)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Dummy Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    print(f"[Plot] Saved plot as {filename}")
    plt.close()
