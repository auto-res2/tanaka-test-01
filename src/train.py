import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .preprocess import DummyDataset

def get_spot_finetune_optimizer(model, lr=1e-4):
    """
    Create an optimizer that only updates parameters whose names include "adaln".
    In our dummy model, we simulate by using a substring match on the parameter name.
    """
    spot_params = [p for name, p in model.named_parameters() if "adaln" in name]
    if not spot_params:
        spot_params = model.parameters()
    optimizer = torch.optim.Adam(spot_params, lr=lr)
    return optimizer

def train_spot_finetune(model, dataloader, optimizer, num_steps=100, device="cpu"):
    """
    Dummy training loop for selective fine-tuning.
    Uses a mean-squared error loss between the output and input patch.
    """
    print("\n[Experiment 2] Starting spot-fine-tuning...")
    model.train()
    losses = []
    for step, (images, labels) in enumerate(dataloader):
        if step >= num_steps:
            break
        images = images.to(device)
        bs, C, H, W = images.shape
        resized = torch.nn.functional.interpolate(images, size=(model.patch_size, model.patch_size))
        t = torch.rand(bs, device=device)
        y = torch.randint(0, 10, (bs,), device=device)
        grid = torch.zeros(bs, 2, 1, device=device)
        mask = torch.ones(bs, 1, device=device)
        out = model.forward(resized, t, y, grid, mask, size=(model.patch_size, model.patch_size))
        loss = F.mse_loss(out, resized)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    plt.figure()
    plt.plot(losses, marker="o")
    plt.title("Spot Fine-Tuning Loss")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    pdf_filename = "training_loss_baseline.pdf"
    plt.savefig(pdf_filename, bbox_inches="tight")
    print(f"[Experiment 2] Training loss plot saved as {pdf_filename}")
    plt.close()
    return losses

def hybrid_pipeline(model, image, finetune_steps=20, lr=1e-4, device="cpu"):
    """
    Hybrid pipeline: first, do training-free processing; then, apply a few steps
    of spot fine-tuning on the given image; finally, process the image again.
    """
    from .evaluate import region_based_pipeline
    
    print("\n[Experiment 3] Running hybrid pipeline (training-free then incremental fine-tuning)...")
    result_initial = region_based_pipeline(image, model, patch_size=256, overlap=32, device=device)
    optimizer = get_spot_finetune_optimizer(model, lr)
    dummy_dataset = DummyDataset(image.unsqueeze(0), torch.tensor([1]))
    dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=True)
    train_spot_finetune(model, dummy_loader, optimizer, num_steps=finetune_steps, device=device)
    result_updated = region_based_pipeline(image, model, patch_size=256, overlap=32, device=device)
    return result_initial, result_updated
