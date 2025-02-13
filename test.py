import torch
from train import evaluate
from datasets.cifar10 import getCIFAR10Dataloaders
from models.convnext import ConvNext

print("Testing script started")

# Set device
use_cuda_if_avail = True
device = "cuda" if use_cuda_if_avail and torch.cuda.is_available() else "cpu"
print(f"Using {device.upper()} for testing")

# Configuration
config = {
    "bs": 128,  # batch size
    "lr": 0.004,  # learning rate
    "l2reg": 0.0000001,  # weight decay
    "max_epoch": 200,
    "blocks": [64, 64, 128, 128, 256, 256, 512, 512]
}

# Load test data
train_loader, val_loader, test_loader = getCIFAR10Dataloaders(config)
print("Data loaded successfully")

# Load model
model = ConvNext(3, 10, config["blocks"])
model.to(device)
print("Model moved to device")

# Load best checkpoint
best_checkpoint_path = "chkpts/2zIUEn_CIFAR10_epoch 199"
try:
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    print("Checkpoint loaded successfully")

    # Handle checkpoint loading
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel or DistributedDataParallel wrapping
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load state dictionary into the model
    model.load_state_dict(state_dict)
    print("Model state dictionary loaded successfully")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit()

# Set model to evaluation mode
model.eval()

# Evaluate on test set
test_loss, test_acc = evaluate(model, test_loader)

print("Evaluation completed")

# Print results
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")