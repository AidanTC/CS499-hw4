import torch
from train import evaluate
from datasets.cifar10 import getCIFAR10Dataloaders
from models.convnext import ConvNext

# print("Testing script yo")

# Set device
use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
    # print("Using GPU")
else:
    device = "cpu"

# Configuration
config = {
    "bs":256,   # batch size
    "lr":0.004, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":200,
    "blocks":[64,64,128,128,256,256,512,512]
}

# Load test data
train_loader, val_loader, test_loader = getCIFAR10Dataloaders(config)

model = ConvNext(3, 10, config["blocks"])
model.to(device)

best_checkpoint_path = "chkpts/VKiHMc_CIFAR10_epoch 199"
checkpoint = torch.load(best_checkpoint_path, map_location=device)
# print("Checkpoint loaded ")

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
# print("loaded")

model.eval()
test_loss, test_acc = evaluate(model, test_loader)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")