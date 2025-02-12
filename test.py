# import torch
# from train import evaluate
# from datasets.cifar10 import getCIFAR10Dataloaders
# from models.convnext import ConvNext


# print("testing")


# # Set device
# use_cuda_if_avail = True
# if use_cuda_if_avail and torch.cuda.is_available():
#     device = "cuda"
#     print("Using GPU for testing")
# else:
#     device = "cpu"

# print("testing")


# config = {
#     "bs":128,   # batch size
#     "lr":0.004, # learning rate
#     "l2reg":0.0000001, # weight decay
#     "max_epoch":200,
#     "blocks":[64,64,128,128,256,256,512,512]
# }

# # Load test data
# train_loader, val_loader, test_loader = getCIFAR10Dataloaders(config)

# print("loaded")

# # Load model
# model = ConvNext(3, 10, config["blocks"])
# model.to(device)
# print("model to")

# # Load best checkpoint
# # checkpoint = torch.load("best_checkpoint.pth", map_location=device)  # Adjust checkpoint path if needed
# # model.load_state_dict(checkpoint["model_state_dict"])

# best_checkpoint_path = "chkpts\ TDy1wu_CIFAR10 _epoch 199"
# checkpoint = torch.load(best_checkpoint_path, map_location=device)
# print(checkpoint.keys())
# print()

# state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
# print(state_dict.keys())  # Compare with model's state_dict keys
# print()

# print(model.state_dict().keys())
# print()


# # model.load_state_dict()

# print("checkpoint")


# # Evaluate on test set
# test_loss, test_acc = evaluate(model, test_loader)
# print("evaluated")


# # Print results
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
