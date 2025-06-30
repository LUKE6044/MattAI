import torch

state = torch.load("model.pth", map_location="cpu")
print("Chiavi nello state dict:")
for k, v in state.items():
    print(f"{k}: shape {tuple(v.shape)}")
