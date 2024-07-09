import torch

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic=True
    DEVICE=torch.device('cuda')
else:
    DEVICE=torch.device('cpu')

print(DEVICE)