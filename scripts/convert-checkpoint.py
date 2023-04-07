import torch
from collections import OrderedDict


checkpoint = torch.load("../models/sceneego/checkpoints/6.pth.tar")

state_d = OrderedDict()

# copy weights
network_key = "backbone"
for key, value in checkpoint['state_dict'].items():
    if network_key in key:
        state_d[key] = value

# remove redundant state_dict keys
remove_prefix = network_key+"."
state_d = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_d.items()}

# save 
checkpoint_dir = "../models/sceneego/checkpoints"
torch.save( OrderedDict({"state_dict": state_d}), f"{checkpoint_dir}/{network_key}-6.pth.tar" )
