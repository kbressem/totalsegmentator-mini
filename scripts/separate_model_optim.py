import argparse
import torch
from monai.networks.nets import UNet

parser = argparse.ArgumentParser(description="Separate model state dicts and optimizer state dicts.")
parser.add_argument("--file", type=str, required=True, help="Path to the combined state dict")
parser.add_argument("--model_key", type=str, required=False, default="network", help="Key under which the model weights can be accessed in the state dict")
parser.add_argument("--optim_key", type=str, required=False, default="optimizer", help="Key under which the optimizer weights can be accessed in the state dict")

args = parser.parse_args()

if __name__ == "__main__":
    state_dict = torch.load(args.file)
    torch.save(state_dict["optimizer"], "optimizer.pt")
    torch.save(state_dict["network"], "model.pt")
    
    ### Verify    
    try: 
        network = UNet(
            spatial_dims= 3, 
            in_channels = 1, 
            out_channels = 105, 
            channels = [64, 128, 256, 512],
            strides = [2, 2, 2],
            num_res_units = 3
        )
    
        network.load_state_dict(torch.load("model.pt"))
    except Exception as e: 
        
        network_micro = UNet(
            spatial_dims = 3, 
            in_channels = 1, 
            out_channels = 105, 
            channels = [32, 64, 128, 256], 
            strides = [2, 2, 2], 
            num_res_units = 2
        )

        network.load_state_dict(torch.load("model.pt"))
     
