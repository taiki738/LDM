import argparse
import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add the root directory to the Python path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ldm.util import instantiate_from_config

def get_parser():
    parser = argparse.ArgumentParser(description="Collect latent vectors (z) from a VAE.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the LDM config file (e.g., afhq-ldm-final-fix-attempt.yaml)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="diagnostics",
        help="Directory to save the output .npy file.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum number of samples to collect.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on ('cuda' or 'cpu').",
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()

    # Create output directory
    os.makedirs(opt.outdir, exist_ok=True)
    output_path = os.path.join(opt.outdir, "z_pool.npy")

    print(f"Loading config from: {opt.config}")
    config = OmegaConf.load(opt.config)

    # Load VAE model (first_stage_model)
    print("Instantiating VAE model...")
    vae_model = instantiate_from_config(config.model.params.first_stage_config)
    
    # Manually load checkpoint as init_from_ckpt is not always called
    if 'ckpt_path' in config.model.params.first_stage_config.params:
        ckpt_path = config.model.params.first_stage_config.params.ckpt_path
        print(f"Loading VAE weights from checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        
        # The VAE was trained within an LDM, which might have a "first_stage_model." prefix
        # We need to filter the state dict to only load the VAE weights
        vae_sd = {}
        for k, v in sd.items():
            if k.startswith("first_stage_model."):
                # remove prefix
                k = k.replace("first_stage_model.", "", 1)
                vae_sd[k] = v
        
        if not vae_sd: # If no keys started with the prefix, maybe it's a raw VAE checkpoint
             vae_sd = {k: v for k, v in sd.items() if k.startswith("encoder") or k.startswith("decoder") or k.startswith("quant_conv") or k.startswith("post_quant_conv")}

        missing, unexpected = vae_model.load_state_dict(vae_sd, strict=False)
        print(f"VAE Restored: {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
        if len(missing) > 0:
            print("Missing keys:", missing)
        if len(unexpected) > 0:
            print("Unexpected keys:", unexpected)

    device = torch.device(opt.device)
    vae_model.to(device)
    vae_model.eval()

    # Load dataset
    print("Instantiating DataModule...")
    data_module = instantiate_from_config(config.data)
    data_module.prepare_data()
    data_module.setup()
    
    dataloader = data_module.train_dataloader()
    print(f"Dataloader created. Number of batches: {len(dataloader)}")

    # Collect latent vectors
    z_list = []
    total_samples = 0
    
    print(f"Starting to collect latent vectors (max: {opt.max_samples})...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if total_samples >= opt.max_samples:
                break
            
            # Assuming the batch is a dictionary with the image key
            # Following the logic in AutoencoderKL.get_input
            inputs = vae_model.get_input(batch, config.model.params.first_stage_key).to(device)
            
            # Encode and get the posterior distribution
            posterior = vae_model.encode(inputs)
            # Sample from the posterior
            z = posterior.sample()
            
            z_list.append(z.cpu())
            
            total_samples += z.shape[0]
            print(f"Batch {i+1}/{len(dataloader)} | Samples collected: {total_samples}", end='\r')

    print("\nCollection finished.")

    # Concatenate and save
    all_z = torch.cat(z_list, dim=0)
    if all_z.shape[0] > opt.max_samples:
        all_z = all_z[:opt.max_samples]

    print(f"Final shape of z_pool: {all_z.shape}")
    np.save(output_path, all_z.numpy())
    print(f"Saved latent vectors to {output_path}")
