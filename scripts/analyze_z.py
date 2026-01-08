import argparse
import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import torchvision
from omegaconf import OmegaConf
from scipy import stats
from PIL import Image

# Add the root directory to the Python path to allow for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ldm.util import instantiate_from_config
from main import DataModuleFromConfig # Assuming DataModuleFromConfig is in main.py


def get_parser():
    parser = argparse.ArgumentParser(description="Analyze latent vectors (z) and generate diagnostic images.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the LDM config file (e.g., afhq-ldm-final-fix-attempt.yaml)",
    )
    parser.add_argument(
        "--z_pool_path",
        type=str,
        default="diagnostics/z_pool.npy",
        help="Path to the collected latent vectors .npy file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="diagnostics",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--max_images_to_decode",
        type=int,
        default=8,
        help="Maximum number of images to decode for recon_vs_random and empirical_prior_samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on ('cuda' or 'cpu').",
    )
    return parser

# Helper to save images
def save_image_grid(images, filename, nrow=4):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    # Scale to 0-1 and convert to PIL Image
    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1) # C H W -> H W C
    grid = (grid.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(grid).save(filename)


if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()

    os.makedirs(opt.outdir, exist_ok=True)
    recon_vs_random_dir = os.path.join(opt.outdir, "recon_vs_random")
    os.makedirs(recon_vs_random_dir, exist_ok=True)
    empirical_prior_samples_dir = os.path.join(opt.outdir, "empirical_prior_samples")
    os.makedirs(empirical_prior_samples_dir, exist_ok=True)
    visualizations_dir = os.path.join(opt.outdir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    device = torch.device(opt.device)

    # Load config
    config = OmegaConf.load(opt.config)

    # Load VAE model
    print("Instantiating VAE model...")
    vae_model = instantiate_from_config(config.model.params.first_stage_config)
    
    # Manually load checkpoint (similar to collect_z.py)
    if 'ckpt_path' in config.model.params.first_stage_config.params:
        ckpt_path = config.model.params.first_stage_config.params.ckpt_path
        print(f"Loading VAE weights from checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        
        vae_sd = {}
        for k, v in sd.items():
            if k.startswith("first_stage_model."):
                k = k.replace("first_stage_model.", "", 1)
                vae_sd[k] = v
        
        if not vae_sd: # Fallback for pure VAE checkpoint
             vae_sd = {k: v for k, v in sd.items() if k.startswith("encoder") or k.startswith("decoder") or k.startswith("quant_conv") or k.startswith("post_quant_conv")}

        missing, unexpected = vae_model.load_state_dict(vae_sd, strict=False)
        print(f"VAE Restored: {len(missing)} missing keys, {len(unexpected)} unexpected keys.")

    vae_model.to(device)
    vae_model.eval()

    # Load z_pool
    print(f"Loading z_pool from {opt.z_pool_path}...")
    z_pool_np = np.load(opt.z_pool_path)
    z_pool = torch.from_numpy(z_pool_np).float().to(device)
    latent_dim = z_pool.shape[1]

    # --- 1. Compute and save z_stats.json ---
    print("Computing z_pool statistics...")
    per_dim_mean = z_pool.mean(axis=0).cpu().numpy().tolist()
    per_dim_std = z_pool.std(axis=0).cpu().numpy().tolist()
    norms = torch.linalg.norm(z_pool, axis=1).cpu().numpy().tolist()
    
    # Kurtosis can be slow for large datasets, sample a subset if z_pool is very large
    if z_pool_np.shape[0] > 100000: # Arbitrary large number
        kurt_subset = z_pool_np[np.random.choice(z_pool_np.shape[0], 100000, replace=False)]
        per_dim_kurtosis = stats.kurtosis(kurt_subset, axis=0).tolist()
    else:
        per_dim_kurtosis = stats.kurtosis(z_pool_np, axis=0).tolist()
    
    z_stats = {
        "num_samples": z_pool.shape[0],
        "latent_dim": latent_dim,
        "per_dim_mean": per_dim_mean,
        "per_dim_std": per_dim_std,
        "norms_mean": np.mean(norms),
        "norms_std": np.std(norms),
        "per_dim_kurtosis": per_dim_kurtosis
    }
    z_stats_path = os.path.join(opt.outdir, "z_stats.json")
    with open(z_stats_path, 'w') as f:
        json.dump(z_stats, f, indent=4)
    print(f"Saved z_stats to {z_stats_path}")
    print(f"Average per-dim std: {np.mean(per_dim_std):.4f}")
    print(f"Average norm: {np.mean(norms):.4f} (Expected for N(0,I) is sqrt(latent_dim) = {np.sqrt(latent_dim):.4f})")


    # --- 2. Generate recon_vs_random images ---
    print("Generating recon_vs_random images...")
    data_module = instantiate_from_config(config.data)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.train_dataloader()
    
    input_images = []
    original_z_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = vae_model.get_input(batch, config.model.params.first_stage_key).to(device)
            input_images.append(inputs)
            
            posterior = vae_model.encode(inputs)
            original_z_list.append(posterior.sample())
            
            if len(input_images) * inputs.shape[0] >= opt.max_images_to_decode:
                break
    
    input_images = torch.cat(input_images, dim=0)[:opt.max_images_to_decode]
    original_z = torch.cat(original_z_list, dim=0)[:opt.max_images_to_decode]

    recon_images = vae_model.decode(original_z)
    save_image_grid(recon_images, os.path.join(recon_vs_random_dir, "recon_z.png"), nrow=opt.max_images_to_decode // 2)

    random_z = torch.randn_like(original_z)
    random_z_images = vae_model.decode(random_z)
    save_image_grid(random_z_images, os.path.join(recon_vs_random_dir, "random_z.png"), nrow=opt.max_images_to_decode // 2)
    
    # Sample from z_pool for empirical prior
    num_to_sample = min(opt.max_images_to_decode, z_pool.shape[0])
    indices = torch.randperm(z_pool.shape[0])[:num_to_sample]
    empirical_z_samples = z_pool[indices]
    empirical_z_images = vae_model.decode(empirical_z_samples)
    save_image_grid(empirical_z_images, os.path.join(recon_vs_random_dir, "empirical_z.png"), nrow=opt.max_images_to_decode // 2)

    # Save actual input images for comparison
    save_image_grid(input_images, os.path.join(recon_vs_random_dir, "input_images.png"), nrow=opt.max_images_to_decode // 2)

    # --- 3. Generate empirical_prior_samples ---
    print("Generating empirical_prior_samples...")
    save_image_grid(empirical_z_images, os.path.join(empirical_prior_samples_dir, "empirical_prior_samples.png"), nrow=opt.max_images_to_decode // 2)

    # --- 4. Generate z_histograms ---
    print("Generating z_histograms...")
    plt.figure(figsize=(15, 10))
    for i in range(latent_dim):
        plt.hist(z_pool[:, i].cpu().numpy(), bins=50, alpha=0.7, label=f'Dim {i+1}')
    plt.title('Histograms of Latent Dimensions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(visualizations_dir, "z_histograms.png"))
    plt.close()
    
    # Optional: PCA/t-SNE - requires scikit-learn and potentially long computation
    # Skipping for now to prioritize core debugging outputs
    print("Skipping PCA/t-SNE for now. Can be added later if needed.")

    print("\nAnalysis complete. Check the 'diagnostics' directory for outputs.")
