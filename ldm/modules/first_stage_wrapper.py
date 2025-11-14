import torch
import pytorch_lightning as pl
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

# Import the VAE model
from ..models.vanilla_vae_2d import VanillaVAE2D


class VAEWrapper(pl.LightningModule):
    def __init__(self, ckpt_path, vae_params):
        super().__init__()
        # Instantiate the VAE model from the other repository
        self.vae = VanillaVAE2D(**vae_params)
        
        # Load the state dict
        self.init_from_ckpt(ckpt_path)

        # This model should not be trained, so set to eval mode
        self.eval()
        self.freeze()

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        # Create a new state_dict with 'model.' prefix removed
        new_sd = {}
        for k, v in sd.items():
            if k.startswith('model.'):
                new_sd[k[len('model.'):]] = v
            else:
                new_sd[k] = v
        missing, unexpected = self.vae.load_state_dict(new_sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        # The VAE's encode method returns [mu, log_var]
        mu, log_var = self.vae.encode(x)
        # We need to wrap this in the DiagonalGaussianDistribution expected by LDM
        # The VAE from PyTorch-VAE returns a list [mu, log_var]
        # The DiagonalGaussianDistribution wants the parameters concatenated
        posterior = DiagonalGaussianDistribution(torch.cat([mu, log_var], dim=1))
        return posterior

    def decode(self, z):
        # The VAE's decode method takes the latent representation
        # The LDM framework will provide a sample 'z' from the posterior
        return self.vae.decode(z)

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
