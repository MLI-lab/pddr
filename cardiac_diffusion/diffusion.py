import torch
from torch import nn


class Diffusion(nn.Module):
    def __init__(
            self,
            restoration_fn,
            *,
            timesteps:int=1000,
            schedule='cosine',
            noise_based=False
    ):
        super().__init__()
        self.restoration_fn = restoration_fn
        self.timesteps = timesteps
        self.schedule = schedule
        self.noise_based = noise_based

        self.betas = self.beta_schedule()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))        

    def beta_schedule(self, s = 0.008):
        """
        Beta schedule for the diffusion process.
        """
        if self.schedule == 'linear':
            # linear schedule implementation from 'Improved Denoising Diffusion Probabilistic Models'
            scale = 1000 / self.timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, self.timesteps) 

        if self.schedule == 'cosine':
            # cosine schedule implementation from 'Cold Diffusion Models'
            steps = self.timesteps + 1
            x = torch.linspace(0, steps, steps)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0, 0.999) 
        
        raise NotImplementedError("Beta schedule not implemented.")
    
    def extract(self, alphas, t, x_shape):
        # implementation from 'Cold Diffusion Models'
        b, *_ = t.shape
        out = alphas.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def degrade(self, x, t, noise=None):
        """
        Degrades the input data x with scale t using a Gaussian noise.

        This implementation of the `degrade` method uses a Gaussian noise to
        degrade the input data `x` with the scale `t`.

        Parameters:
            x: The input data to be degraded.
            t: The scale of degradation.
            noise: The Gaussian noise to be used for degradation. Defaults to resampling noise in every degradation step.

        Returns:
            The degraded input data.
        """
        if noise is None:
            noise = torch.randn_like(x)

        x_degraded = self.extract(self.sqrt_alphas_cumprod, t, x.shape) * x + \
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        
        return x_degraded

    @torch.no_grad()
    def sample(self, x, t, eta=0.95):
        """
        Sample from the diffusion model using DDIM.
        (Eq. 12 in Denoising Diffusion Implicit Models, Song et al. ICLR 2021)

        Parameters:
            x: The degraded input data.
            t: The scale of degradation.
            [eta]: For eta=0 this becomes deterministic sampling without stochastic noise, for eta=1 this becomes DDPM.
        
        Returns:
            A sample from the diffusion model.
        """
        x_current = x
        x_direct = None

        while t:
            estimation = self.restoration_fn(x_current, t)

            if self.noise_based:
                noise = estimation
                x_intermediate = (x_current - noise * self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)) / self.extract(self.sqrt_alphas_cumprod, t, x.shape)
            else:
                x_intermediate = estimation
                noise = (x_current -  self.extract(self.sqrt_alphas_cumprod, t, x.shape) * x_intermediate) / self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

            if x_direct is None:
                x_direct = x_intermediate

            sqrt_alpha_prev_time = self.extract(self.sqrt_alphas_cumprod, t-1, x.shape)
            sqrt_alpha_curr_time = self.extract(self.sqrt_alphas_cumprod, t, x.shape)
            
            sigma = ((1 - sqrt_alpha_prev_time.pow(2)) / (1 - sqrt_alpha_curr_time.pow(2))).sqrt() * \
                        (1 - sqrt_alpha_curr_time.pow(2) / sqrt_alpha_prev_time.pow(2)).sqrt()
            if sigma.isnan().any():
                sigma = torch.zeros_like(sigma, device= noise.device)
            sigma = sigma * eta

            if t == 1:
                sigma = torch.zeros_like(sigma, device= noise.device)
            # print(f't: {t.item()}, sigma: {sigma.item()}')

            scaled_estimate =  sqrt_alpha_prev_time * x_intermediate
            deterministic_noise = torch.sqrt(1 - sqrt_alpha_prev_time.pow(2) - sigma.pow(2)) * noise
            stochastic_noise = sigma * torch.randn_like(x_intermediate)

            x_current = scaled_estimate + deterministic_noise + stochastic_noise

            t -= 1

        return x_direct, x_current

    def forward(self, x, t=None, noise=None):
        b, c, f, h, w, device = *x.shape, x.device

        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=device)

        x_degraded = self.degrade(x, t, noise=noise)
        estimation = self.restoration_fn(x_degraded, t)
        
        return estimation
