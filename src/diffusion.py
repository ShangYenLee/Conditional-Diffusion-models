import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image


class Diffusion:
    def __init__(self, noise_steps=400, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"): #modify noise_steps 1000 to 500
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        a = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * a, a

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            #generate_bar = tqdm(reversed(range(1, self.noise_steps)), position=0, desc=f'Generating')
            #first_zero = torch.tensor([]).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

                #save first_zero
                # if i == 399 or i == 320 or i == 240 or i == 160 or i == 80 or i == 1:
                #     first_zero = torch.cat((first_zero,x[0,:,:,:].unsqueeze(0)),0)
            #generate_bar.close()
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        # first_zero = (first_zero.clamp(-1, 1) + 1) / 2
        # first_zero = (first_zero * 255).type(torch.uint8)
        # for i in range(first_zero.shape[0]):
        #     image = first_zero[i,:,:,:]
        #     image = image.permute(1, 2, 0).to('cpu').numpy()
        #     image = Image.fromarray(image)
        #     image.save(os.path.join('./fig', f'first_zero_{i}.png'))

        return x