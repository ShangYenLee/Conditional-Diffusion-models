import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from src.diffusion import *
from src.utils import *

#initial setting
parser = argparse.ArgumentParser()
parser.add_argument('--size',type=int,default=28,help='image size')
parser.add_argument('--out_dir',type=str,default='./checkpoint/pred')
parser.add_argument('--checkpoint',type=str,default='./checkpoint/ddpm_ep20.pt')
opt = parser.parse_args()
os.makedirs(opt.out_dir,exist_ok=True)
torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

#load model
model = torch.load(opt.checkpoint)
model.to(device)
diffusion = Diffusion(img_size=opt.size, device=device)


labels = torch.zeros(100).long().to(device)

for i in tqdm(range(10)):
    sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
    labels += 1
    for j in range(100):
        image = sampled_images[j,:,:,:]
        image = image.permute(1, 2, 0).to('cpu').numpy()
        image = Image.fromarray(image)
        image.save(os.path.join(opt.out_dir, f'{i}_{j+1:0>3}.png'))