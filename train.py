import os
import csv
import argparse
import datetime
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from src.dataset import get_digits_dataset
from src.diffusion import *
from src.models import *
from src.utils import *

parser = argparse.ArgumentParser()

parser.add_argument('-bs',type=int,default=100,help='batch size')
parser.add_argument('-ep',type=int,default=300,help='epoch')
parser.add_argument('-lr',type=float,default=3e-4,help='learning rate')
parser.add_argument('-dp','--dropout',type=float,default=0.2,help='dropout')
parser.add_argument('--size',type=int,default=28,help='image size')
parser.add_argument('--in_channels',type=int,default=3,help='input channels')
parser.add_argument('--num_classes',type=int,default=10,help='number of class')
parser.add_argument('--topk',type=int,default=5,help='top k checkpoint')
parser.add_argument('--step_size',type=int,default=5,help='step size')
parser.add_argument('--save_dir',type=str,default='./checkpoint')
parser.add_argument('--data_root',type=str,default='./mnistm')
parser.add_argument('--num_workers',type=int,default=4,help='num_workers')

opt = parser.parse_args()

def train():
	ckpt_loc = os.path.join(opt.save_dir,f'{datetime.today().strftime("%m-%d-%H-%M-%S")}_DDPM')
	mod_loc = os.path.join(ckpt_loc,'model')
	img_loc = os.path.join(ckpt_loc,'generate')
	os.makedirs(ckpt_loc,exist_ok=True)
	os.makedirs(mod_loc,exist_ok=True)
	os.makedirs(img_loc,exist_ok=True)
	csv_file = open(os.path.join(ckpt_loc,'result.csv'),mode='w',newline='')
	writer = csv.writer(csv_file)
	writer.writerow(['epoch','Loss'])
	torch.manual_seed(1)
	device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

	#model
	model = UNet_conditional(num_classes=opt.num_classes).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
	criterion = nn.MSELoss()
	diffusion = Diffusion(img_size=opt.size, device=device)
	
	train_set = get_digits_dataset(opt,'train') ##need to modify

	train_loader = DataLoader(
		train_set,
		batch_size=opt.bs,
		num_workers=opt.num_workers,
		shuffle=True
	 	)
    

	for epoch in range(1,opt.ep+1):
		train_bar = tqdm(train_loader)
		for i, data in enumerate(train_bar):
			images = data['image'].to(device)
			labels = data['label'].to(device)
			t = diffusion.sample_timesteps(images.shape[0]).to(device)
			x_t, noise = diffusion.noise_images(images, t)
			if np.random.random() < 0.1:
				labels = None
			predicted_noise = model(x_t, t, labels)
			loss = criterion(noise, predicted_noise)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			train_bar.set_postfix(MSE=loss.item())
		writer.writerow([f'Training {epoch}',loss.item()
		])

		if epoch % 5 == 0:
			labels = torch.arange(10).long().to(device)
			sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
			save_images(sampled_images, os.path.join(img_loc, f"{epoch}.png"),nrow=10)
			torch.save(model, os.path.join(mod_loc, f"{epoch:0>4}ckpt.pt"))

if __name__ == '__main__':
    train()