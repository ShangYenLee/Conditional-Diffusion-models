from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import torch

class LoadImg():
	def __init__(self,keys):
		self.keys = keys

	def __call__(self,data):
		for key in self.keys:
			if key in data:
				data[key] = Image.open(data[key])
				data[key] = np.array(data[key])

			else:
				raise KeyError(f'{key} is not a key of data')

		return data

class rand_crop():
	def __init__(self,keys,size):
		self.keys = keys
		self.size = size

	def __call__(self,data):
		crop_params = transforms.RandomCrop.get_params(data['image'],(self.size,self.size))
		for key in self.keys:
			if key in data:
				data[key] = F.crop(data[key],*crop_params)

			else:
				raise KeyError(f'{key} is not a key of data')

		return data

class rand_HFlip():
	def __init__(self,keys):
		self.keys = keys

	def __call__(self,data):
		if random.random() > 0.5:
			for key in self.keys:
				if key in data:
					data[key] = F.hflip(data[key])
				else:
					raise KeyError(f'{key} is not a key of data')					

		return data

class rand_VFlip():
	def __init__(self,keys):
		self.keys = keys

	def __call__(self,data):
		if random.random() > 0.5:
			for key in self.keys:
				if key in data:
					data[key] = F.vflip(data[key])
				else:
					raise KeyError(f'{key} is not a key of data')					

		return data

class ImgNormal():
	def __init__(self,keys):
		self.keys = keys

	def __call__(self,data):
		for key in self.keys:
			if key in data:
				data[key] = transforms.Normalize((0.5,),(0.5,))(data[key])
			else:
				raise KeyError(f'{key} is not a key of data')

		return data

class Noise():
	def __init__(self,keys):
		self.keys = keys

	def __call__(self,data):
		for key in self.keys:
			if key in data:
				data[key] = transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))(data[key])
			else:
				raise KeyError(f'{key} is not a key of data')

		return data

class rand_rot90():
	def __init__(self,keys):
		self.keys = keys

	def __call__(self,data):
		if random.random() > 0.5:
			for key in self.keys:
				if key in data:
					data[key] = torch.rot90(data[key],k=1,dims=(0,1))
				else:
					raise KeyError(f'{key} is not a key of data')					

		return data


class totensor():
	def __init__(self,keys):
		self.keys = keys

	def __call__(self,data):
		for key in self.keys:
			if key in data:
				data[key] = transforms.ToTensor()(data[key])

			else:
				raise KeyError(f'{key} is not a key of data')

		return data

class resize():
	def __init__(self,keys,size):
		self.keys = keys
		self.size = size

	def __call__(self,data):
		for key in self.keys:
			if key in data:
				data[key] = transforms.Resize([self.size,self.size])(data[key])

			else:
				raise KeyError(f'{key} is not a key of data')

		return data
