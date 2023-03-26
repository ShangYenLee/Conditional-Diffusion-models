import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from src.transforms import *

class mnistm_dataset(Dataset):
	def __init__(self, opt, type, transform=None):
		super(P2_dataset,self).__init__()
		self.root = opt.data_root
		self.info = pd.read_csv(os.path.join(self.root,f'{type}.csv'))
		self.filename = self.info['image_name']
		self.label = self.info['label']
		self.transform = transform

	def __len__(self):
		return len(self.label)

	def __getitem__(self,i):
		label = int(self.label[i])
		filename = self.filename[i]
		data = {
			'image': os.path.join(self.root,'data',filename),
			'label': label
		}

		if self.transform is not None:
			data = self.transform(data)

		return data


def get_digits_dataset(opt, type):
	train_transform = transforms.Compose([
		LoadImg(keys=['image']),
		totensor(keys=['image']),
		ImgNormal(keys=['image']),
		])

	train_set = mnistm_dataset(opt,type,train_transform)

	return train_set

