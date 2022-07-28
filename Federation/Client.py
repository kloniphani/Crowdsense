import json
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import flwr as fl


def load_local_data(data_dir, class_identifier):
	audio_files = glob(str(data_dir) + '/*/*')
	data_set = []
	for audio_file in tqdm(audio_files):
		audio_file = audio_file.split('\\')
		data_set.append({'name': audio_file[-1],
					'relative_path': '\{0}\{1}'.format(audio_file[-2], audio_file[-1]),
					'class': audio_file[-2],
					'classID': class_identifier[audio_file[-2]]})
		
	return(pd.DataFrame(data_set))

# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Spectogram as images
def save_spec_as_image(dataset, image_data_dir):
	from torchvision.utils import save_image
	with tqdm(total=int(len(dataset))) as pbar:
		for i, data in enumerate(dataset, 0):
			temp = data[2].split('\\')
			try:
				folder = image_data_dir/temp[1]
				folder.mkdir(parents=True, exist_ok=False)
			except FileExistsError:
				pass
			
			data = data[:,0,:,:]
			save_image(data[0], folder/'{0}.png'.format(temp[2][:-4]))
			pbar.update(1)

class CifarClient(fl.client.NumPyClient):
	def __init__(self, MODEL, CLASSES, name, train_dl, val_dl) -> None:
		super().__init__()

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(f"{self.device}" " is available.")		
		self.learning_rate = 1e-3
		self.batch_size = 8
		self.n_epochs = 3

		self.classes = CLASSES
		self.name = name

		self.model = MODEL.to(self.device)
		self.model.share_memory()
		self.optimastion = Optimisation(model=self.model, name=self.name, device=self.device, classes=self.classes, learning_rate=self.learning_rate)

		self.train_dl = train_dl
		self.val_dl = val_dl

	def get_parameters(self) -> List[np.ndarray]:
		return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

	def set_parameters(self, parameters) -> None:
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)

	def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
		self.set_parameters(parameters)
		self.optimastion.train(self.train_dl, self.val_dl, self.batch_size, self.n_epochs)
		return self.get_parameters(), len(self.train_dl), {}

	def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
		self.set_parameters(parameters)
		loss, accuracy = self.optimastion.inference(self.val_dl)
		return float(loss), len(self.val_dl), {"accuracy": float(accuracy)}

if __name__ == "__main__":
	from torch.utils.data import random_split, DataLoader
	from SoundDS import *
	from Model import *
	from Modelutil import *


	# ----------------------------
	# Prepare training data from Metadata file
	# ----------------------------
	import pandas as pd
	from pathlib import Path
	from glob import glob
	from tqdm import tqdm

	download_path = Path.cwd()/'Datasets'
	original_data_dir = download_path/'Audio'
	sample_data_dir = download_path/'Sample'
	image_data_dir = download_path/'Image'
	model_path = Path.cwd()/'Models/LeNet.pt'

	with open('class.json') as f:
		CLASSES = json.load(f)
		df = load_local_data(original_data_dir, CLASSES)
		print(df)
		
		AugmentedDATASET = AugmentedSoundDS(df, original_data_dir)
		RawDATASET = RawSoundDS(df, original_data_dir)

		# ----------------------------
		# Loading data from the local folders to Pytorch dataloader for training
		# ----------------------------
	
		# Random split of 90:10 between training and validation
		num_items = len(RawDATASET)
		num_train = round(num_items * 0.9)
		num_val = num_items - num_train
		train_ds, val_ds = random_split(RawDATASET, [num_train, num_val])

		num_items = len(RawDATASET)
		num_train = round(num_items * 0.9)
		num_val = num_items - num_train
		_, _ = random_split(AugmentedDATASET, [num_train, num_val])

		batch_size = 8
		# Create training and validation data loaders
		train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
		val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
		try:
			MODEL = torch.load(model_path)
		except Exception as a:
			MODEL = LeNet(len(CLASSES))
	
		CifarFederationClient = CifarClient(MODEL, CLASSES, 'LeNet', train_dl, val_dl)

		fl.client.start_numpy_client("0.0.0.0:5588", client=CifarFederationClient)