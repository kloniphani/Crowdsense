import json
from turtle import down
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import torch.multiprocessing as mp

def load_torchvison_data(batch_size):
	# list all transformations
	transform = transforms.Compose(
		[transforms.ToTensor()])

	# download and load training dataset
	trainset = torchvision.datasets.MNIST(root='./data', train=True,
											download=True, transform=transform)
	train_dl = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=True, num_workers=2)

	# download and load testing dataset
	testset = torchvision.datasets.MNIST(root='./data', train=False,
										download=True, transform=transform)
	val_dl = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											shuffle=False, num_workers=2)

	return train_dl, val_dl

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
			
if __name__ == "__main__":
	from torch.utils.data import random_split, DataLoader
	from SoundDS import *
	from ImageDS import *
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

	with open(r'class.json') as f:
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
		num_train = round(num_items * 0.01)
		num_val = num_items - num_train
		train_ds , val_ds = random_split(RawDATASET, [num_train, num_val])

		num_items = len(AugmentedDATASET)
		num_train = round(num_items * 0.99)
		num_val = num_items - num_train
		aug_ds, _ = random_split(AugmentedDATASET, [num_train, num_val])

		# parameters
		num_processes = 4
		y_dim = 430
		x_dim = 768
		batch_size = 64
		n_epochs = 1
		learning_rate = 1e-3
		input_dim = 430
		output_dim = 23
		hidden_dim = 64
		layer_dim = 1
		dropout = 0.2
		weight_decay = 1e-6
		n_steps = 1
		n_channels = 1

		# Create training and validation data loaders
		train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
		val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

		train_dl, val_dl = load_torchvison_data(batch_size)

		# Device
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(f"{device}" " is available.")
		#MODEL = LeNet(output_dim).to(device)
		#MODEL = AE(output_size=output_dim, input_size=input_dim).to(device)
		#MODEL = LinearSVM(x_dim, y_dim).to(device)
		
		"""MODEL = LeNet(output_dim).to(device)
		if not torch.cuda.is_available():
			MODEL.share_memory()
		opt = Optimisation(model=MODEL, name='LeNet', device=device, classes=CLASSES, learning_rate=learning_rate)		
		opt.train(train_dl, val_dl, batch_size, n_epochs)
		opt.plot_losses()
		opt.inference(val_dl)
		opt.plot_confusion_matrix()
		
		MODEL = RNNModel(n_steps, n_channels, input_dim, hidden_dim, layer_dim, output_dim, device).to(device)
		if not torch.cuda.is_available():
			MODEL.share_memory()
		opt = Optimisation(model=MODEL, name='RNN', device=device, classes=CLASSES, learning_rate=learning_rate)
		opt.train(train_dl, val_dl, batch_size, n_epochs)
		opt.plot_losses()
		opt.inference(val_dl)
		opt.plot_confusion_matrix()"""


		MODEL = GRUModel(input_dim, hidden_dim, layer_dim, output_dim, dropout).to(device)
		if not torch.cuda.is_available():
			MODEL.share_memory()
		opt = Optimisation(model=MODEL, name='GRU', device=device, classes=CLASSES, learning_rate=learning_rate)
		opt.train(train_dl, val_dl, batch_size, n_epochs)
		opt.plot_losses()
		opt.inference(val_dl)
		opt.plot_confusion_matrix()

		MODEL = BasicRNN(input_dim, output_dim, hidden_dim, layer_dim).to(device)
		if not torch.cuda.is_available():
			MODEL.share_memory()
		opt = Optimisation(model=MODEL, name='RNN', device=device, classes=CLASSES, learning_rate=learning_rate)
		opt.train(train_dl, val_dl, batch_size, n_epochs)
		opt.plot_losses()
		opt.inference(val_dl)
		opt.plot_confusion_matrix()

		MODEL = LSTMModel( input_dim, hidden_dim, layer_dim, output_dim, dropout).to(device)
		if not torch.cuda.is_available():
			MODEL.share_memory()
		opt = Optimisation(model=MODEL, name='LSTM', device=device, classes=CLASSES, learning_rate=learning_rate)
		opt.train(train_dl, val_dl, batch_size, n_epochs)
		opt.plot_losses()
		opt.inference(val_dl)
		opt.plot_confusion_matrix()


