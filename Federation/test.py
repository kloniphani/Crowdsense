import json
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

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

		# parameters
		input_dim = 28
		output_dim = 56
		hidden_dim = 100
		layer_dim = 1
		batch_size = 64
		dropout = 0.2
		n_epochs = 1
		learning_rate = 1e-3
		weight_decay = 1e-6
		n_steps = 1
		n_channels = 2

		# Create training and validation data loaders
		train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
		val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

		# Device
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(f"{device}" " is available.")

		MODEL = LeNet(output_dim)
		MODEL = MODEL.to(device)
		MODEL.share_memory()
		opt = Optimisation(model=MODEL, name='LeNet', classes=CLASSES, device=device, learning_rate=learning_rate)
		opt.train(train_loader=train_dl, val_loader=val_dl, batch_size=batch_size, num_epochs=n_epochs)
		opt.plot_losses()

		#predictions, values = opt.evaluate(val_dl, batch_size=batch_size, n_features=input_dim)