import json
import torch


if __name__ == "__main__":	
	from model import *
	from utility import *

	# ----------------------------
	# Prepare training data from Metadata file
	# ----------------------------
	import pandas as pd
	from pathlib import Path

	download_path = Path.cwd()/'Datasets'
	original_data_dir = download_path/'Audio'
	sample_data_dir = download_path/'Sample'
	image_data_dir = download_path/'Image'

	with open('class.json') as f:
		CLASSES = json.load(f)
			
		# parameters
		input_dim = 28
		output_dim = 56
		hidden_dim = 100
		layer_dim = 1
		
		dropout = 0.2
		n_epochs = 1
		learning_rate = 1e-3
		weight_decay = 1e-6
		n_steps = 1
		n_channels = 2

		# Device
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(f"{device}" " is available.")

		MODEL = LeNet(output_dim)
		MODEL = MODEL.to(device)
		if not torch.cuda.is_available():
			MODEL.share_memory()
		opt = Optimisation(model=MODEL, name='LeNet', data_dir=original_data_dir, classes=CLASSES, device=device, learning_rate=learning_rate)
		opt.train(num_epochs=n_epochs)
		opt.plot_losses()
		opt.inference()
		opt.plot_confusion_matrix()

