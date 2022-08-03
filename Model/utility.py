
import enum
from tkinter import N
import seaborn as sn
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from torch.utils.data import random_split, DataLoader

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from glob import glob

from dataset import *


class Optimisation():
	"""Optimisation is a helper class that allows training, validation, prediction.

	Optimisation is a helper class that takes model, loss function, optimizer function
	learning scheduler (optional), early stopping (optional) as inputs. In return, it
	provides a framework to train and validate the models, and to predict future values
	based on the models.

	Attributes:
		model (RNNModel, LSTMModel, GRUModel, CNNModel): Model class created for the type of RNN
		loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
		optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
		train_losses (list[float]): The loss values from the training
		val_losses (list[float]): The loss values from the validation
		last_epoch (int): The number of epochs that the models is trained
		device (gpu, cpu): Main training unit
	"""
	def __init__(self, model, name, device, data_dir, classes, learning_rate=0.001):
		"""
		Args:
			model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
			loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
			optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
			device (gpu, cpu): Main training unit
		"""
		self.model = model
		self.name = name
		self.original_data_dir = data_dir 		
		self.train_losses = []
		self.val_losses = []
		self.train_acc = []
		self.val_acc = []
		self.device = device
		self.learning_rate = learning_rate
		self.classes = classes
		self.batch_size = 64
		self.worker = 4

		self.y_pred = []
		self.y_true = []

		# Loss Function, Optimizer and Scheduler
		self.loss_fn = nn.MSELoss(reduction="mean")
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate)

		self.dataset = self.load_local_data(self.original_data_dir, self.classes)
		print(self.dataset)

		#AugmentedDATASET = dataset(self.dataset, original_data_dir, True)
		RawDATASET = dataset(self.dataset, self.original_data_dir)
		self.mode = 'Raw'

		# ----------------------------
		# Loading data from the local folders to Pytorch dataloader for training
		# ----------------------------
	
		# Random split of 90:10 between training and validation
		num_items = len(RawDATASET)
		num_train = round(num_items * 0.9)
		num_val = num_items - num_train
		self.train_ds, self.val_ds = random_split(RawDATASET, [num_train, num_val])

		# Create training and validation data loaders
		self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)
		self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.worker)

	def load_local_data(self, data_dir, class_identifier):
		audio_files = glob(str(data_dir) + '/*/*')
		data_set = []
		for audio_file in tqdm(audio_files):
			audio_file = audio_file.split('\\')
			data_set.append({'name': audio_file[-1],
						'relative_path': '\{0}\{1}'.format(audio_file[-2], audio_file[-1]),
						'class': audio_file[-2],
						'classID': class_identifier[audio_file[-2]]})
			
		return(pd.DataFrame(data_set))

	def train_step(self, inputs, labels, batch_size):
		"""The method train_step completes one step of training.

		Given the features (inputs) and the target values (labels) tensors, the method completes
		one step of the training. First, it activates the train mode to enable back prop.
		After generating predicted values (labels) by doing forward propagation, it calculates
		the losses by using the loss function. Then, it computes the gradients by doing
		back propagation and updates the weights by calling step() function.

		Args:
			inputs (torch.Tensor): Tensor for features to train one step
			labels (torch.Tensor): Tensor for target values to calculate losses

		"""
		# Sets model to train mode
		self.model.train()

		# Makes predictions
		outputs = self.model(inputs)

		# Computes loss
		if self.name == 'LinearSVM':
			loss = torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size
		else:
			loss = self.criterion(outputs, labels)

		# Computes gradients
		loss.backward()

		# Updates parameters and zeroes gradients
		self.optimizer.step()
		self.optimizer.zero_grad()

		# Returns the loss
		return inputs, outputs, loss.item()

	def train(self, num_epochs=50):
		"""The method train performs the model training

		The method takes DataLoaders for training and validation datasets, batch size for
		mini-batch training, number of epochs to train, and number of features as inputs.
		Then, it carries out the training by iteratively calling the method train_step for
		num_epochs times. If early stopping is enabled, then it  checks the stopping condition
		to decide whether the training needs to halt before num_epochs steps. Finally, it saves
		the model in a designated file path.

		Args:
			train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
			val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
			batch_size (int): Batch size for mini-batch training
			num_epochs (int): Number of epochs, i.e., train steps, to train

		"""
		# Scheduler
		scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.learning_rate,
													steps_per_epoch=int(len(self.train_loader)),
													epochs=num_epochs,
													anneal_strategy='linear')

		

		lambda1 = lambda num_epochs: 0.95 ** num_epochs
		scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

		for epoch in range(1, num_epochs + 1):
			running_losses = []
			correct_predictions = []
			predictions = []
			records = []

			with tqdm(total=int(len(self.train_loader))) as pbar:
				for i, data in enumerate(self.train_loader):
					# Get the input features and target labels, and put them on the GPU
					inputs, labels = data[0].to(self.device), data[1].to(self.device)

					# Normalize the inputs
					inputs_m, inputs_s = inputs.mean(), inputs.std()
					inputs = (inputs - inputs_m) / inputs_s

					inputs, outputs, loss = self.train_step(inputs, labels, self.batch_size)					
					scheduler.step()

					# Keep stats for Loss and Accuracy
					running_losses.append(loss)					

					# Get the predicted class with the highest score
					_, prediction = torch.max(outputs, 1)
					# Count of predictions that matched the target label
					correct_predictions.append((prediction == labels).sum().item())
					predictions.append(prediction.shape[0])

					# keeping records of the predictions
					records.append({'data': data, 'predictions': prediction})

					pbar.update(1)

				# Stats at the end of the epoch
				num_batches = len(self.train_loader)
				avg_loss = np.sum(running_losses) / num_batches
				acc = (sum(correct_predictions)/sum(predictions)) * 100
				self.train_acc.append(acc)
				self.train_losses.append(avg_loss)
			
			batch_val_losses = []	
			batch_val_acc = []	
			validations = []
			with torch.no_grad():
				with tqdm(total=int(len(self.val_loader))) as pbar:								
					for  i, data in enumerate(self.val_loader):
						# Get the input features and target labels, and put them on the GPU
						inputs, labels = data[0].to(self.device), data[1].to(self.device)
						self.model.eval()
						yhat = self.model(inputs)
						val_loss = self.criterion(yhat, labels).item()

						batch_val_losses.append(np.sqrt(val_loss)/self.batch_size)

						# Get the validation class with the highest score
						_, validation = torch.max(yhat, 1)
						# Count of predictions that matched the target label
						batch_val_acc.append((validation == labels).sum().item())
						validations.append(validation.shape[0])

						pbar.update(1)

					validation_loss = np.mean(batch_val_losses)
					self.val_losses.append(validation_loss)
					val = (sum(batch_val_acc)/sum(validations)) * 100
					self.val_acc.append(val)

			
			self.train_loader = self.augmentation(records)
			
			print(f"[{epoch}/{num_epochs}] Training loss: {avg_loss:.4f}% \tTraining Accuracy: {acc:.4f}% \tValidation loss: {validation_loss:.4f}% \tValidation Accuracy: {val:.4f}%\n")

		from pathlib import Path
		model_dir = Path.cwd()/'model'/f'{datetime.now().strftime("%Y-%m-%d")}'
		try:
			model_dir.mkdir(parents=True, exist_ok=False)
		except:
			pass

		model_path = Path.cwd()/'model'/f'{self.name}-{self.mode}.pt'
		excel_path = model_dir/f'{self.name}_{str(self.device)[:3]}_e{num_epochs}_b{self.batch_size}_{datetime.now().strftime("%H-%M")}.xlsx'
		
		pd.DataFrame({'Training Accuracy': self.train_acc, 
			'Training Losses': self.train_losses, 
			'Validation Accuracy': self.val_acc, 
			'Validation Losses': self.val_losses}).to_excel(str(excel_path))
		
		torch.save(self.model, str(model_path))

	# ----------------------------
	# Inference
	# ----------------------------
	def inference (self):
		from tqdm import tqdm

		correct_prediction = 0
		total_prediction = 0
		loss = 0.0

		with tqdm(total=int(len(self.val_loader))) as pbar:
			# Disable gradient updates
			with torch.no_grad():
				for data in self.val_loader:
					# Get the input features and target labels, and put them on the GPU
					inputs, labels = data[0].to(self.device), data[1].to(self.device)

					# Normalize the inputs
					inputs_m, inputs_s = inputs.mean(), inputs.std()
					inputs = (inputs - inputs_m) / inputs_s

					# Get predictions
					self.model.eval()
					outputs = self.model(inputs)

					# Get the losses
					loss += self.criterion(outputs, labels).item()

					# Get the predicted class with the highest score
					_, prediction = torch.max(outputs, 1)
					self.y_pred.extend(prediction.cpu().numpy())
					self.y_true.extend(labels.cpu().numpy())

					# Count of predictions that matched the target label
					correct_prediction += (prediction == labels).sum().item()
					total_prediction += prediction.shape[0]

					pbar.update(1)
			
		acc = (correct_prediction/total_prediction) * 100
		print(f'Evaluation Accuracy: {acc:.4f}%, Total items: {total_prediction}\n\n')
		return loss, acc

	def plot_losses(self):
		"""The method plots the calculated loss values for training and validation
		"""

		from pathlib import Path
		model_dir = Path.cwd()/'model'/f'{self.mode}'/f'{datetime.now().strftime("%Y-%m-%d")}'
		try:
			model_dir.mkdir(parents=True, exist_ok=False)
		except:
			pass

		image_path = model_dir/f'{self.name}_{str(self.device)[:3]}_{datetime.now().strftime("%H-%M")}.png'


		plt.figure()
		plt.plot(self.train_acc, label="Training Accuracy")
		plt.plot(self.train_losses, label="Training Loss")
		plt.plot(self.val_acc, label="Validation Accuracy")
		plt.plot(self.val_losses, label="Validation Loss")
		plt.legend()
		plt.title("Model Statistics")
		plt.xlabel("Epoch #")
		plt.ylabel("Percentage")		
		plt.savefig(str(image_path))
		#plt.show()
		plt.close()

	def plot_confusion_matrix(self):
		from pathlib import Path
		model_dir = Path.cwd()/'model'/f'{self.mode}'/f'{datetime.now().strftime("%Y-%m-%d")}'
		try:
			model_dir.mkdir(parents=True, exist_ok=False)
		except:
			pass

		image_path = model_dir/f'Matrix_{self.name}_{str(self.device)[:3]}_{datetime.now().strftime("%H-%M")}.png'
		report_path = model_dir/f'Report_{self.name}_{str(self.device)[:3]}_{datetime.now().strftime("%H-%M")}.json'


		cf_matrix = confusion_matrix(self.y_true, self.y_pred, labels=np.unique([value for key, value in self.classes.items()]))
		df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index=[key for key, value in self.classes.items()], columns=[key for key, value in self.classes.items()])

		plt.figure(figsize=(10, 8))
		plt.title("Confusion Matrix")
		
		sn.set(rc = {'figure.figsize':(4,4)})
		ax = sn.heatmap(data=df_cm, annot=False)		

		plt.ylabel('True label')
		plt.xlabel('Predicted label')

		plt.tight_layout()
		plt.savefig(str(image_path))
		#plt.show()
		plt.close()


		print('Confusion matrix : \n', cf_matrix)
		"""
		returned = cf_matrix.ravel()
		if len(returned) == 4:
			tp, fn, fp, tn = returned
			print('Outcome values : \n', tp, fn, fp, tn)
		else:
			print('Outcome values : \n')
			for var in returned:
				print(var)
		"""
		# classification report for precision, recall f1-score and accuracy
		matrix = classification_report(self.y_true, self.y_pred, labels=np.unique([value for key, value in self.classes.items()]))
		print('Classification report : \n', matrix)
		import json
		with open(report_path, "w") as outfile:
			matrix = classification_report(self.y_true, self.y_pred, labels=np.unique([value for key, value in self.classes.items()]), output_dict=True)
			outfile.write(json.dumps(matrix, indent = 4))

	# ----------------------------
	# Augmentation
	# ----------------------------
	def augmentation (self, records):
		from tqdm import tqdm
		from pathlib import Path
		from torch.utils.data import DataLoader, ConcatDataset
		from dataset import dataset

		data_frame = pd.DataFrame()

		with torch.no_grad():
			with tqdm(total=int(len(records))) as pbar:
				for record in records:
					df = pd.DataFrame()
					df['classID'] = record['data'][1]
					df['relative_path'] = record['data'][2]
					df['class'] = df['relative_path'].apply(lambda x: str(x).split('\\')[1])
					df['name'] = df['relative_path'].apply(lambda x: str(x).split('\\')[2])
					df['prediction'] = record['predictions'].cpu().detach().numpy()
					df = df[df['classID']!=df['prediction']]
					del df['prediction']

					data_frame = pd.concat([data_frame, df])
					pbar.update(1)

			AugmentedDATASET = dataset(data_frame, self.original_data_dir, True)
			NewDATASET = ConcatDataset([self.train_ds, AugmentedDATASET])
						
		return DataLoader(NewDATASET, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)
