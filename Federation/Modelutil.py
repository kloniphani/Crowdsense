
from tkinter import N
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
import plotly.offline as pyo
from datetime import datetime
from tqdm import tqdm


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
	def __init__(self, model, name, device, classes, learning_rate=0.001):
		"""
		Args:
			model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
			loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
			optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
			device (gpu, cpu): Main training unit
		"""
		self.model = model
		self.name = name		
		self.train_losses = []
		self.val_losses = []
		self.train_acc = []
		self.val_acc = []
		self.device = device
		self.learning_rate = learning_rate
		self.classes = classes

		self.y_pred = []
		self.y_true = []

		# Loss Function, Optimizer and Scheduler
		self.loss_fn = nn.MSELoss(reduction="mean")
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate)
		
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

	def train(self, train_loader, val_loader, batch_size=64, num_epochs=50):
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
													steps_per_epoch=int(len(train_loader)),
													epochs=num_epochs,
													anneal_strategy='linear')

		for epoch in range(1, num_epochs + 1):
			running_losses = []
			correct_predictions = []
			predictions = []
			records = []

			with tqdm(total=int(len(train_loader))) as pbar:
				for i, data in enumerate(train_loader):
					# Get the input features and target labels, and put them on the GPU
					inputs, labels = data[0].to(self.device), data[1].to(self.device)

					# Normalize the inputs
					inputs_m, inputs_s = inputs.mean(), inputs.std()
					inputs = (inputs - inputs_m) / inputs_s

					inputs, outputs, loss = self.train_step(inputs, labels, batch_size)					
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
				num_batches = len(train_loader)
				avg_loss = np.sum(running_losses) / num_batches
				acc = (sum(correct_predictions)/sum(predictions)) * 100
				self.train_acc.append(acc)
				self.train_losses.append(avg_loss)
			
			batch_val_losses = []	
			batch_val_acc = []	
			validations = []
			with torch.no_grad():
				with tqdm(total=int(len(val_loader))) as pbar:								
					for  i, data in enumerate(val_loader):
						# Get the input features and target labels, and put them on the GPU
						inputs, labels = data[0].to(self.device), data[1].to(self.device)
						self.model.eval()
						yhat = self.model(inputs)
						val_loss = self.criterion(yhat, labels).item()

						batch_val_losses.append(np.sqrt(val_loss)/batch_size)

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

			#self.augmentation(train_loader, records, batch_size)

			print(f"[{epoch}/{num_epochs}] Training loss: {avg_loss:.4f}% \tTraining Accuracy: {acc:.4f}% \tValidation loss: {validation_loss:.4f}% \tValidation Accuracy: {val:.4f}%\n")

		from pathlib import Path
		model_dir = Path.cwd()/'Model'/f'{datetime.now().strftime("%Y-%m-%d")}'
		try:
			model_dir.mkdir(parents=True, exist_ok=False)
		except:
			pass

		model_path = Path.cwd()/'Model'/f'{self.name}.pt'
		excel_path = model_dir/f'{self.name}_{str(self.device)[:3]}_e{num_epochs}_b{batch_size}_{datetime.now().strftime("%H-%M")}.xlsx'
		
		pd.DataFrame({'Training Accuracy': self.train_acc, 
			'Training Losses': self.train_losses, 
			'Validation Accuracy': self.val_acc, 
			'Validation Losses': self.val_losses}).to_excel(str(excel_path))
		
		torch.save(self.model, str(model_path))

	def evaluate(self, test_loader, batch_size=1, n_features=1):
		"""The method evaluate performs the model evaluation

		The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
		and number of features as inputs. Similar to the model validation, it iteratively
		predicts the target values and calculates losses. Then, it returns two lists that
		hold the predictions and the actual values.

		Note:
			This method assumes that the prediction from the previous step is available at
			the time of the prediction, and only does one-step prediction into the future.

		Args:
			test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
			batch_size (int): Batch size for mini-batch training
			n_features (int): Number of feature columns

		Returns:
			list[float]: The values predicted by the model
			list[float]: The actual values in the test set.

		"""
		with torch.no_grad():
			with tqdm(total=int(len(test_loader))) as pbar:
				predictions = []
				values = []
				for x_test, y_test in test_loader:
					x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
					y_test = y_test.to(self.device)
					self.model.eval()
					yhat = self.model(x_test)
					predictions.append(yhat.to(self.device).detach().numpy())
					values.append(y_test.to(self.device).detach().numpy())
					pbar.update(1)

		return predictions, values

	# ----------------------------
	# Inference
	# ----------------------------
	def inference (self, val_dl):
		from tqdm import tqdm

		correct_prediction = 0
		total_prediction = 0
		loss = 0.0

		with tqdm(total=int(len(val_dl))) as pbar:
			# Disable gradient updates
			with torch.no_grad():
				for data in val_dl:
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
		model_dir = Path.cwd()/'Model'/f'{datetime.now().strftime("%Y-%m-%d")}'
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
		model_dir = Path.cwd()/'Model'/f'{datetime.now().strftime("%Y-%m-%d")}'
		try:
			model_dir.mkdir(parents=True, exist_ok=False)
		except:
			pass

		image_path = model_dir/f'Matrix_{self.name}_{str(self.device)[:3]}_{datetime.now().strftime("%H-%M")}.png'
		report_path = model_dir/f'Report_{self.name}_{str(self.device)[:3]}_{datetime.now().strftime("%H-%M")}.json'


		cf_matrix = confusion_matrix(self.y_true, self.y_pred, labels=[value for key, value in self.classes.items()])
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
		matrix = classification_report(self.y_true, self.y_pred, labels=[value for key, value in self.classes.items()])
		print('Classification report : \n', matrix)
		import json
		with open(report_path, "w") as outfile:
			matrix = classification_report(self.y_true, self.y_pred, labels=[value for key, value in self.classes.items()], output_dict=True)
			outfile.write(json.dumps(matrix, indent = 4))

	# ----------------------------
	# Augmrntation
	# ----------------------------
	def augmentation (self, train_dataset, records, batch_size=1):
		from tqdm import tqdm
		from pathlib import Path
		from torch.utils.data import DataLoader
		from SoundDS import AugmentedSoundDS
		
		original_data_dir = Path.cwd()/'Datasets'/'Audio'

		with torch.no_grad():
			with tqdm(total=int(len(records))) as pbar:
				for record in records:
					df = pd.DataFrame(record['data'][1], columns=['classID'])
					df['relative_path'] = record['data'][2]
					df['prediction'] = record['predictions'].cpu().detach().numpy()
					df = df[df['classID']!=df['prediction']]

					AugmentedDATASET = AugmentedSoundDS(df, original_data_dir)
					aug_dl = DataLoader(AugmentedDATASET, batch_size=batch_size, shuffle=True, num_workers=4)

					train_dataset.append(aug_dl)

					pbar.update(1)
				
				print()



