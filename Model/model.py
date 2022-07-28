from tkinter import HIDDEN
import torch
import torch.nn as nn
from torch.nn import init

# ----------------------------
# CNN Audio Classification Model
# ----------------------------
class LeNet (nn.Module):
	# ----------------------------
	# Build the model architecture
	# ----------------------------
	def __init__(self, output_dim):
		super().__init__()
		conv_layers = []

		# First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
		self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
		self.relu1 = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(8)
		init.kaiming_normal_(self.conv1.weight, a=0.1)
		self.conv1.bias.data.zero_()
		conv_layers += [self.conv1, self.relu1, self.bn1]

		# Second Convolution Block
		self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu2 = nn.ReLU()
		self.bn2 = nn.BatchNorm2d(16)
		init.kaiming_normal_(self.conv2.weight, a=0.1)
		self.conv2.bias.data.zero_()
		conv_layers += [self.conv2, self.relu2, self.bn2]

		# Third Convolution Block
		self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu3 = nn.ReLU()
		self.bn3 = nn.BatchNorm2d(32)
		init.kaiming_normal_(self.conv3.weight, a=0.1)
		self.conv3.bias.data.zero_()
		conv_layers += [self.conv3, self.relu3, self.bn3]

		# Fourth Convolution Block
		self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu4 = nn.ReLU()
		self.bn4 = nn.BatchNorm2d(64)
		init.kaiming_normal_(self.conv4.weight, a=0.1)
		self.conv4.bias.data.zero_()
		conv_layers += [self.conv4, self.relu4, self.bn4]

		# Fifth Convolution Block
		self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu5 = nn.ReLU()
		self.bn5 = nn.BatchNorm2d(128)
		init.kaiming_normal_(self.conv5.weight, a=0.1)
		self.conv5.bias.data.zero_()
		conv_layers += [self.conv5, self.relu5, self.bn5]

		# Sixth Convolution Block
		self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu6 = nn.ReLU()
		self.bn6 = nn.BatchNorm2d(256)
		init.kaiming_normal_(self.conv6.weight, a=0.1)
		self.conv6.bias.data.zero_()
		conv_layers += [self.conv6, self.relu6, self.bn6]


		# Linear Classifier
		self.ap = nn.AdaptiveAvgPool2d(output_size=1)
		self.lin = nn.Linear(in_features=256, out_features=output_dim)

		# Wrap the Convolutional Blocks
		self.conv = nn.Sequential(*conv_layers)

	# ----------------------------
	# Forward pass computations
	# ----------------------------
	def forward(self, X):			
		# Run the convolutional blocks
		X = self.conv(X)

		# Adaptive pool and flatten for input to linear layer
		X = self.ap(X)
		X = X.view(X.shape[0], -1)

		# Linear layer
		X = self.lin(X)

		# Final output
		return X

# ----------------------------
# AE Audio Classification Model
# ----------------------------
class AE(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.encoder_hidden_layer = nn.Linear(
			in_features=kwargs["input_size"], out_features=kwargs['output_size']
		)
		self.encoder_output_layer = nn.Linear(
			in_features=kwargs["output_size"], out_features=kwargs['output_size']
		)
		self.decoder_hidden_layer = nn.Linear(
			in_features=kwargs['output_size'], out_features=kwargs['output_size']
		)
		self.decoder_output_layer = nn.Linear(
			in_features=kwargs['output_size'], out_features=kwargs["input_size"]
		)

	def forward(self, features):
		activation = self.encoder_hidden_layer(features)
		activation = torch.relu(activation)
		code = self.encoder_output_layer(activation)
		code = torch.relu(code)
		activation = self.decoder_hidden_layer(code)
		activation = torch.relu(activation)
		activation = self.decoder_output_layer(activation)
		reconstructed = torch.relu(activation)
		return reconstructed

# ----------------------------
# SVM Audio Classification Model
# ----------------------------
class LinearSVM(nn.modules.Module):    
	"""Support Vector Machine"""

	def __init__(self, x_dim, y_dim):
		super(LinearSVM, self).__init__()
		self.w = nn.Parameter(torch.randn(x_dim, y_dim), requires_grad=True)
		self.b = nn.Parameter(torch.randn(1), requires_grad=True)

	def forward(self, x):
		h = x.matmul(self.w.t()) + self.b
		return h


# ----------------------------
# RNN Audio Classification Model
# ----------------------------
class BasicRNN (nn.Module):
	# ----------------------------
	# Build the model architecture
	# ----------------------------
	def __init__(self, n_channels, output_dim):
		pass

	def __init__(self, input_size, output_size, hidden_dim, n_layers):
		super(BasicRNN, self).__init__()

		# Defining some parameters
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers

		#Defining the layers
		# RNN Layer
		self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
		# Fully connected layer
		self.fc = nn.Linear(hidden_dim, output_size)
    
	def forward(self, x):
		batch_size = x.size(0)

		# Initializing hidden state for first input using method defined below
		hidden = self.init_hidden(batch_size)

		# Passing in the input and hidden state into the model and obtaining outputs
		out, hidden = self.rnn(x, hidden)

		# Reshaping the outputs such that it can be fit into the fully connected layer
		out = out.contiguous().view(-1, self.hidden_dim)
		out = self.fc(out)

		return out, hidden
    
	def init_hidden(self, batch_size):
		# This method generates the first hidden state of zeros which we'll use in the forward pass
		# We'll send the tensor holding the hidden state to the device we specified earlier as well
		hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
		return hidden

# ----------------------------
# RNN Audio Classification Model
# ----------------------------
class RNNClassifier (nn.Module):
	def __init__(self, batch_size, n_steps, n_channels, n_inputs, n_neurons, n_outputs, device):
		super(RNNClassifier, self).__init__()

		self.n_neurons = n_neurons
		self.batch_size = batch_size
		self.n_steps = n_steps
		self.n_channels = n_channels
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.device = device

		self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons) 

		self.FC = nn.Linear(self.n_neurons, self.n_outputs)
        
	def init_hidden(self,):
		# (num_layers, batch_size, n_neurons)
		return (torch.zeros(1, self.batch_size, self.n_neurons, device=self.device))
        
	def forward(self, X):
		# transforms X to dimensions: n_steps X batch_size X n_inputs
		X = X.permute(1, 0, 2) 
		
		self.batch_size = X.size(1)
		self.hidden = self.init_hidden()
		
		lstm_out, self.hidden = self.basic_rnn(X, self.hidden)      
		out = self.FC(self.hidden)
		
		return out.view(-1, self.n_outputs) # batch_size X n_output

# ----------------------------
# LSTM Audio Classification Model
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

# ----------------------------
# GRU Audio Classification Model
# ----------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

# ----------------------------
# RNN Audio Classification Model
# ----------------------------
class RNNModel (nn.Module):
	def __init__(self, n_steps, n_channels, input_dim, hidden_dim, layer_dim, output_dim, device):
		super(RNNModel, self).__init__()

		self.hidden_dim = hidden_dim
		self.n_steps = n_steps
		self.n_channels = n_channels
		self.input_dim = input_dim
		self.layer_dim = layer_dim
		self.output_dim = output_dim
		self.device = device

		self.basic_rnn = nn.RNN(self.input_dim, self.hidden_dim, layer_dim, batch_first=True) 
		self.gru = nn.GRU(self.input_dim, self.hidden_dim, layer_dim, batch_first=True) 
		self.lsmt = nn.LSTM(self.input_dim, self.hidden_dim, layer_dim, batch_first=True) 

		self.i2hidden = nn.Linear(self.layer_dim, self.input_dim)
		self.i2output = nn.Linear(self.layer_dim, self.output_dim)
		self.fc = nn.Linear(hidden_dim, output_dim)

		self.softmax = nn.LogSoftmax(dim=self.n_channels)
        
	def forward(self, X):
		hidden = (torch.zeros(self.layer_dim, X.size(1), self.hidden_dim, device=self.device).requires_grad_())

		print(hidden.shape)
		print(X.shape)

		combined, hn = self.basic_rnn(X, hidden.detach())
		i2hidden = self.i2hidden(combined)
		i2output = self.i2output(combined)
		output = self.softmax(i2output[:, -1, :])
		
		return output, i2hidden


# ----------------------------
# CNN Audio Classification Model
# ----------------------------
class CNNModel (nn.Module):
	# ----------------------------
	# Build the model architecture
	# ----------------------------
	def __init__(self, n_channels, output_dim):
		super().__init__()
		conv_layers = []

		# First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
		self.conv1 = nn.Conv2d(n_channels, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
		self.relu1 = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(8)
		init.kaiming_normal_(self.conv1.weight, a=0.1)
		self.conv1.bias.data.zero_()
		conv_layers += [self.conv1, self.relu1, self.bn1]

		# Second Convolution Block
		self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu2 = nn.ReLU()
		self.bn2 = nn.BatchNorm2d(16)
		init.kaiming_normal_(self.conv2.weight, a=0.1)
		self.conv2.bias.data.zero_()
		conv_layers += [self.conv2, self.relu2, self.bn2]

		# Third Convolution Block
		self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu3 = nn.ReLU()
		self.bn3 = nn.BatchNorm2d(32)
		init.kaiming_normal_(self.conv3.weight, a=0.1)
		self.conv3.bias.data.zero_()
		conv_layers += [self.conv3, self.relu3, self.bn3]

		# Fourth Convolution Block
		self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu4 = nn.ReLU()
		self.bn4 = nn.BatchNorm2d(64)
		init.kaiming_normal_(self.conv4.weight, a=0.1)
		self.conv4.bias.data.zero_()
		conv_layers += [self.conv4, self.relu4, self.bn4]

		"""
		# Fifth Convolution Block
		self.conv5 = nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu5 = nn.ReLU()
		self.bn5 = nn.BatchNorm2d(2048)
		init.kaiming_normal_(self.conv5.weight, a=0.1)
		self.conv5.bias.data.zero_()
		conv_layers += [self.conv5, self.relu5, self.bn5]

		# Sixth Convolution Block
		self.conv6 = nn.Conv2d(2048, 4096, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu6 = nn.ReLU()
		self.bn6 = nn.BatchNorm2d(4096)
		init.kaiming_normal_(self.conv6.weight, a=0.1)
		self.conv6.bias.data.zero_()
		conv_layers += [self.conv6, self.relu6, self.bn6]
		"""

		# Linear Classifier
		self.ap = nn.AdaptiveAvgPool2d(output_size=1)
		self.lin = nn.Linear(in_features=64, out_features=output_dim)

		# Wrap the Convolutional Blocks
		self.conv = nn.Sequential(*conv_layers)

	# ----------------------------
	# Forward pass computations
	# ----------------------------
	def forward(self, X):
		# Adding a dimension if the shape lenght is 3
		if len(X.shape) == 3: 
			X = X.unsqueeze(1)
			
		# Run the convolutional blocks
		X = self.conv(X)

		# Adaptive pool and flatten for input to linear layer
		X = self.ap(X)
		X = X.view(X.shape[0], -1)

		# Linear layer
		X = self.lin(X)

		# Final output
		return X
