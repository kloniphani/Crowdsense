from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# ----------------------------
# Image Dataset
# ----------------------------
class ImageDS(Dataset):
	def __init__(self, df, data_path):
		self.df = df
		self.data_path = str(data_path)
            
	# ----------------------------
	# Number of items in dataset
	# ----------------------------
	def __len__(self):
		return len(self.df)    
    
	# ----------------------------
	# Get i'th item in dataset
	# ----------------------------
	def __getitem__(self, idx):
		# Absolute file path of the iamge file - concatenate the audio directory with
		# the relative path
		image_file = self.data_path + self.df.loc[idx, 'relative_path']
		# Get the Class ID
		class_id = self.df.loc[idx, 'classID']

		image_pil = Image.open(image_file)
		transform_to_tensor = transforms.ToTensor()
		image_tensor = transform_to_tensor(image_pil)
		#image_tensor_resized = F.interpolate(image_tensor, size=(172))
		#image_tensor_resized = image_tensor.resize_(1, 16, 86)
	

		return image_tensor, class_id, self.df.loc[idx, 'relative_path']