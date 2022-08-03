from torch.utils.data import Dataset
from torchvision import transforms

from audio import *

# ----------------------------
# Sound Dataset
# ----------------------------
class dataset(Dataset):
	def __init__(self, df, data_path, augment=False):
		self.df = df
		self.df.reset_index(drop=True, inplace=True)
		self.data_path = str(data_path)
		self.augment = augment
		self.duration = 4000
		self.sr = 44100
		self.channel = 2
		self.shift_pct = 0.4
            
	# ----------------------------
	# Number of items in dataset
	# ----------------------------
	def __len__(self):
		return len(self.df)    
    
	# ----------------------------
	# Get i'th item in dataset
	# ----------------------------
	def __getitem__(self, idx):
		# Absolute file path of the audio file - concatenate the audio directory with
		# the relative path
		audio_file = self.data_path + self.df.loc[idx, 'relative_path']
		# Get the Class ID
		class_id = self.df.loc[idx, 'classID']

		aud = augmentation.open(audio_file)
		# Some sounds have a higher sample rate, or fewer channels compared to the
		# majority. So make all sounds have the same number of channels and same 
		# sample rate. Unless the sample rate is the same, the pad_trunc will still
		# result in arrays of different lengths, even though the sound duration is
		# the same.
		reaud = augmentation.resample(aud, self.sr)
		rechan = augmentation.rechannel(reaud, self.channel)

		dur_aud = augmentation.pad_trunc(rechan, self.duration)

		if self.augment == False:
			sgram = augmentation.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
		else:
			shift_aud = augmentation.time_shift(dur_aud, self.shift_pct)
			sgram = augmentation.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
			sgram = augmentation.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

		return sgram, class_id, self.df.loc[idx, 'relative_path']

	# ----------------------------
	# Append item to the dataset
	# ----------------------------
	def append(self, item):
		self.df.append(item)
