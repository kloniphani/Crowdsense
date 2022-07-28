import math
import random
import os
import requests
import torch
import torchaudio

import matplotlib.pyplot as plt
import torchaudio.functional as F
import torchaudio.transforms as T

from torchaudio import transforms
from IPython.display import Audio, display
from pathlib import Path
from email.mime import audio


class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def effect_filtering(aud, lowpass='300', speed='0.8'):
        sig, sr = aud

        # Define effects
        effects = [
            ["lowpass", "-1", lowpass],  # apply single-pole lowpass filter
            ["speed", speed],  # reduce the speed
            # This only changes sample rate, so it is necessary to
            # add `rate` effect with original sample rate after this.
            ["rate", f"{sr}"],
            ["reverb", "-w"],  # Reverbration gives some dramatic feeling
            ['gain', '-n'],  # normalises to 0dB
        ]

        # Apply effects
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            sig, sr, effects)
        return (waveform, sample_rate)

    @staticmethod
    def background_noise(aud, snr_db=3):
        import glob
        import numpy as np

        sig, sr = aud

        files_location = ".\Datasets\Audio\\00 - Quiet\*"
        BACKGROUND_NOISES = [f for f in glob.glob(files_location)]
        effects = [
            ["remix", "1"],
            ["lowpass", f"{sr // 2}"],
            ["rate", f'{sr}'],
        ]

        noise, _ = torchaudio.sox_effects.apply_effects_file(
            np.random.choice(BACKGROUND_NOISES), effects=effects)
        noise = noise[:, :sig.shape[1]]

        sig_power = sig.norm(p=2)
        noise_power = noise.norm(p=2)

        snr = math.exp(snr_db / 10)
        scale = snr * noise_power / sig_power
        noisy_sig = (scale * sig + noise) / 2

        return (noisy_sig, sr)

    @staticmethod
    def codecs(aud, config={"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}):
        sig, sr = aud
        return (F.apply_codec(sig, sr, **config), sr)

    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(
                freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(
                time_mask_param)(aug_spec, mask_value)

        return aug_spec

    # ----------------------------
    # Capture and Process Data from Mic
    # ----------------------------
    @staticmethod
    def generator():
        while True:
            yield

    @staticmethod
    def capture_audio(path=Path.cwd()/'Recordings', min_db=20, duration=4, sample_rate=48000, loop=True):
        import sounddevice as sd
        import soundfile as sf
        import numpy as np

        import time
        import pyaudio

        from pathlib import Path
        from alive_progress import alive_bar

        sd.default.samplerate = sample_rate
        sd.default.channels = 2

        maxValue = 2**16
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                        input=True, frames_per_buffer=1024)

        if loop is True:
            with alive_bar() as bar:
                while True:
                    data = np.fromstring(stream.read(1024), dtype=np.int16)
                    dataL = data[0::2]
                    dataR = data[1::2]

                    peakL = (np.abs(np.max(dataL) - np.min(dataL)) /
                             maxValue) * 100
                    peakR = (np.abs(np.max(dataR) - np.min(dataR)) /
                             maxValue) * 100

                    if peakL >= min_db or peakR >= min_db:
                        audio_data = sd.rec(int(sample_rate * duration))
                        audio_name = Path(
                            path)/'record_{:.0f}.wav'.format(time.time())
                        sf.write(audio_name, audio_data, sample_rate)

        audio_data = sd.rec(int(sample_rate * duration))
        audio_name = Path(path)/'record_{:.0f}.wav'.format(time.time())
        sf.write(audio_name, audio_data, sample_rate)

    @staticmethod
    def plot_waveform(audio, title="Waveform", xlim=None, ylim=None, file_name=None, show=False):
        waveform, sample_rate = audio
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)

        if file_name is not None:
            file_name = file_name.split('\\')
            plt.savefig(
                f'.\Results\{file_name[-1].split(".")[0]} - {title} - Waveform.png', format="png")
        if show == True:
            plt.show(block=True)
        plt.close()

    @staticmethod
    def plot_specgram(audio, title="Spectrogram", xlim=None, file_name=None, show=False):
        waveform, sample_rate = audio
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)

        if file_name is not None:
            file_name = file_name.split('\\')
            plt.savefig(
                f'.\Results\{file_name[-1].split(".")[0]} - {title} - Spectrogram.png', format="png")
        if show == True:
            plt.show(block=True)
        plt.close()

    @staticmethod
    def play_audio(waveform, sample_rate):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        if num_channels == 1:
            display(Audio(waveform[0], rate=sample_rate))
        elif num_channels == 2:
            display(Audio((waveform[0], waveform[1]), rate=sample_rate))
        else:
            raise ValueError(
                "Waveform with more than 2 channels are not supported.")

    @staticmethod
    def plot_mel_specgram(audio, title="Mel Spectrogram", xlim=None, file_name=None, show=False):
        waveform, sample_rate = audio
        waveform = waveform.numpy()

        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or 'Filter bank')
        axs.imshow(waveform, aspect='auto')
        axs.set_ylabel('frequency bin')
        axs.set_xlabel('mel bin')
        fig.suptitle(title)


        if file_name is not None:
            file_name = file_name.split('/')
            plt.savefig(f'.\Results\{file_name[-1].split(".")[0]} - {title} - Spectrogram.png', format="png")
        if show == True:
            plt.show(block=True)
        plt.close()


if __name__ == "__main__":
    import glob
    import numpy as np

    duration = 5000
    sr = 44100
    channel = 2
    shift_pct = 0.4

    CurrentPath = Path().absolute()
    files_location = CurrentPath/"Datasets"/"Audio"/"20 - Gun shot"
    AUDIO_FILES = [f for f in glob.glob(
        str(files_location) + "/*")]

    Audio_Utility = AudioUtil()
    audio_file = np.random.choice(AUDIO_FILES)
    audio_file = str(CurrentPath/"Datasets"/"Audio"/"20 - Gun shot"/"518-4-0-0.wav")

    print("01 - Opening audio file")
    Audio = Audio_Utility.open(str(audio_file))
    Audio_Utility.plot_waveform(audio=Audio, title="Original", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=Audio, title="Original", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("02 - Resampling audio tensor")
    reaud = AudioUtil.resample(Audio, sr)
    Audio_Utility.plot_waveform(audio=reaud, title="Resampled", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=reaud, title="Resampled", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("03 - Rechanelling audio tensor")
    rechan = AudioUtil.rechannel(reaud, channel)
    Audio_Utility.plot_waveform(audio=rechan, title="Rechanneled", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=rechan, title="Rechanneled", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("04 - Trimming audio tensor")
    dur_aud = AudioUtil.pad_trunc(rechan, duration)
    Audio_Utility.plot_waveform(audio=dur_aud, title="Resized", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=dur_aud, title="Resized", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("05 - Shifting audio tensor")
    shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
    Audio_Utility.plot_waveform(audio=shift_aud, title="Time Shifted", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=shift_aud, title="Time Shifted", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("06 - Applying effect on audio tensor")
    effects_aud = AudioUtil.effect_filtering(
        shift_aud, lowpass='400', speed='1')
    Audio_Utility.plot_waveform(audio=effects_aud, title="Effect Applied", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=effects_aud, title="Effect Applied", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("07 - Applying background noise on audio tensor")
    background_aud = AudioUtil.background_noise(effects_aud, snr_db=10)
    Audio_Utility.plot_waveform(audio=shift_aud, title="With Background Noise", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=shift_aud, title="With Background Noise", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("08 - Applying codecs on audio tensor")
    codec_aud = AudioUtil.codecs(background_aud)
    Audio_Utility.plot_waveform(audio=codec_aud, title="Codecs Applied", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=codec_aud, title="Codecs Applied", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("09 - Converting audio tensor to Mel Spectrogram")
    sgram = AudioUtil.spectro_gram(
        shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    Audio_Utility.plot_waveform(audio=sgram, title="MelSpectrogram", xlim=(-.1, 3.2), file_name=audio_file, show=False)
    Audio_Utility.plot_specgram(audio=sgram, title="MelSpectrogram", xlim=(-.1, 3.2), file_name=audio_file, show=False)

    print("10 - Masking Mel Spectrogram")
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    Audio_Utility.plot_waveform(audio=aug_sgram, title="MelSpectrogram Masked", xlim=(-.1, 3.2), file_name=audio_file, show=True)
    Audio_Utility.plot_specgram(audio=aug_sgram, title="MelSpectrogram Masked", xlim=(-.1, 3.2), file_name=audio_file, show=True)
