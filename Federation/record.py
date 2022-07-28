from AudioUtil import *

def record_audio():
    AudioUtil.capture_audio(loop=True)


if __name__ == "__main__":
    print("\n----------------------------------------------------------------------------------")
    print("FEDRATED INTERNET OF THINGS\nAudio Deep Learning to Aid People Classify and Identify Noise Pollution in Cape Town")
    print("----------------------------------------------------------------------------------")

    record_audio()