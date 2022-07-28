import json
import torch, torchaudio
import traceback, socket
import numpy as np

from Cloud import Google
from pathlib import Path
from AudioUtil import *
from Model import *


# ----------------------------
# Uploading data to the cloud
# ----------------------------
database = Google().database

recording_path = Path.cwd()/'Recordings'
data_dir = Path.cwd()/'Datasets'/'Audio'
model_path = Path.cwd()/'Models/LeNet.pt'


def publish():
    from time import time, sleep
    from datetime import datetime
    from uuid import uuid1 as get_mac
    from pathlib import Path
    from alive_progress import alive_bar

    import sounddevice as sd
    from scipy.io.wavfile import write

    import os

    mac = str(get_mac())
    name = socket.gethostname()
    locations = [{'lat':-33.9335203, 'lng':18.6257693}, {'lat':33.8625047, 'lng':18.4331918}, 
        {'lat':-33.8625047, 'lng':18.4331918}, {'lat':-33.8625047, 'lng':18.4331918}, 
        {'lat':-33.8625047, 'lng':18.4331918}, {'lat':-33.8890138, 'lng':18.4829736}]

    geo_location = np.random.choice(locations)

    database.child('nodes').child(name).push({"device": mac, "location": geo_location})

    with open("class.json") as f:
        CLASSES = json.load(f)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        MODEL = torch.load(model_path, map_location=device)
        MODEL.to(device)
        MODEL.eval()

        fs = 44100  # Sample rate
        seconds = 4  # Duration of recording
        
        
    
        while True:
            try:
                _time = time()

                myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                sd.wait()  # Wait until recording is finished
                write(str(recording_path/f"record_{int(_time)}.ogg"), fs, myrecording)  # Save as WAV file 
                
                aud = AudioUtil.open(str(recording_path/f"record_{int(_time)}.ogg"))
                sgram = AudioUtil.spectro_gram(aud).to(device)

                # Get predictions
                prediction = MODEL(sgram.unsqueeze(0))
                probs = torch.nn.functional.softmax(prediction, dim=1)
                conf, classes = torch.max(probs, 1)

                audio = [k for k, v in CLASSES.items() if v == int(classes.item())][0]
                path = data_dir/f"{audio}"

                samples_sf = myrecording[::, 0].tolist()
                source = [20 * np.log10(abs(i)/1) for i in samples_sf if math.isinf(i) is False and i > 0]
                min_d = np.amin(source)
                max_d = np.amax(source)
                ave_d = np.average(source)

                #predicted_class = np.argmax(prediction.cpu().detach().numpy())
                data = {
                    "name": name,
                    "device": mac,
                    "timestamp": _time,
                    "class": int(classes.item() + 1),
                    "audio": audio,
                    "confidence": float(conf.item()),
                    "location": geo_location,
                    "decibels": {
                        "min": min_d,
                        "max": max_d,
                        "ave": ave_d
                    } 
                }

                database.child('data').child(name).push(data)

                try:                    
                    os.makedirs(path, exist_ok=False)
                except Exception:
                    pass

                #os.remove(audio_file)
                filename = path/f"record_{int(_time)}.ogg"
                write(str(filename), fs, myrecording)  # Save as WAV file 
                print("{0}\t[{1}]".format(datetime.fromtimestamp(_time).strftime('%Y-%m-%d %H:%M:%S'), audio))
            except Exception:
                traceback.print_exc()
                break




if __name__ == "__main__":
    print("\n----------------------------------------------------------------------------------")
    print("FEDRATED INTERNET OF THINGS\nAudio Deep Learning to Aid People Classify and Identify Noise Pollution in Cape Town")
    print("----------------------------------------------------------------------------------")

    publish()
 
