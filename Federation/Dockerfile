# syntax=docker/dockerfile:1

FROM ubuntu:latest 
RUN apt-get update && apt-get install -y software-properties-common gcc && \
	add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.9 python3-distutils python3-pip python3-apt python3-tk
RUN apt-get install portaudio19-dev python3-pyaudio -y

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 5557

ENTRYPOINT [ "python3",  "Predict.py"] 

