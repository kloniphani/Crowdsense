# syntax=docker/dockerfile:1

FROM ubuntu:latest
RUN apt-get update && apt-get install -y software-properties-common gcc && \
	add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.9 python3-distutils python3-pip python3-apt
RUN apt-get install -y portaudio19-dev python3-pyaudio 

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3",  "WebServer.py"] 
EXPOSE 8080

