FROM tensorflow/tensorflow:latest

RUN apt-get update && \
    apt-get upgrade -y

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN apt install sudo
RUN sudo apt-get install curl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin/kubectl
RUN kubectl version --client

RUN python -c "import keras, yfinance, pandas, numpy, sklearn, tensorflow, jinja2"

