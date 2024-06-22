# The dockerfile is built to produce image of the pytorch implementation of Unsupervised Domain Adaptation by Backpropagation

FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN pip install pillow \
 && pip install torchvision \
 && pip install wandb \
 && mkdir /DANN

#VOLUME ["/DANN/dataset"]

EXPOSE 22

#COPY ./data_loader.py ./functions.py ./main.py ./model.py ./huseyin_functions.py ./README.md ./test.py /DANN/ 
#COPY /home/huseyin/fungtion/dannpy_yeniden/DANN_py3/ /DANN/
COPY . /DANN/

WORKDIR /DANN