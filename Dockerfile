FROM ultralytics/yolov5:v6.2

LABEL Mohamed Benkedadra <mohamed.benkedadra@umons.ac.be>

RUN pip install -U pip && pip install -r requirements.txt
RUN pip install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

ENV PYTHONUNBUFFERED=1

ARG WANDB_API_KEY

RUN wandb login $WANDB_API_KEY

RUN mkdir -p /workspace

ADD . /workspace/

WORKDIR /workspace
