"""

DEPRECATED FILE ???

"""

FROM ultralytics/ultralytics:latest

LABEL Mohamed Benkedadra <mohamed.benkedadra@umons.ac.be>

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /workspace
ADD . /workspace/
WORKDIR /workspace/yolov8


RUN pip install -U pip && pip install -r /workspace/yolov8/ultralytics/requirements.txt
RUN pip install -U pip && pip install -r /workspace/requirements.txt

ARG WANDB_API_KEY

RUN wandb login $WANDB_API_KEY