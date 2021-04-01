FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
