FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
