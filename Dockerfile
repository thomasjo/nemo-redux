FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
