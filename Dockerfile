FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
