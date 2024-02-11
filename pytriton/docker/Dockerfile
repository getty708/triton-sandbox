FROM nvcr.io/nvidia/tritonserver:24.01-py3
ENV WORKDIR=/workspace/triton-sandbox

RUN apt update &&  apt install -y python-is-python3

WORKDIR $WORKDIR
COPY requirements.txt ${WORKDIR}/requirements.txt
RUN pip install -r ${WORKDIR}/requirements.txt
