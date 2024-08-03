FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS base_image
ENV WORKDIR=/workspace/benchmark/

RUN apt-get update \
    &&  apt-get install -y \
    python-is-python3 \
    pkg-config \
    python3-opencv \
    && apt-get autoremove \
    && apt-get clean

WORKDIR $WORKDIR

COPY requirements.txt ${WORKDIR}/requirements.txt
RUN pip install -r ${WORKDIR}/requirements.txt \
    && pip cache purge
RUN opentelemetry-bootstrap -a install
