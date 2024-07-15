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

# ============ CPU Image =============
FROM base_image AS cpu_image
COPY requirements.txt ${WORKDIR}/requirements.txt
RUN pip install -r ${WORKDIR}/requirements.txt \
    && pip cache purge

# ============ GPU Image =============
FROM cpu_image AS gpu_image
COPY requirements-gpu.txt ${WORKDIR}/
RUN pip install -r ${WORKDIR}/requirements-gpu.txt \
    && pip cache purge
