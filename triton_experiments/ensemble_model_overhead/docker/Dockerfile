FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS base_image
ARG TARGET_DEVICE
ENV WORKDIR=/workspace/triton-sandbox/triton_tutorials

RUN apt update \
    &&  apt install -y python-is-python3 pkg-config\
    && apt autoremove \
    && apt clean

WORKDIR $WORKDIR

# ============ GPU Image =============
FROM base_image AS gpu_image
COPY requirements*.txt ${WORKDIR}/
RUN pip install -r ${WORKDIR}/requirements.txt \
    && pip install -r ${WORKDIR}/requirements-gpu.txt \
    && pip cache purge

# ============ CPU Image =============
FROM base_image AS cpu_image
COPY requirements.txt ${WORKDIR}/requirements.txt
RUN pip install -r ${WORKDIR}/requirements.txt \
    && pip cache purge
