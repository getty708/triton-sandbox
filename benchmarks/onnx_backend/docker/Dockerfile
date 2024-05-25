FROM nvcr.io/nvidia/tritonserver:24.01-py3 AS base_image
ENV WORKDIR=/workspace/benchmark/onnx_backend/

RUN apt update \
    &&  apt install -y python-is-python3 pkg-config\
    && apt autoremove \
    && apt clean

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
