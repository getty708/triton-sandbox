FROM nvcr.io/nvidia/deepstream:7.0-samples-multiarch AS base_image
ENV WORKDIR=/workspace/benchmark
ENV PATH=/usr/src/tensorrt/bin:$PATH
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

RUN apt-get update \
    &&  apt-get install -y \
    python-is-python3 \
    pkg-config \
    python3-opencv \
    python3-pip \
    graphviz \
    && apt-get autoremove \
    && apt-get clean

# ============ Setup pyds =============
ENV DEEPSTREAM_PYTHON_BINDING_SOURCE_DIR=/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps
WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps

# Ref: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test3
RUN apt update && apt install python3-gi python3-dev python3-gst-1.0 -y && pip3 install pathlib

# Install python bindings
# Ref: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings
RUN apt install -y \
    python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson \
    python3 python3-pip python3.10-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev
RUN apt-get install -y apt-transport-https ca-certificates  && update-ca-certificates
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
RUN cd 3rdparty/gstreamer/subprojects/gst-python/
RUN meson setup build \
    && cd build \
    && ninja \
    && ninja install
WORKDIR ${DEEPSTREAM_PYTHON_BINDING_SOURCE_DIR}/bindings
RUN mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc)
RUN pip3 install ./pyds-1.1.11-py3-none*.whl
RUN pip install cuda-python

# msgcnv
RUN apt-get update &&  apt-get install libglib2.0 libglib2.0-dev libssl-dev \
    && apt-get install libjansson4  libjansson-dev \
    && apt-get install librdkafka1



# ============ GPU Image =============
FROM base_image AS gpu_image
WORKDIR $WORKDIR

# COPY requirements.txt ${WORKDIR}/requirements.txt
# RUN pip3 install -r ${WORKDIR}/requirements.txt \
#     && pip3 cache purge

# RUN apt install -y git python3 python3-pip cmake g++ build-essential     libglib2.0-dev libglib2.0-dev-bin python-gi-dev libtool m4 autoconf automake
# COPY requirements-gpu.txt ${WORKDIR}/
# RUN pip3 install -r ${WORKDIR}/requirements-gpu.txt \
#     && pip3 cache purge

