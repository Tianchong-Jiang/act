# Base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Update PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the 'aloha' conda environment with Python 3.8.10
RUN conda create -n aloha python=3.8.10 -y

# Activate the conda environment and install Python packages
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate aloha && \
    pip install \
        torch \
        torchvision \
        pyquaternion \
        pyyaml \
        rospkg \
        pexpect \
        mujoco==2.3.7 \
        dm_control==1.0.14 \
        opencv-python \
        matplotlib \
        einops \
        packaging \
        h5py \
        ipython"

# Copy and install the 'detr' package from 'act/detr'
COPY act/detr /opt/act/detr
RUN /bin/bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate aloha && \
    pip install -e /opt/act/detr"

ENTRYPOINT ["/bin/bash", "-c", "source $CONDA_DIR/etc/profile.d/conda.sh && conda activate aloha && exec bash"]
