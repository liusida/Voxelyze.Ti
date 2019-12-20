FROM nvidia/cuda:10.1-devel-ubuntu16.04
RUN apt update && apt install -y cmake nvidia-cuda-toolkit
RUN echo "PATH=/usr/local/cuda/bin:$PATH" >> /root/.bashrc
