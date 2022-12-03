FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev \
    libgtk3.0 libsm6 cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev \
    libboost-python-dev libtinyxml-dev bash \
    wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev openssh-server \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3 zlib1g-dev unrar \
    libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg

RUN pip install opencv-python future pathlib 
RUN pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax dm-haiku rlax
RUN pip install tensorflow tbp-nightly
RUN pip install pandas easydict
RUN pip install pandas
RUN pip install seaborn
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && tar -xvf mujoco.tar.gz \
    && mkdir ~/.mujoco && cp -r mujoco210 ~/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
RUN pip install  gym-super-mario-bros==7.3.2 \
    opencv-python future pyglet minigrid -U 'mujoco-py<2.2,>=2.1' gym[atari,box2d] procgen \
    pathlib ray[default] pygame mujoco
RUN wget http://www.atarimania.com/roms/Roms.rar -O roms.rar && unrar x roms.rar \
    && ale-import-roms ROMS/ > ale.out
RUN python -c 'import mujoco_py; import gym'
RUN echo -e "\033[?25h"
RUN rm -r /var/lib/apt/lists/*

WORKDIR /entry

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user