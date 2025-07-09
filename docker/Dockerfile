# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
FROM nvcr.io/nvidia/cuda:12.5.0-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES all
RUN apt update
RUN apt install -y vim libglib2.0-0 libsm6 libxrender1 libxext6 wget unzip
RUN apt install -y build-essential git cmake python3-pip libgl1-mesa-dev git-lfs
RUN apt install -y libxrandr-dev libxinerama-dev libxi-dev  libbullet-dev libxcursor-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev  \
        libgl1-mesa-dev  \
        libegl1-mesa-dev \
        libgles2-mesa-dev 

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux \
        bash-completion \
        command-not-found \
        software-properties-common \
        curl \
        coreutils \
        pkg-config \
        xdg-user-dirs \
        libgl1-mesa-dev \
        freeglut3-dev \
        mesa-utils \
        vulkan-tools \
        libvulkan-dev \
        libglfw3-dev \
        python3.10-venv \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install the required Python packages
RUN python3.10 -m pip install torch>=1.10.1 \
    numpy>=1.22.4 \
    flask>=2.3.2 \
    seaborn>=0.12.2 \
    open3d>=0.17.0 \
    transformers==4.26.0 \
    salesforce-lavis>=1.0.2 \
    pre-commit>=3.1.1 \
    pytest>=7.2.1 \
    pytest-cov>=4.0.0 \
    bosdyn-client>=3.3.2 \
    bosdyn-api>=3.3.2 \
    six>=1.16.0


RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN apt update && apt install -y python3-tk
RUN pip install pip==24.0
RUN pip install easydict transformers tensorboardX jsonlines dtw fastdtw torchvision openai-clip gdown gym==0.21.0 
RUN pip install xformers==0.0.24

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt update
RUN apt install ros-noetic-desktop-full -y
RUN apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y
RUN apt install -y python3-rosdep libgflags-dev python3-catkin-tools python3-pip tmux git vim python3-vcstool
# Install misc.
RUN pip install gdown networkx open3d numpy==1.23.0 tmuxp 
RUN apt-get install -qy g++ libeigen3-dev
RUN apt-get install -y libspdlog-dev
RUN apt install -y libsdl-image1.2-dev ros-noetic-move-base-msgs

RUN apt install libprotobuf-dev protobuf-compiler -y 
# set up ssh
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts 
WORKDIR /root
RUN mkdir -p /root/catkin_ws/src
RUN . /opt/ros/noetic/setup.sh && mkdir -p /root/catkin_ws/src && cd /root/catkin_ws && catkin init

RUN apt update
RUN apt install -y libgoogle-glog-dev ros-noetic-nmea-msgs ros-noetic-rosserial-server ros-noetic-teleop-twist-joy ros-noetic-topic-tools python3-scipy ros-noetic-rosbridge-server ros-noetic-lms1xx  ros-noetic-realsense2-camera ros-noetic-ddynamic-reconfigure ros-noetic-rosserial-python ros-noetic-nmea-navsat-driver

## Use ccache build 
RUN apt install -y ccache ninja-build
ENV CCACHE_DIR=/ccache
ENV PATH="/usr/lib/ccache:$PATH"
# RUN --mount=type=cache,target=/ccache/,mode=0755 ccache -M 10G && cd /root/catkin_ws && . /opt/ros/noetic/setup.sh && catkin build && ccache -s

RUN --mount=type=ssh cd /root/catkin_ws/src && git clone --recursive git@github.com:tyuezhan/SplaTAM.git
RUN --mount=type=ssh cd /root/catkin_ws/src && git clone --recursive git@github.com:tyuezhan/mpl_py.git

# zed
RUN apt update && apt install -y libnvidia-encode-545 libnvidia-decode-545 zstd
RUN wget https://download.stereolabs.com/zedsdk/4.2/cu12/ubuntu20 && chmod +x ubuntu20 && ./ubuntu20 -- silent
RUN cd /root/catkin_ws/src/ && git clone --recursive https://github.com/KumarRobotics/zed-ros-wrapper.git
RUN cd /root/catkin_ws && . /opt/ros/noetic/setup.sh && rosdep init && rosdep update && rosdep install --from-paths . --ignore-src -r -y
RUN --mount=type=cache,target=/ccache/,mode=0755 ccache -M 10G && cd /root/catkin_ws && . /opt/ros/noetic/setup.sh && catkin build && ccache -s
RUN --mount=type=ssh cd /root/catkin_ws/src && vcs import . < SplaTAM/extras/robot_extras.rosinstall --recursive
RUN --mount=type=cache,target=/ccache/,mode=0755 ccache -M 10G && cd /root/catkin_ws && . /opt/ros/noetic/setup.sh && catkin build && ccache -s

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update \
      && apt-get install --no-install-recommends -qy curl apt-transport-https gnupg \
      sudo ca-certificates git software-properties-common \
      build-essential \
      libgl1-mesa-dev libjpeg-dev libglm-dev libgl1-mesa-glx \
      libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev \
      python3.10-dev python3-pip python-is-python3 \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean all


RUN pip install git+https://github.com/naokiyokoyama/frontier_exploration.git \
                git+https://github.com/ChaoningZhang/MobileSAM.git \
                git+https://github.com/naokiyokoyama/depth_camera_filtering \
                git+https://github.com/naokiyokoyama/bd_spot_wrapper.git \
                git+https://github.com/IDEA-Research/GroundingDINO.git \
		hydra-core

RUN python3.10 -m pip install --upgrade pip && \
    git clone --recursive --branch v0.2.4 https://github.com/facebookresearch/habitat-sim.git && \
    cd habitat-sim && \
    python3.10 -m pip install -r requirements.txt && \ 
    python3.10 setup.py install --headless --with-cuda --bullet 

RUN git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git #&& \
    cd habitat-lab/habitat-lab && \
    pip install -r requirements.txt && \
    cd ../../ && \
    pip install -r habitat_baselines/habitat_baselines/rl/requirements.txt && \
    pip install -r habitat_baselines/habitat_baselines/rl/ddppo/requirements.txt && \
    python3.10 habitat-lab/habitat-lab/setup.py develop --all
    
RUN cd habitat-lab/habitat-lab/ && python3.10 -m pip install .
RUN cd habitat-lab/habitat-baselines && python3.10 -m pip install . 

RUN cd habitat-lab && \
    wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip && \
    unzip habitat-test-scenes.zip 


RUN git clone https://github.com/KumarRobotics/vlfm.git && \
    cd vlfm && pip install . 
