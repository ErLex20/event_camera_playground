# DUA environment image.
#
# Roberto Masocco <r.masocco@dotxautomation.com>
#
# March 9, 2026

# Copyright 2026 dotX Automation s.r.l.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM dotxautomation/dua-foundation:x86-cudev

# Change this to match the UID of the owner of this file on the system
ARG USER_UID=1000

ENV DEBIAN_FRONTEND=noninteractive

# IMAGE SETUP START #
# Install DAGr and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install E2Calib runtime dependencies into the default DUA Python environment.
RUN /opt/dua-venv/bin/python -m pip install --no-cache-dir \
    "numpy>=1.24,<2" \
    "h5py>=3.10" \
    scipy \
    opencv-python \
    tqdm

RUN unset PYTHONPATH && \
    /usr/bin/python3.12 -m venv /opt/dagr-venv && \
    . /opt/dagr-venv/bin/activate && \
    unset PYTHONPATH && \
    pip install --upgrade pip setuptools wheel && \
    pip install "numpy<2" && \
    pip install \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir \
        torch-scatter torch-cluster torch-spline-conv torch-sparse \
        -f https://data.pyg.org/whl/torch-2.7.0+cu128.html && \
    pip install torch-geometric h5py hdf5plugin && \
    pip install \
        wandb numba plotly matplotlib pycocotools opencv-python \
        scikit-video pandas ruamel.yaml seaborn
ENV HDF5_PLUGIN_PATH=/opt/dagr-venv/lib/python3.12/site-packages/hdf5plugin/plugins

ENV DAGR_DIR=/opt/dagr
RUN git clone --branch acretu/fix/install --single-branch \
        https://github.com/ErLex20/dagr.git ${DAGR_DIR} && \
    mkdir -p ${DAGR_DIR}/libs && \
    cd ${DAGR_DIR}/libs && \
    git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && git checkout 70f454304e1a38378200459dd2dbca0f0f4a5ab4 && cd .. && \
    git clone https://github.com/Megvii-BaseDetection/YOLOX.git && \
    cd YOLOX && \
    git checkout 618fd8c08b2bc5fac9ffbb19a3b7e039ea0d5b9a && \
    sed -i 's/onnx-simplifier==0.4.10/onnx-simplifier>=0.4.13/' requirements.txt && \
    cd .. && \
    git clone https://github.com/uzh-rpg/dsec-det.git && \
    cd dsec-det && git checkout 81e381dc0fc1b1a540a604a970a37de038abb83b && \
    cd ../.. && \
    chgrp -R internal /opt/dagr && \
    chmod -R g+rwX /opt/dagr

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
RUN . /opt/dagr-venv/bin/activate && \
    unset PYTHONPATH && \
    pip install --no-build-isolation -e ${DAGR_DIR}/libs/dsec-det && \
    pip install --no-build-isolation -e ${DAGR_DIR}/libs/detectron2 && \
    pip install --no-build-isolation -e ${DAGR_DIR}/libs/YOLOX && \
    pip install --no-build-isolation -e ${DAGR_DIR}
# prophesee_evk4_driver START #
# Install metavision_driver and related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-metavision-driver \
      ros-jazzy-event-camera-py \
      ros-jazzy-ament-cmake-clang-format  && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/ros/jazzy/opt/openeb_vendor/lib:$LD_LIBRARY_PATH
# prophesee_evk4_driver END #
# IMAGE SETUP END #

# Copy user configuration files to /etc/skel
COPY ./aliases.sh \
  ./bashrc \
  ./colcon-defaults.yaml \
  ./commands.sh \
  ./nanorc \
  ./p10k.zsh \
  ./ros2.sh \
  ./vimrc \
  ./zshrc \
  /etc/skel/staging/

# Do the following:
# - Populate /etc/skel with dot-named config files and needed directories, all mount points
# - Handle existing user conflicts and create the internal user
# - Grant passwordless sudo to internal user and suppress hostname resolution warning
# - Create mount point for remote filesystems
RUN mkdir -p /etc/skel/.colcon /etc/skel/.ssh && \
  chmod -R 700 /etc/skel/.ssh && \
  mkdir /etc/skel/workspace /etc/skel/zsh_history && \
  mv /etc/skel/staging/colcon-defaults.yaml /etc/skel/.colcon/defaults.yaml && \
  for f in /etc/skel/staging/*; do mv "$f" "/etc/skel/.$(basename $f)"; done && \
  rmdir /etc/skel/staging; \
  (getent passwd ${USER_UID} || true) && userdel -r $(getent passwd ${USER_UID} | cut -d: -f1) 2>/dev/null || true; \
  useradd -r -m -s /usr/bin/zsh -u ${USER_UID} -G adm,dialout,internal,plugdev,sudo,tty,video neo && \
  printf 'Defaults !fqdn\nneo ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/neo && \
  chmod 0440 /etc/sudoers.d/neo && \
  mkdir -p /mnt/remote && \
  chown -R neo:neo /mnt/remote

# Switch to internal user
USER neo
ENV HOME=/home/neo
WORKDIR ${HOME}

# Configure Zsh for internal user
ENV ZSH=${HOME}/.oh-my-zsh
ENV ZSH_CUSTOM=${ZSH}/custom
ENV ZSH_PLUGINS=${ZSH_CUSTOM}/plugins
ENV ZSH_THEMES=${ZSH_CUSTOM}/themes
RUN wget -qO- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | zsh || true; \
  git clone --single-branch --branch 'master' --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_PLUGINS}/zsh-syntax-highlighting; \
  git clone --single-branch --branch 'master' --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_PLUGINS}/zsh-autosuggestions; \
  git clone --single-branch --depth 1 https://github.com/romkatv/powerlevel10k.git ${ZSH_THEMES}/powerlevel10k

ENV DEBIAN_FRONTEND=dialog

# By default, start a basic shell
CMD ["bash"]
