FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime as base

# updates and install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# colored prompts
RUN echo PS1="'"'\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\ \033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '"'" >> /root/.bashrc
RUN echo alias ls="'"'ls --color=auto'"'" >> /root/.bashrc

# enable nvidia
ENV NVIDIA_VISIBLE_DEVICES all
ENV EVIDIA_DRIVER_CAPABILITIES=compute,utility

# =========== development image============
FROM base as development

# pycocotools
RUN conda install -c conda-forge pycocotools

# segment anything, for possible additional annotations
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# for opencv
RUN apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 

# install packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

        
# =========== prod image ==================
# tbd, but opportunity for cutting down the docker image size if
# needed in production