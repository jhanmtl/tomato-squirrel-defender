FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime as base


# colored prompts
RUN echo PS1="'"'\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\ \033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '"'" >> /root/.bashrc
RUN echo alias ls="'"'ls --color=auto'"'" >> /root/.bashrc

# enable nvidia
ENV NVIDIA_VISIBLE_DEVICES all
ENV EVIDIA_DRIVER_CAPABILITIES=compute,utility

# =========== development image============
FROM base as development

# install packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# pycocotools
RUN conda install -c conda-forge pycocotools

# install Segement Anything Model, for additional data anntation
# updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# =========== prod image ==================
# tbd, but opportunity for cutting down the docker image size if
# needed in production