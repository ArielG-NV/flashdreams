FROM nvcr.io/nvidia/pytorch:26.02-py3

# install ffmpeg
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
