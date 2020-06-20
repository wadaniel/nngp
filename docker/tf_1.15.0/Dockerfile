FROM f41e06396ff3

## update aptitude
#RUN apt-get update && apt-get -y upgrade
#
## install aptitude essentials
#RUN apt-get -y install \
#    build-essential \
#    cmake \
#    git \
#    vim \
#    curl \
#    mpich \
#    libhdf5-mpich-dev \
#    python3-dev \
#    python3-pip \
#    python3-numpy

RUN python -m pip install keras numpy pandas

# add non-root user
RUN useradd -m sepp
WORKDIR /home/sepp
USER sepp
