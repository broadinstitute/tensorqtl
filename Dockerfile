# Dockerfile for tensorQTL
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
MAINTAINER Francois Aguet

RUN apt-get update && apt-get install -y software-properties-common && \
    apt-get update && apt-get install -y \
        apt-transport-https \
        build-essential \
        cmake \
        curl \
        libboost-all-dev \
        libbz2-dev \
        libcurl3-dev \
        liblzma-dev \
        libncurses5-dev \
        libssl-dev \
        python3 \
        python3-pip \
        sudo \
        unzip \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/

# htslib
RUN cd /opt && \
    wget --no-check-certificate https://github.com/samtools/htslib/releases/download/1.9/htslib-1.9.tar.bz2 && \
    tar -xf htslib-1.9.tar.bz2 && rm htslib-1.9.tar.bz2 && cd htslib-1.9 && \
    ./configure --enable-libcurl --enable-s3 --enable-plugins --enable-gcs && \
    make && make install && make clean

# bcftools
RUN cd /opt && \
    wget --no-check-certificate https://github.com/samtools/bcftools/releases/download/1.9/bcftools-1.9.tar.bz2 && \
    tar -xf bcftools-1.9.tar.bz2 && rm bcftools-1.9.tar.bz2 && cd bcftools-1.9 && \
    make && make install && make clean

# install R
ENV DEBIAN_FRONTEND noninteractive
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
RUN apt update && apt install -y r-base r-base-dev
ENV R_LIBS_USER=/opt/R/3.6
RUN Rscript -e 'if (!requireNamespace("BiocManager", quietly = TRUE)) {install.packages("BiocManager")}; BiocManager::install("qvalue");'

# python modules
RUN pip3 install --upgrade pip setuptools
RUN pip3 install numpy pandas scipy
RUN pip3 install pandas-plink ipython jupyter matplotlib pyarrow torch rpy2 gcsfs

RUN cd /opt && \
    # wget https://github.com/broadinstitute/tensorqtl/archive/v1.0.2.tar.gz && \
    # tar -xf v1.0.2.tar.gz && mv tensorqtl-1.0.2 tensorqtl && \
    # rm v1.0.2.tar.gz
    wget https://github.com/broadinstitute/tensorqtl/archive/d739c40c41fb0d3dcb68047e98029c2cf74f9134.tar.gz && \
    tar -xf d739c40c41fb0d3dcb68047e98029c2cf74f9134.tar.gz && mv tensorqtl-d739c40c41fb0d3dcb68047e98029c2cf74f9134 tensorqtl && \
    rm d739c40c41fb0d3dcb68047e98029c2cf74f9134.tar.gz
RUN pip3 install -e /opt/tensorqtl/
