BootStrap: docker
From: nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# labels visible upon inspect
%labels
    Maintainer Francois Aguet
    tensorqtl_version 1.0.10
    R_Version 4.4.2
    Ubuntu_version 24.04
    CUDA_version 12.9.0
    Image_Version v1.0
    Repository https://github.com/broadinstitute/tensorqtl

# help message
%help
    This will run tensorqtl, use --nv and have nvidia-container-cli available on the host to make sure you can run with GPU acceleration (https://docs.sylabs.io/guides/latest/user-guide/gpu.html)

# run tensorqtl
%apprun tensorqtl
    exec python -m tensorqtl "${@}"

# run tensorqtl
%runscript
    exec python -m tensorqtl "${@}"

# install instructions
%post
    #####################
    # Software versions #
    #####################
    export R_VERSION=4.4.2
    export HTSLIB_VERSION=1.21
    export BCFTOOLS_VERSION=1.21
    export TENSORQTL_VERSION=1.0.10
    export PGENLIB_COMMIT='cf54067a3a40417125a9d13640d5ea3450d02dc0'


    ##############
    # fix locale #
    ##############
    apt-get update
    # install locales
    apt-get install -y --no-install-recommends \
        locales
    # Configure default locale
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
    locale-gen en_US.utf8
    /usr/sbin/update-locale LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8


    #####################################
    # Install R and python dependencies #
    #####################################
    apt-get update
    # install system libraries R and Python will depend on
    apt-get install -y --no-install-recommends \
        software-properties-common \
        dirmngr \
        wget \
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
        python3-apt \
        python3-venv \
        python3-dev \
        libssl-dev \
        sudo \
        unzip \
        wget \
        zlib1g-dev \
        gcc-11 \
        gcc-13 \
        git
    # get public key for CRAN
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
        tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
    # add R CRAN repository
    add-apt-repository \
        "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
    # install R
    apt-get install -y --no-install-recommends \
        r-recommended=${R_VERSION}* \
        r-base=${R_VERSION}* \
        r-base-core=${R_VERSION}* \
        r-base-dev=${R_VERSION}*
  

    ################################
    # Install HTSlib and BCFtools #
    ###############################
    # install htslib
    cd /opt
    # download from github page
    wget --no-check-certificate https://github.com/samtools/htslib/releases/download/${HTSLIB_VERSION}/htslib-${HTSLIB_VERSION}.tar.bz2
    # extract archive
    tar -xf htslib-${HTSLIB_VERSION}.tar.bz2 && rm htslib-${HTSLIB_VERSION}.tar.bz2 && cd htslib-${HTSLIB_VERSION}
    # configure before installation
    ./configure --enable-libcurl --enable-s3 --enable-plugins --enable-gcs
    # install and clean up
    make && make install && make clean
    cd

    # install bcftools
    cd /opt
    # download from github page
    wget --no-check-certificate https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2
    tar -xf bcftools-${BCFTOOLS_VERSION}.tar.bz2 && rm bcftools-${BCFTOOLS_VERSION}.tar.bz2 && cd bcftools-${BCFTOOLS_VERSION}
    # configure before installation
    ./configure --with-htslib=system
    # install and clean up
    make && make install && make clean
    cd


    ##################
    # Install qvalue #
    ##################
    Rscript -e 'if (!requireNamespace("BiocManager", quietly = TRUE)) {install.packages("BiocManager")}; BiocManager::install("qvalue");'
  

    #####################
    # Install tensorqtl #
    #####################
    # install tensorqtl dependencies
    cd /opt
    python3 -m venv venv
    . /opt/venv/bin/activate
    # upgrade setup tools
    pip install --upgrade pip setuptools
     # install other libraries
    pip install numpy==1.26.4 pandas scipy
    pip install pandas-plink ipython jupyter jupyterlab IProgress matplotlib pyarrow torch rpy2 gcsfs
    # install dependencies
    pip install "cython>=0.29.21"
    # clone pgenlib
    git clone https://github.com/chrchang/plink-ng
    # go to python folder of pgenlib
    cd plink-ng/2.0/Python
    # load the specific commit
    git checkout ${PGENLIB_COMMIT}
    # install pgenlib
    python setup.py build_clib build_ext -i
    python setup.py install
    # back to home
    cd /opt
    # finally install tensorqtl version
    pip install tensorqtl==${TENSORQTL_VERSION}
    # or install the dev version from the master branch
    #git clone https://github.com/broadinstitute/tensorqtl.git
    #cd tensorqtl
    #pip install .
    #cd
    echo '. /opt/venv/bin/activate' >> $SINGULARITY_ENVIRONMENT

    ###########
    # cleanup #
    ###########
    apt-get clean
    apt-get autoremove -y

    