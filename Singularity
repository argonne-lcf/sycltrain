Bootstrap: docker
From: centos:7

%runscript
exec echo "dpc++ singularity tutorial image"
exec cat /etc/redhat-release

%environment

%setup

%post
# setup repo
echo -e "[oneAPI]\nname=Intel(R) oneAPI repository\nbaseurl=https://yum.repos.intel.com/oneapi\nenabled=1\ngpgcheck=1\nrepo_gpgcheck=1\ngpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB" > /tmp/oneAPI.repo
mv /tmp/oneAPI.repo /etc/yum.repos.d

# install software
yum update -y
yum install -y epel-release
yum install -y centos-release-scl
yum install -y devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-gcc-gfortran
scl enable devtoolset-8 -- bash
yum install -y gdb git make cmake
yum install -y python
yum install -y intel-basekit
yum install -y intel-hpckit
yum install -y strace

# remove unneeded icd files
rm /etc/OpenCL/vendors/Altera.icd
rm /etc/OpenCL/vendors/Intel_FPGA_SSG_Emulator.icd

# setup demo code
mkdir -p /code
cd /code
git clone https://github.com/kevin-harms/sycltrain.git
