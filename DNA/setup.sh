#!/bin/bash

# Download the data and dependencies
gdown 1_gHXRWcQIlcRma17TkTHp02Dlaz_QBhj
gdown 14OJtDDdV3PQ6q3HrIQNG7kkTr6iC1myu

# Unzip and clean up
unzip r_dada_dependencies.zip -d /
unzip alejando_data_xochimilco.zip -d data

rm r_dada_dependencies.zip
rm alejando_data_xochimilco.zip

# Download the utils.R file
wget https://github.com/MScEcologyAndDataScienceUCL/BIOS0032_AI4Environment/raw/refs/heads/main/DNA/utils.R

# Install basta
uv tool install --from https://github.com/timkahlke/BASTA.git --with leveldb --with plyvel --with krona --with wget --python 3.11 basta

uv tool update-shell

# Install blast
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.17.0+-x64-linux.tar.gz
tar zxvpf ncbi-blast-2.17.0+-x64-linux.tar.gz -C /usr/local/ --strip-components=1
rm ncbi-blast-2.17.0+-x64-linux.tar.gz
