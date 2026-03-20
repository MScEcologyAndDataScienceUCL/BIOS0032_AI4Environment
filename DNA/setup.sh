#!/bin/bash

# Download the data and dependencies
gdown 1_gHXRWcQIlcRma17TkTHp02Dlaz_QBhj
gdown 14OJtDDdV3PQ6q3HrIQNG7kkTr6iC1myu

# Unzip
unzip alejando_data_xochimilco.zip -d data
unzip /content/r_dada_dependencies.zip -d /

# Download the utils.R file
wget https://github.com/MScEcologyAndDataScienceUCL/BIOS0032_AI4Environment/raw/refs/heads/main/DNA/utils.R

# Install basta
uv tool install --from https://github.com/timkahlke/BASTA.git --with leveldb --with plyvel --with krona --with wget --python 3.11 basta
