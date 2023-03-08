#!/bin/bash
echo "Installing spatial libraries..."
( 
    sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
    sudo apt-get update && \
    sudo apt-get install -y libudunits2-dev libgdal-dev libgeos-dev libproj-dev libsqlite0-dev
) 1>&2
if [ $? != 0 ]
then
  echo "Unable to install spatial dependencies"
    exit 1
fi
echo "done."

echo "Downloading data..."
if [ ! -d "data" ] 
then
    git clone https://github.com/MScEcologyAndDataScienceUCL/BIOS0032_AI4Environment tmp 1>&2
    cp -r tmp/9_AItoEcologicalModels/data/ data/ 1>&2
    rm -rf tmp 1>&2
fi
echo "done"

# Install R libraries
echo 'dependencies = c("dplyr", "lme4", "rgdal", "sf", "terra", "MetBrewer")' >> requirements.R
echo 'dependencies = dependencies[!(dependencies %in% installed.packages()[,"Package"])]' >> requirements.R
echo 'if(length(dependencies)) install.packages(dependencies)' >> requirements.R

echo "Installing R libraries..."
Rscript requirements.R 1>&2
echo "done."

echo "Installing Python dependencies..."
# Install package to run R from Python
pip install rpy2==3.5.1 1>&2
echo "done."

echo "All done!"
