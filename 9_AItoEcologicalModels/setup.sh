#!/bin/bash
echo -n "Installing spatial libraries... "
(
	sudo -n add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable &&
		sudo -n apt-get -qq update &&
		sudo -n apt-get -yqq install \
			libudunits2-dev \
			libgdal-dev \
			libgeos-dev \
			libproj-dev \
			libsqlite0-dev
) >>spatial.log 2>&1
if [ $? != 0 ]; then
	echo "unable to install spatial dependencies!"
else
	echo "done."
fi

echo -n "Downloading data... "
if [ ! -d "data" ]; then
	git clone https://github.com/MScEcologyAndDataScienceUCL/BIOS0032_AI4Environment tmp >>data.log 2>&1
	cp -r tmp/9_AItoEcologicalModels/data/ data/ >>data.log 2>&1
	rm -rf tmp >>data.log 2>&1
fi
echo "done."

r_packages="dplyr lme4 rgdal sf terra MetBrewer"

echo "Installing R libraries:"
for r_package in $r_packages; do
  echo -n "    [+] Installing $r_package... "
  Rscript -e "if (!requireNamespace('$r_package', quietly = TRUE)) install.packages('$r_package')" >>r.log 2>&1
  echo "done."
done
echo "done."

echo -n "Installing Python dependencies... "
pip install rpy2==3.5.1 >>python.log 2>&1
echo "done."

echo "All done!"
