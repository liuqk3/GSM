#!/usr/bin/env bash

echo " =====================build solvers ======================"
python bbox_setup.py build_ext --inplace

echo " ================== build psroi_pooling ================== "
cd models/psroi_pooling
python build.py

echo " =============== build roi_align =================="
cd ../
cd roi_align
python setup.py install

cd ../../