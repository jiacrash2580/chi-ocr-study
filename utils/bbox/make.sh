#!/usr/bin/env bash
cython -3 bbox.pyx
cython -3 nms.pyx
python3 setup.py build_ext --inplace
mv bbox*.so bbox.so
mv nms*.so nms.so
rm -rf build