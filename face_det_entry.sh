#!/bin/bash

pip install requests
pip install futures

cd /workspace/mnt/group/alg-pro/pengyuanzhuo/face_cluster/conductor-sdk-test
python setup.py install

cd ..
python face_det_entry.py
