#! /bin/bash

cd /workspace/mnt/group/alg-pro/pengyuanzhuo/face_cluster/conductor-sdk-test
python setup.py install

cd ..
pip install scipy

python face_cluster_entry.py
