#! /bin/bash

cd /workspace/mnt/group/alg-pro/pengyuanzhuo/face_cluster/conductor-sdk-test
python setup.py install

pip install mxnet==1.0.0 -i https://pypi.douban.com/simple

cd ..
python face_align_entry.py
