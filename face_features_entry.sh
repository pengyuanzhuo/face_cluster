#! /bin/bash

cd /workspace/mnt/group/alg-pro/pengyuanzhuo/face_cluster/conductor-sdk-test
python setup.py install

pip install mxnet-cu90==1.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install easydict

cd ..
#python features.py
python face_features_entry.py
