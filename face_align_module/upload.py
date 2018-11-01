#-*- coding: utf-8 -*-
# upload file to bucket

import os
import sys
import json
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor

def _upload(file_fn,bucket,key=None):
    if key is None:
        key = os.path.split(file_fn)[-1]
    os.system('qrsctl put %s %s %s'%(bucket,key,file_fn))
    return file_fn

def upload(file_fn_list,bucket,key=None,thread=10):
    """
    file_fn_list : file full path list
    bucket : dst bucket
    key : upload name
    thread : num of thread to upload
    """
    with ProcessPoolExecutor(max_workers=thread) as exe:
        future_tasks = [exe.submit(_upload,file_fn,bucket,key) for file_fn in file_fn_list]
        for task in as_completed(future_tasks):
            print('%s done.'%task.result())

def upload_face(cluster_log,bucket,key=None,thread=10):
    url_list = []
    with open(cluster_log,'r') as f:
        cluser_list = json.load(f)
        for cluster in cluser_list:
            pass

if __name__ == "__main__":
    upload_face('./cluster.json','ww')
