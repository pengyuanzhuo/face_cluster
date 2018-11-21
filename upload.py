#-*- coding: utf-8 -*-
# upload file to bucket

import os
import json
import time
import requests
from qiniu import Auth, put_file, etag
import qiniu.config
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

def get_bucket_host(bucket_name, auth=None, ak=None, sk=None):
    """
    获取指定bucket的host
    Args:
    ----
    bucket_name: 待查询的bucket名
    auth: 七牛鉴权
    ak: bucket_name 所在账号对应的ak
    sk: ...sk
    auth 或 (ak, sk)两者指定一个即可

    Return:
    ------
    host: bucket 对应的访问外链
    """
    request_url = 'http://api.qiniu.com/v6/domain/list?tbl=%s'%bucket_name
    if auth is None:
        auth = Auth(ak, sk)
    token = auth.token_of_request(request_url)
    request_header = {'Authorization': 'QBox %s'%token}
    r = requests.get(request_url, headers=request_header)
    host = 'http://' + r.json()[0]
    return host

def upload(local_file, bucket_name, ak, sk, bucket_host=None, prefix=None, key=None):
    """
    上传指定文件到指定bucket
    Args:
    ----
    local_file: 待上传文件路径
    bucket_name: 指定上传的bucket
    ak: bucket 所在账号ak
    sk: bucket 所在账号sk
    bucket_host: bucket 对应的外链域名, 若不指定, 则动态获取
    prefix: 上传前缀, 若不指定, 则无前缀
    key: 上传后保存的文件名, 若不指定, 则与本地文件同名
    
    Return:
    ------
    url: 上传文件的访问外链
    """
    auth = Auth(ak, sk)
    if bucket_host is None:
        # 获取bucket外链
        bucket_host = get_bucket_host(bucket_name, auth)

    if key is None:
        key = os.path.split(local_file)[-1]

    if prefix is not None:
        key = os.path.join(prefix, key)

    token = auth.upload_token(bucket_name, key, 3600)
    ret, info = put_file(token, key, local_file)
    assert ret['key'] == key
    assert ret['hash'] == etag(local_file)

    url = bucket_host + '/' + key
    # 强制刷新CDN
    rd = int(round(time.time()*1000))
    url = url + '?v=%d'%rd
    return url


def upload_concurrent(file_list, bucket_name, ak, sk, prefix=None, key=None, thread=10):
    """
    多线程上传
    file_list: 待上传文件list
    bucket_name: 指定上传的bucket
    ak: bucket 所在账号ak
    sk: ...sk
    prefix: 上传前缀, 若不指定, 则无前缀
    key: 上传后文件名, 若不指定, 则与本地文件同名
    thread: 上传线程数

    Return:
    ------
    url_dict: 上传结果dict {local_file: url}
              若某个文件上传失败, 则对应的url为None
    """
    url_dict = {}
    bucket_host = get_bucket_host(bucket_name, ak=ak, sk=sk)
    with ThreadPoolExecutor(max_workers=thread) as exe:
        future_tasks = {exe.submit(upload, 
                                   local_file, 
                                   bucket_name, 
                                   ak, 
                                   sk, 
                                   bucket_host, 
                                   prefix): local_file 
                        for local_file in file_list}
        all_file = len(file_list)
        count = 1
        for task in as_completed(future_tasks):
            if task.done():
                local_file = future_tasks[task]
                try:
                    url = task.result()
                    print('%d / %d %s done.'%(count, all_file, url))
                except Exception as e:
                    url = None
                    print('%d / %d %s --> %s'%(count, all_file, local_file, e))
                count += 1
                url_dict[local_file] = url
    return url_dict

def upload_dir(root_dir, bucket_name, ak, sk, keep_struct=True, prefix=None, thread=10):
    """
    上传文件夹内容
    Args:
    ----
    root_dir: 待上传根目录, 该目录下所有文件都将被上传.
              skip 隐藏文件, __MACOSX
    bucket_name: 指定上传的bucket
    ak: bucket所在账号的access key
    sk: ..secret key
    prefix: 上传前缀, 默认为None
    keep_struct: bool, 上传后是否保持原目录结构, 默认True
    thread: 上传线程数, 默认10

    Return:
    url_dict: 上传结果dict {local_file: url}
              若某个文件上传失败, 则对应的url为None

    Warining: UNCLEAR IN MEANING!!! (TODO)
    """
    url_dict = {}

    # 遍历目录
    for root, _, files in os.walk(root_dir):
        files_fn_list = [os.path.join(root, file) for file in files
                         if not (file.startswith('.') and 
                            file == "__MACOSX")]
        if keep_struct:
            sub_dir = root.split(root_dir)[-1]
            sub_dir = sub_dir.strip('/')
            _prefix = os.path.join(prefix, sub_dir) if prefix is not None else sub_dir
        else:
            _prefix = prefix
        sub_url_dict = upload_concurrent(files_fn_list, bucket_name, ak, sk, _prefix, thread=thread)
        url_dict.update(sub_url_dict)

    return url_dict

