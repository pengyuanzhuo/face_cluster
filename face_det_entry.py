# -*- coding: utf-8 -*-

import os
import json
import time
import ConfigParser
from conductor.sdk import ConductorSDK
from face_det_module.face_det import face_det


def urls(urlfile):
    url_list = []
    with open(urlfile, 'r') as f:
        for line in f:
            url = line.strip()
            url_list.append(url)
    return url_list

if __name__ == "__main__":
    wfname = "image_face_cluster_v1"

    cfg = ConfigParser.ConfigParser()
    cfg.read('./param.conf')
    ak = cfg.get('auth', 'ak')
    sk = cfg.get('auth', 'sk')

    while True:
        # task: 待处理任务, image_face_cluster_v1-image_face_detect_task
        # inputs: 人脸检测输入文件路径, 即url list 文件路径
        # output: 人脸检测结果保存路径
        task, inputs, output = ConductorSDK.poll_image_face_detect_inprogress_task(wfname)
        if task is None:
            time.sleep(1)
            continue
        
        print("Processing task ---> %s"%task)
        print("Input ---> %s"%inputs)
        print("Output ---> %s"%output)

        # get urllist file
        with open(inputs, 'r') as f:
            url_file = json.loads(f.readlines()[0].strip())
        img_urls = urls(url_file)
        result = os.path.join(output, 'face_det.json')
        task_status = "COMPLETED"
        try:
            face_det(img_urls, result, ak, sk)
        except Exception as e:
            print("face det --> %s"%e)
            task_status = "FAILED"

        post_status = ConductorSDK.post_image_face_detect_inprogress_result(task,
                                                                            task_status,
                                                                            result)
        print("face det post status ---> %s"%post_status)
