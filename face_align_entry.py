# -*- coding: utf-8 -*-

import os
import json
import time
import ConfigParser
from conductor.sdk import ConductorSDK
from face_align_module.face_align import face_align_crop
import mxnet as mx

print(mx.__version__)

if __name__ =="__main__":
    wfname = "image_face_cluster_v1"

    cfg = ConfigParser.ConfigParser()
    cfg.read('./params.conf')

    model_path = cfg.get('face_align', 'model_path')
    save_dir = cfg.get('face_align', 'aligned_face_dir')
    bucket_name = cfg.get('face_align', 'bucket_name')
    ak = cfg.get('auth', 'ak')
    sk = cfg.get('auth', 'sk')
    threshold = cfg.get('face_align', 'threshold')
    gpu_id = cfg.get('face_align', 'gpu_id')

    while True:
        # task: image_face_cluster_v1-image_face_alignment_task
        # inputs: 人脸对齐输入文件路径, 即人脸检测结果
        # output: 人脸对齐结果保存路径
        task, inputs, output = ConductorSDK.poll_image_face_alignment_inprogress_task(wfname)
        if task is None:
            time.sleep(1)
            continue

        print("Processing task ---> %s"%task)
        print("Input ---> %s"%inputs)
        print("Output ---> %s"%output)

        align_result = os.path.join(output, 'face_aligned.json')

        try:
            success = face_align_crop(inputs,
                                      model_path,
                                      save_dir,
                                      bucket_name,
                                      ak,
                                      sk,
                                      align_result,
                                      threshold = threshold,
                                      gpu_id = gpu_id)

            task_status = "COMPLETED" if success == 0 else "FAILED"
        except Exception as e:
            print("Align ---> Exception: %s"% e)
            task_status = "FAILED"

        post_status = ConductorSDK.post_image_face_alignment_inprogress_result(task,
                                                                               task_status,
                                                                               align_result)

        print("face align post status ---> %s"% post_status)
       
