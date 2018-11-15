# -*- coding: utf-8 -*-

import os
import json
import time
import ConfigParser
from conductor.sdk import ConductorSDK
from face_features_module.mxnet_feature_extractor import features

if __name__ == "__main__":
    wfname = "image_face_cluster_v1"

    cfg = ConfigParser.ConfigParser()
    cfg.read('.params.conf')

    conf_json = cfg.get('face_feature', 'network_config')
    save_dir = cfg.get('face_feature', 'save_dir')

    while True:
        # task: image_face_cluster_v1-image_face_feature_task
        # inputs: 特征提取文件输入文件路径, 即人脸对齐结果
        # output: 特征提取文件结果保存目录
        task, inputs, output = ConductorSDK.poll_image_face_feature_inprogress_task(wfname)
        if task is None:
            time.sleep(1)
            continue

        print("Processing task ---> %s"%task)
        print("Input ---> %s"%inputs)
        print("Output ---> %s"%output)
        feature_result = os.path.join(output, 'face_feature.json')

        success = features(inputs,
                           feature_result,
                           conf_json, 
                           save_dir=save_dir)

        task_status = "COMPLETED" if success == 0 else "FAILED"
        print("task status ---> %s"%task_status)

        post_status = ConductorSDK.post_image_face_feature_inprogress_result(task,
                                                                             task_status,
                                                                             feature_result)
        print("face feature post status ---> %s"%post_status)
        
