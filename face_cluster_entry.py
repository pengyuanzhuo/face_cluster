# -*- coding: utf-8 -*-

import os
import time
import json
import ConfigParser
from conductor.sdk import ConductorSDK
from face_cluster_module.hierarchical import clusters

if __name__ == "__main__":
    wfname = "image_face_cluster_v1"

    cfg = ConfigParser.ConfigParser()
    cfg.read('./params.conf')
    threshold = cfg.get('face_cluster', 'threshold')
    nProcess = cfg.get('face_clusetr', 'nProcess')

    while True:
        task, inputs, output = ConductorSDK.poll_image_face_cluster_inprogress_task(wfname)
        if task is None:
            time.sleep(1)
            continue

        print("Processing task ---> %s"%task)
        print("Input ---> %s"%inputs)
        print("Output ---> %s"%output)
        cluster_result = os.path.join(output, 'face_cluster.json')

        success = clusters(inputs,
                           cluster_result,
                           threshold,
                           nProcess=nProcess)

        task_status = "COMPLETED" if success == 0 else "FAILED"
        print("cluster task status ---> %s"%task_status)

        post_status = ConductorSDK.post_image_face_cluster_inprogress_result(task,
                                                                             task_status,
                                                                             cluster_result)
        print("face cluster post status ---> %s"% post_status)

        if post_status == "COMPLETED":
            # 清理工作
            pass
