# -*- coding:utf-8 -*-
import os
import json
from det import face_det
from align import face_align_crop
from mxnet_feature_extractor import features
from hierarchical import clusters

if __name__ == "__main__":
    print('step1 ,face det')
    img_urls = []
    with open('./input.json') as f:
        for line in f:
            #line = json.loads(line.strip())
            #url = line['url']
            img_urls.append(line.strip())
    face_det(img_urls,'./face_det.json')

    # step 2 ,face align
    print('step2 ,face align')
    face_align_crop('./face_det.json','./align_model','./aligned_face','aligned.json',threshold=0.7,gpu_id=-1)

    # step 3 , feat extra
    print('step3, feature extra')
    features('./aligned.json','feat.json','./extractor_config.json','./features')

    # step 4 , cluster
    print('step4, cluster')
    clusters('./feat.json','./cluster.json',0.6,nProcess=1)
