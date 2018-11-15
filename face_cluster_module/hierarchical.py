# -*- coding: utf-8 -*-
import json
import numpy as np
import scipy.cluster.hierarchy as hcluster
from data_reader import multiprocess_feature_data_reader,feature_data_reader_fromList
import shutil
import os
import sys

def clusters(feat_log, cluster_log, threshold, nProcess=1):
    """
    threshold : cluster threshold
    """
    if not os.path.exists(feat_log):
        print('%s is None'%feat_log)
        return -1
    feat_face_dict = {}
    with open(feat_log,'r') as f:
        for line in f:
            line = json.loads(line.strip())
            feat = line['feat']
            face = line['aligned_url']
            feat_face_dict[feat] = face
    if feat_face_dict == {}:
        print('no face to cluster')
        return -2
    print(feat_face_dict.keys())
    # feats n X dimshi
    feats, feat_path_list = multiprocess_feature_data_reader(feat_face_dict.keys(), nProcess)
    # --- cluster ---
    print("feat shape --->", feats.shape)
    clusters = hcluster.fclusterdata(feats,
                                     threshold,
                                     metric="cosine",
                                     method="average",
                                     criterion="distance")
    cluser_num = {}
    for cluser in set(clusters):
        num = np.sum(clusters == cluser)
        cluser_num[cluser] = num
    
    id_urls_dict = {}
    for i in range(len(feat_path_list)):
        id = clusters[i]
        if cluser_num[id] == 1:
            id = -1
        if str(id) in id_urls_dict:
            id_urls_dict[str(id)].append(feat_face_dict[feat_path_list[i]])
        else:
            id_urls_dict[str(id)] = []
            id_urls_dict[str(id)].append(feat_face_dict[feat_path_list[i]])

    # 封装成指定格式输出
    item_list = []
    for id, urls in id_urls_dict.items():
        item = {}
        item['cluster_id'] = id
        item['img_lst'] = [{"url": url} for url in urls]
        item_list.append(item)
    with open(cluster_log,'w') as f:
        json.dump(item_list, f, indent=4)
    return 0

if __name__ == "__main__":
    clusters('./feat.json', './cluster.json', 0.6)

