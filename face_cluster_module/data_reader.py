# -*- coding:utf-8 -*-

# from sklearn import cluster
#from scipy.spatial.distance import cosine as ssd_cosine_dist
import numpy as np
#import scipy.misc as sc
import time
import os
import os.path as osp
import shutil
import multiprocessing
from matio import load_mat

FEATURE_DIENSION = 512

def read_txtlist(path, sourceDir):
    pathList = []
    with open(path, 'r') as f:
        aPath = f.readline().strip()
        while aPath:
            if aPath.startswith('./'):
                aPath = aPath[2:]
            if aPath.endswith(".dat"):
                print ('now we don\'t support this feature format, just npy and bin feature are available!!!')
            if aPath.endswith(".npy") or aPath.endswith(".bin"):
                pathList.append(os.path.join(sourceDir, aPath))
                aPath = f.readline().strip()
    return pathList

def feature_data_reader_fromList(filePathList):
    name = multiprocessing.current_process().name
    # Use first one to initialize
    if filePathList[0].endswith('npy'):
        globle_para=True
    else:
        globle_para=False

    if globle_para:
        feature_list = np.load(filePathList[0])
    else:
        feature_list = load_mat(filePathList[0])
    assert feature_list.shape[0] > 0
    # Concat else
    cnt = 0
    noHeadFilePathList = filePathList[1:]
    while(cnt < len(noHeadFilePathList)):
        fileFullPath = noHeadFilePathList[cnt]
        if cnt%1000 == 0:
            print "Process", name, "done concating", cnt
        if globle_para:
            featureVec = np.load(fileFullPath)
        else:
            featureVec = load_mat(fileFullPath) 

        if len(featureVec.shape) > 0:# == 512:
            feature_list = np.vstack((feature_list, featureVec))
        else:
            print 'in', "Process", name
            print feature_list.shape[0], len(noHeadFilePathList), "Process", name
            noHeadFilePathList.pop(cnt)
            print feature_list.shape[0], len(noHeadFilePathList), "Process", name
            cnt -= 1
            print feature_list.shape, featureVec.shape, fileFullPath
        cnt += 1
    print feature_list.shape[0], len(noHeadFilePathList), "Process", name
    newFilePathList = [filePathList[0]] + noHeadFilePathList
    print feature_list.shape[0], len(newFilePathList), "Process", name
    return np.asarray(feature_list), newFilePathList

def multiprocess_feature_data_reader(feat_list, nProcess=1):
    """
    Args:
    -----
    feat_list : feat list file path
    nProcess : num of process
    """
    feature_list = None
    total_line = len(feat_list)
    p = multiprocessing.Pool(nProcess)
    pos = 0
    step = total_line / nProcess + 1
    resList = []
    for i in range(nProcess):
        if i == nProcess - 1:
            resList.append(p.apply_async(feature_data_reader_fromList,args=(feat_list[pos:],)))
        else:
            resList.append(p.apply_async(feature_data_reader_fromList,args=(feat_list[pos:pos+step],)))
            pos += step
    p.close()
    p.join()

    for i in range(nProcess):
        if i == 0:
            feature_list, filePathList = resList[i].get()
        else:
            feature_block, filePathList_part = resList[i].get()
            feature_list = np.vstack((feature_list, feature_block))
            filePathList = filePathList + filePathList_part
    return np.asarray(feature_list), filePathList
