# -*- coding:utf-8 -*-

from __future__ import print_function
import os
import cv2
import json
import urllib
import numpy as np
import uuid
from mtcnn_aligner.fx_warp_and_crop_face import warp_and_crop_face, get_reference_facial_points
import math
import mxnet as mx
from facex_lib.upload import upload

def _adjust_input(in_data):
    """
    adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

    out_data = out_data.transpose((2, 0, 1))

    out_data = np.expand_dims(out_data, 0)
    out_data = (out_data - 127.5) * 0.0078125
    return out_data


def _generate_bboxes(scores_map, reg, scale, threshold):
    """
        generate bbox from feature scores_map
    Parameters:
    ----------
        scores_map: numpy array , n x m x 1
            detect score for each position
        reg: numpy array , n x m x 4
            bbox
        scale: float number
            scale of this detection
        threshold: float number
            detect threshold
    Returns:
    -------
        bbox array
    """
    stride = 2
    cellsize = 12

    t_index = np.where(scores_map > threshold)

    # find nothing
    if t_index[0].size == 0:
        return np.array([])

    dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = scores_map[t_index[0], t_index[1]]
    bbox = np.vstack([np.round((stride * t_index[1] + 1) / scale),
                      np.round((stride * t_index[0] + 1) / scale),
                      np.round(
        (stride * t_index[1] + 1 + cellsize) / scale),
        np.round(
        (stride * t_index[0] + 1 + cellsize) / scale),
        score,
        reg])

    return bbox.T

def _bbox_reg(bbox, reg):
    """
        calibrate bboxes

    Parameters:
    ----------
        bbox: numpy array, shape n x 5
            input bboxes
        reg:  numpy array, shape n x 4
            bboxex adjustment

    Returns:
    -------
        bboxes after refinement

    """
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)

    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox[:, 0:4] = bbox[:, 0:4] + aug

    return bbox


def _pad(bboxes, w, h):
    """
        pad the the bboxes, alse restrict the size of it

    Parameters:
    ----------
        bboxes: numpy array, n x 5
            input bboxes
        w: float number
            width of the input image
        h: float number
            height of the input image
    Returns :
    ------s
        dy, dx : numpy array, n x 1
            start point of the bbox in target image
        edy, edx : numpy array, n x 1
            end point of the bbox in target image
        y, x : numpy array, n x 1
            start point of the bbox in original image
        ex, ex : numpy array, n x 1
            end point of the bbox in original image
        tmph, tmpw: numpy array, n x 1
            height and width of the bbox

    """
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + \
        1,  bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box, )), np.zeros((num_box, ))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1

    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def _convert_to_square(bbox):
    """
        convert bbox to square

    Parameters:
    ----------
        bbox: numpy array , shape n x 5
            input bbox

    Returns:
    -------
        square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)

    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1

    return square_bbox

def _nms(boxes, overlap_threshold, mode='Union'):
    """
        non max suppression

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to
        # the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'Min':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlap_threshold)[0])))
    return pick

class MtcnnAligner(object):
    """
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
        this is a mxnet version
    """

    def __init__(self,
                 model_folder='./model',
                 accurate_landmark=True,
                 gpu_id=0
                 ):
        """
            Initialize the detector

            Parameters:
            ----------
                model_folder : string
                    path for the models
                gpu_id: int
                    >=0 gpu id;
                    <0 cpu mode
                accurate_landmark: bool
                    use accurate landmark localization or not

        """
        self.accurate_landmark = accurate_landmark
        self.gpu_id = gpu_id

        if gpu_id >= 0:
            ctx = mx.gpu(gpu_id)
        else:
            ctx = mx.cpu()

        # load 4 models from folder
        models = ['det1', 'det2', 'det3', 'det4']
        models = [os.path.join(model_folder, f) for f in models]

        self.RNet = mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.ONet = mx.model.FeedForward.load(models[2], 1, ctx=ctx)
        self.LNet = mx.model.FeedForward.load(models[3], 1, ctx=ctx)

    def align_face(self, img, face_rects):
        """
            detect face over img
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
        Retures:
        -------
            bboxes: numpy array, n x 5 (x1,y1,x2,y2,score)
                bboxes
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
                landmarks
        """

        if img is None:
            return [], []

        if isinstance(img, str):
            img = cv2.imread(img)

        # only works for color image
        if len(img.shape) != 3:
            return [], []

        height = img.shape[0]
        width = img.shape[1]

        ###############
        # First stage (unused and deleted)
        ###############

        total_boxes = np.array(face_rects)
        if total_boxes.size == 0:
            return [], []

        # convert list of 4 pts (x1,y1,x2,y2,x3,y3,x4,y4) into list of 2 pts
        # (x1,y1,x3,y3)
        if total_boxes.ndim == 3:
            # t_shape = total_boxes.shape
            # total_boxes = total_boxes.reshape((-1, t_shape[1]*t_shape[2]))
            # total_boxes = total_boxes[:, (0,1,4,5)]
            total_boxes = total_boxes[:, (0, 2), :]
            total_boxes = total_boxes.reshape((-1, 4))
        total_boxes = np.hstack(
            (total_boxes[:, 0:4], np.ones((total_boxes.shape[0], 1))))

        #############################################
        # second stage
        #############################################
        num_box = total_boxes.shape[0]
        total_boxes = _convert_to_square(total_boxes)

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = _pad(
            total_boxes, width, height)
        # (3, 24, 24) is the input shape for RNet
        input_buf = np.zeros((num_box, 3, 24, 24), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            # fix bug
            edy[i]=dy[i]+ ey[i]-y[i]   
            edx[i]=dx[i]+ ex[i]-x[i]
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1,
                :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            input_buf[i, :, :, :] = _adjust_input(cv2.resize(tmp, (24, 24)))

        output = self.RNet.predict(input_buf)

        # add scores
        total_boxes[:, 4] = output[1][:, 1].reshape((-1,))
        reg = output[0]

        total_boxes = _bbox_reg(total_boxes, reg)

        total_boxes = _convert_to_square(total_boxes)
        total_boxes[:, 0:4] = np.round(total_boxes[:, 0:4])

        #############################################
        # third stage
        #############################################
        num_box = total_boxes.shape[0]

        # pad the bbox
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = _pad(
            total_boxes, width, height)
        # (3, 48, 48) is the input shape for ONet
        input_buf = np.zeros((num_box, 3, 48, 48), dtype=np.float32)

        for i in range(num_box):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.float32)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1,
                :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            input_buf[i, :, :, :] = _adjust_input(cv2.resize(tmp, (48, 48)))

        output = self.ONet.predict(input_buf)

        # filter the total_boxes with threshold
        # passed = np.where(output[2][:, 1] > threshold[2])
        # total_boxes = total_boxes[passed]

        # if total_boxes.size == 0:
        #     return [], []

        total_boxes[:, 4] = output[2][:, 1].reshape((-1,))
        reg = output[1]
        points = output[0]

        # total_boxes[:, 4] = output[0].reshape((-1,))
        reg = output[1]

        # compute landmark points
        bbw = total_boxes[:, 2] - total_boxes[:, 0] + 1
        bbh = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[:, 0:5] = np.expand_dims(
            total_boxes[:, 0], 1) + np.expand_dims(bbw, 1) * points[:, 0:5]
        points[:, 5:10] = np.expand_dims(
            total_boxes[:, 1], 1) + np.expand_dims(bbh, 1) * points[:, 5:10]

        # nms
        total_boxes = _bbox_reg(total_boxes, reg)
        # pick = nms(total_boxes, 0.7, 'Min')
        # total_boxes = total_boxes[pick]
        # points = points[pick]

        if not self.accurate_landmark:
            return total_boxes.tolist(), points.tolist()

        #############################################
        # extended stage
        #############################################
        num_box = total_boxes.shape[0]
        patchw = np.maximum(
            total_boxes[:, 2] - total_boxes[:, 0] + 1, total_boxes[:, 3] - total_boxes[:, 1] + 1)
        patchw = np.round(patchw * 0.25)

        # make it even
        patchw[np.where(np.mod(patchw, 2) == 1)] += 1

        input_buf = np.zeros((num_box, 15, 24, 24), dtype=np.float32)
        for i in range(5):
            x, y = points[:, i], points[:, i + 5]
            x, y = np.round(x - 0.5 * patchw), np.round(y - 0.5 * patchw)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = _pad(np.vstack([x, y, x + patchw - 1, y + patchw - 1]).T,
                                                               width,
                                                               height)
            for j in range(num_box):
                tmpim = np.zeros((tmpw[j], tmpw[j], 3), dtype=np.float32)
                tmpim[dy[j]:edy[j] + 1, dx[j]:edx[j] + 1,
                      :] = img[y[j]:ey[j] + 1, x[j]:ex[j] + 1, :]
                input_buf[j, i * 3:i * 3 + 3, :,
                          :] = _adjust_input(cv2.resize(tmpim, (24, 24)))

        output = self.LNet.predict(input_buf)

        pointx = np.zeros((num_box, 5))
        pointy = np.zeros((num_box, 5))

        for k in range(5):
            # do not make a large movement
            tmp_index = np.where(np.abs(output[k] - 0.5) > 0.35)
            output[k][tmp_index[0]] = 0.5

            pointx[:, k] = np.round(
                points[:, k] - 0.5 * patchw) + output[k][:, 0] * patchw
            pointy[:, k] = np.round(
                points[:, k + 5] - 0.5 * patchw) + output[k][:, 1] * patchw

        points = np.hstack([pointx, pointy])
        points = points.astype(np.int32)

        return total_boxes.tolist(), points.tolist()

    def list2colmatrix(self, pts_list):
        """
            convert list to column matrix
        Parameters:
        ----------
            pts_list:
                input list
        Retures:
        -------
            colMat:

        """
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        colMat = np.matrix(colMat).transpose()
        return colMat

    def find_tfrom_between_shapes(self, from_shape, to_shape):
        """
            find transform between shapes
        Parameters:
        ----------
            from_shape:
            to_shape:
        Retures:
        -------
            tran_m:
            tran_b:
        """
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(from_shape.shape[0] / 2, 2)
        to_shape_points = to_shape.reshape(to_shape.shape[0] / 2, 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() -
                    mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def extract_image_chips(self, img, points, desired_size=256, _padding=0):
        """
            crop and align face
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
            desired_size: default 256
            _padding: default 0
        Retures:
        -------
            crop_imgs: list, n
                cropped and aligned faces
        """
        crop_imgs = []
        for p in points:
            shape = []
            for k in range(len(p) / 2):
                shape.append(p[k])
                shape.append(p[k + 5])

            if _padding > 0:
                _padding = _padding
            else:
                _padding = 0
            # average positions of face points
            mean_face_shape_x = [0.224152, 0.75610125,
                                 0.490127, 0.254149, 0.726104]
            mean_face_shape_y = [0.2119465, 0.2119465,
                                 0.628106, 0.780233, 0.780233]

            from_points = []
            to_points = []

            for i in range(len(shape) / 2):
                x = (_padding + mean_face_shape_x[i]) / \
                    (2 * _padding + 1) * desired_size
                y = (_padding + mean_face_shape_y[i]) / \
                    (2 * _padding + 1) * desired_size
                to_points.append([x, y])
                from_points.append([shape[2 * i], shape[2 * i + 1]])

            # convert the points to Mat
            from_mat = self.list2colmatrix(from_points)
            to_mat = self.list2colmatrix(to_points)

            # compute the similar transfrom
            tran_m, tran_b = self.find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
            angle = 180.0 / math.pi * \
                math.atan2(probe_vec[1, 0], probe_vec[0, 0])

            from_center = [(shape[0] + shape[2]) / 2.0,
                           (shape[1] + shape[3]) / 2.0]
            to_center = [0, 0]
            to_center[1] = desired_size * 0.4
            to_center[0] = desired_size * 0.5

            ex = to_center[0] - from_center[0]
            ey = to_center[1] - from_center[1]

            rot_mat = cv2.getRotationMatrix2D(
                (from_center[0], from_center[1]), -1 * angle, scale)
            rot_mat[0][2] += ex
            rot_mat[1][2] += ey

            chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
            crop_imgs.append(chips)

        return crop_imgs
