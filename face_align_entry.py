# -*- coding: utf-8 -*-

from face_align_module.face_align import face_align_crop

det_log = './tmp.json'
model_path = './models/align_model'
save_dir = './cache/align'
align_log = './aligned.json'
threshold = 0.5
gpu_id = 0

success = face_align_crop(det_log,
                          model_path,
                          save_dir,
                          align_log,
                          threshold = threshold,
                          gpu_id = gpu_id)
