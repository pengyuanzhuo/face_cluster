# -*- coding: utf-8 -*-
from mtcnn_aligner.align import MtcnnAligner
from mtcnn_aligner import align

def _parse_det(det_list,threshold=0.5):
    """
    parse det result from det log
    Args:
    -----
    det_list : det from log
    """
    bboxes = []
    for face in det_list:
        pts = face['boundingBox']['pts']
        score = face['boundingBox']['score']
        if score >= threshold:
            bboxes.append(pts)
    return bboxes

def _convert_4p_to_xywh(bbox, scale=1.5):
    """
    convert 4 points to xywh
    Args:
    ----
    bbox : 4 points bbox
        [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    scale : scale of face rect
    Return:
    ------
    xywh : xywh bbox
        [x,y,w,h]
    """
    x,y = bbox[0]
    x3,y3 = bbox[2]
    w = x3 - x + 1
    h = y3 - y + 1

    x = int(x - w*(scale - 1) / 2.0)
    y = int(y - w*(scale - 1) / 2.0)
    w = x + int(w * scale)
    h = y + int(h * scale)

    return [x,y,w,h]

def align_crop(img_url, aligner, rect_list, output_size=(112,112), scale=1.5,
               save_dir='./aligned_face',
               default_square=True,
               inner_padding_facter=0,
               outer_padding=(0,0),
               gpu_id=-1):
    """
    single image processing
    Args:
    -----
    img_url : image url
    aligner : face aligner object
    rect_list : list of face bbox
        [[x,y,w,h],[x,y,w,h],...]
    output_size : output aligned face image size
    save_dir : dir to save aligned face images

    Return:
    -------
    face_list : list of face image abs path
    """
    print('face align & crop : %s'%img_url)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if rect_list == []:
        return None
    reference_5pts = get_reference_facial_points(output_size,
                                                 inner_padding_facter,
                                                 outer_padding,
                                                 default_square)
    try:
        # img = cv2.imread(img_path)
        res = urllib.urlopen(img_url)
        img = np.asarray(bytearray(res.read()),dtype='uint8')
        img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    except:
        pass
    if img is None:
        return None
    for rect in rect_list:
        _convert_4p_to_xywh(rect,scale=scale)
    bboxes, points = aligner.align_face(img,rect_list)
    face_list = []
    for i in range(len(bboxes)):
        box = bboxes[i]
        pts = points[i]
        facical5points = np.reshape(pts,(2,-1))
        dst_img = warp_and_crop_face(img,
                                facical5points,
                                reference_5pts,
                                output_size)
        dst_name = os.path.join(os.path.abspath(save_dir),str(uuid.uuid1())) + '.jpg'
        face_list.append(dst_name)
        cv2.imwrite(dst_name,dst_img)
    return face_list

def face_align_crop(det_log, model_path, save_dir, align_log, threshold=0.5, gpu_id=-1):
    """
    batch align
    write aligned face img and log to disk 

    Args:
    -----
    det_log : face det log json file
    model_path : face align model path
    save_dir : dir to save aligned face imgs
    align_log : face align and crop log json file
        {'aligned':'/path/alignedimg',
         'feat':'/path/feat',
          'group': num}
    threshold : face threshold

    Return:
    -------
    int
        -2   no face
        -1   invalid input det log
         0   success
    """
    flag = -2
    if not os.path.exists(det_log):
        print('%s is None'%det_log)
        return -1
    aligner = MtcnnAligner(model_path,True,gpu_id)
    with open(det_log,'r') as f_det:
        with open(align_log,'w') as f_align:
            for line in f_det:
                line = json.loads(line.strip())
                img_url = line['url']
                det = line['det']
                rect_list = _parse_det(det,threshold)
                aligned_imgs = align_crop(img_url,aligner,rect_list,save_dir=save_dir)
                if aligned_imgs is None:
                    continue

                # 上传到face-cluster bucket
                print('upload face imgs')
                upload(aligned_imgs,'face-cluster')
                #upload(list(id_urls_dict.values()[0]),'face-cluster')
                print('upload done.')

                for aligned_img in aligned_imgs:
                    flag = 0
                    item = {'aligned':aligned_img,'feat':'-1','group':-1}
                    f_align.write(json.dumps(item))
                    f_align.write('\n')
    return flag