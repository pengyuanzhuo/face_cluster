# -*- coding: utf-8 -*-
# 多线程调用线上fecex-detect服务
# 默认线程10

ak = "4h1yz7uIQf58BSh5vKaiTnFYw5Hl6a4aWmNjTaYH"
sk = "v-ut4LKmaPvEjawEOUbcfwcUiP0j-MfnLRnC2in9"

import json
import requests
from face_det.ava_auth import AuthFactory
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

def token_gen(ak,sk):
    factory = AuthFactory(ak,sk)
    fauth = factory.get_qiniu_auth
    token = fauth()
    return token

def url_gen(json_file):
    urls = []
    with open(json_file, 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            url = line['url']
            urls.append(url)
    return urls

def request(img_url):
    # res = {'url':img_url,'det':[]}
    res = None
    request_url = 'http://argus.atlab.ai/v1/face/detect'
    headers = {"Content-Type": "application/json"}
    body = json.dumps({"data": {"uri": img_url}})
    token = token_gen(ak,sk)
    try:
        r = requests.post(request_url, data=body,timeout=15, headers=headers, auth=token)
    except Exception as e:
        raise e
        # print('http error.')
    else:
        if r.status_code == 200:
            r = r.json()
            if r['code'] == 0 and r['result']['detections'] is not None:
                #res['det'] = r['result']['detections']
                res = r['result']['detections']
            else:
                # raise Exception("No face")
                res = []
        else:
            raise Exception('http err --> %d'%r.status_code)
    return res

def face_det(img_urls, log, num_thread=10):
    """
    multithread face detect
    Args:
    -----
    img_urls : list of url
    log : face det log json file
    num_thread : num thread

    Return:
    ------
    res_list : list of result
        [{'url':'xxx',
          'det':[{},{},...]}]
    """
    with open(log,'w') as f_log:
        with ThreadPoolExecutor(max_workers=num_thread) as exe:
            future_tasks = {exe.submit(request, url): url for url in img_urls}
            all_url = len(future_tasks)
            count = 1
            for task in as_completed(future_tasks):
                if task.done():
                    url = future_tasks[task]
                    print('fece det %d/%d'%(count,all_url))
                    count += 1
                    try:
                        det = task.result()
                    except Exception as e:
                        print('%s ---> %s'%(url, e))
                        det = []
                    res = {}
                    res['url'] = url
                    res['det'] = det
                    f_log.write(json.dumps(res))
                    f_log.write('\n')

if __name__ == "__main__":
    urls = _url_gen('data/data.json')
    face_det(urls, './facexxxxxxdet.json')
