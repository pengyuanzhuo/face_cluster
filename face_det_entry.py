# -*- coding: utf-8 -*-

from face_det_module.face_det import face_det

ak = "4h1yz7uIQf58BSh5vKaiTnFYw5Hl6a4aWmNjTaYH" 
sk = "v-ut4LKmaPvEjawEOUbcfwcUiP0j-MfnLRnC2in9"

def urls(urlfile):
    url_list = []
    with open(urlfile, 'r') as f:
        for line in f:
            url = line.strip()
            url_list.append(url)
    return url_list

if __name__ == "__main__":
    img_urls = urls('./input.json')
    face_det(img_urls, 'tmp.json', ak, sk)
