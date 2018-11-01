# facex-lib
人脸聚类工具包含以下模块
1. 人脸检测(线上api)
2. 人脸对齐(mtcnn)
3. 人脸特征提取
4. 人脸聚类

## 1. 人脸检测
- 输入: url file, 每行一个url, 例如
```
http://pd6q73oht.bkt.clouddn.com/1.jpg
http://pd6q73oht.bkt.clouddn.com/2.jpg
http://pd6q73oht.bkt.clouddn.com/3.jpg
http://pd6q73oht.bkt.clouddn.com/4.jpg
http://pd6q73oht.bkt.clouddn.com/5.jpg
http://pd6q73oht.bkt.clouddn.com/6.jpg
http://pd6q73oht.bkt.clouddn.com/7.jpg
http://pd6q73oht.bkt.clouddn.com/8.jpg
```
- 输出: json file, 每行一个json, 格式为
```
{
    "url":"http://pd6q73oht.bkt.clouddn.com/2.jpg", # 原图url
    "det":[
        {
            "boundingBox":{        # 检测到的人脸
                "score":0.9999335,
                "pts":[
                    [
                        362,
                        105
                    ],
                    [
                        495,
                        105
                    ],
                    [
                        495,
                        308
                    ],
                    [
                        362,
                        308
                    ]
                ]
            },
            ...
        }
    ]
}
检测结果是一个list, 若未检出人脸, 则list为[]
```

## 2. 人脸 crop & align
这一步会将裁减后的人脸上传到kodo, 用于可视化
- 输入: 人脸检测结果json file, 见1
- 输出: 人脸crop和align结果 json file, 每行一个json, 格式为
```
{
    "url":"http://pd6q73oht.bkt.clouddn.com/2.jpg", # 原图url
    "det":[
        {
            "boundingBox":{
                "score":0.9999335,
                "pts":[
                    [
                        362,
                        105
                    ],
                    [
                        495,
                        105
                    ],
                    [
                        495,
                        308
                    ],
                    [
                        362,
                        308
                    ]
                ]
            }
        }
    ]
    "aligned":["/workspace/facex-lib/aligned_face/8f420fee-9f65-11e8-8bbf-0242ac110005.jpg",], # 对齐人脸存放路径, 用于特征提取
    "aligned_url":["http://bucket/xxx.jpg"]  # 人脸上传到kodo的url, 该list与aligned list 一一对应(用于结果展示)
}
```
每张原始图片可能会检出多张人脸, 因此aligned字段是一个list. 另外, align 和 crop过程会将对齐后的人脸cache到本地, 故aligned字段保存的是路径.

## 3. 人脸特征提取
- 输入: 人脸align 和 crop 结果json file, 格式见2输出
- 输出: 人脸特征提取结果json file. 每行一个json, 格式为:
```
{
    "feat":"/workspace/facex-lib/features/92411f6e-9f65-11e8-8bbf-0242ac110005.npy",       # 人脸特征文件
    "aligned":"/workspace/facex-lib/aligned_face/92411f6e-9f65-11e8-8bbf-0242ac110005.jpg" # 人脸图片路径
    "face_url":"http://bucket/xxx.jpg"  # 人脸对应的kodo url
}
```
人脸特征文件保存在本地, 故feat指向的是特征文件地址

## 4. 人脸聚类
- 输入: 人脸特征提取结果 json file, 见3输出
- 输出: 人脸聚类结果, 格式为
```
[
    {
        "cluster_id": "1",  # 聚类编号, -1 表示未成功聚类
        "img_lst": [
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/90317368-9f65-11e8-8bbf-0242ac110005.jpg"
            }, 
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/9103ed16-9f65-11e8-8bbf-0242ac110005.jpg"
            }, 
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/9164b11e-9f65-11e8-8bbf-0242ac110005.jpg"
            }, 
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/92406f88-9f65-11e8-8bbf-0242ac110005.jpg"
            }
        ]
    }, 
    {
        "cluster_id": "2", 
        "img_lst": [
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/8f420fee-9f65-11e8-8bbf-0242ac110005.jpg"
            }, 
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/91d2e08a-9f65-11e8-8bbf-0242ac110005.jpg"
            }, 
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/90a52fc4-9f65-11e8-8bbf-0242ac110005.jpg"
            }, 
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/8fd9dd42-9f65-11e8-8bbf-0242ac110005.jpg"
            }, 
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/9032864a-9f65-11e8-8bbf-0242ac110005.jpg"
            }
        ]
    }, 
    {
        "cluster_id": "-1", 
        "img_lst": [
            {
                "url": "http://pd6q73oht.bkt.clouddn.com/92411f6e-9f65-11e8-8bbf-0242ac110005.jpg"
            }
        ]
    }
]
```
