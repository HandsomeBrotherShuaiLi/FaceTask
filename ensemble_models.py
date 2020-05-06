'''
@author: Shuai Li
@license: (C) Copyright 2015-2025, Shuai Li.
@contact: li.shuai@wustl.edu
@IDE: pycharm
@file: ensemble_models.py
@time: 3/5/20 23:25
@desc:
'''
import numpy as np
import os,cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm
from PIL import Image,ImageDraw


def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep


def ensemble_models(img_dir,prediction_files,
                    threshold=0.5,show=True,score_threshold=None):
    res = defaultdict(list)
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(len(prediction_files)+1)]
    for idx,f in enumerate(prediction_files):
        data = open(f,'r',encoding='utf-8').readlines()
        for line in data:
            line = line.strip('\n').split(',')
            res[line[0]].append([line[1],line[2],line[3],line[4],line[-1],idx])
    f=open('predictions/ensemble_models_{}_{}_{}.csv'.format(len(prediction_files),threshold,score_threshold or 'no-score-threshold'),'w',encoding='utf8')
    for image_name in tqdm.tqdm(res.keys(),total=len(res.keys())):
        data = np.array(res[image_name],dtype='float32')
        boxes = data[:,:5]
        idxs = py_nms(boxes,thresh=threshold)
        boxes = np.array(boxes[idxs,:4],dtype=np.int)
        new_idxs = []
        for idx in idxs:
            if score_threshold is None:
                f.write('{},{},{},{},{},{},{}\n'.format(
                    image_name,int(data[idx,0]),int(data[idx,1]),int(data[idx,2]),int(data[idx,3]),'face',data[idx,4]
                ))
            else:
                if data[idx,4] >= score_threshold:
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        image_name,int(data[idx,0]),int(data[idx,1]),int(data[idx,2]),int(data[idx,3]),'face',data[idx,4]
                    ))
                    new_idxs.append(idx)
        if score_threshold:
            boxes = np.array(data[new_idxs,:4],dtype=np.int)
        if show:
            image = cv2.imread(os.path.join(img_dir,image_name))
            image = image[:,:,::-1]
            image = Image.fromarray(image)
            im_c = image.copy()
            draw = ImageDraw.Draw(image)
            draw_copy = ImageDraw.Draw(im_c)
            for i in range(len(data)):
                box=data[i,:]
                draw.rectangle((int(box[0]),int(box[1]),int(box[2]),int(box[3])),outline=tuple(colors[int(box[-1])]),width=2)
                draw.text((int(box[0]),int(box[1])+5),'{}_{}'.format(box[-1],box[-2]),fill='black')
            # display(image)
            for i in range(len(boxes)):
                box=boxes[i,:]
                draw_copy.rectangle((int(box[0]),int(box[1]),int(box[2]),int(box[3])),outline=tuple(colors[-1]),width=2)
                draw_copy.text((int(box[0]),int(box[1])+5),'{}_{}'.format(data[idxs[i],-1],data[idxs[i],-2]),fill='black')
            # display(im_c)

ensemble_models(img_dir='data/personai_icartoonface_detval',
                prediction_files=[
                    'predictions/retinanet_directly-train_preprocessed.csv',
                    'predictions/efficientdet_efficientdet_b0_csv_240x360_0007_0.60522_predictions.csv_directly-train_preprocessed.csv',
                    'predictions/efficientdet_efficientdet_b0_csv_240x360_0006_0.63139_directly-train_preprocessed.csv',
                    'predictions/efficientdet_efficientdet_b0_csv_240x360_0011_0.62161_directly-train_preprocessed.csv',
                    'predictions/efficientdet_efficientdet_b0_csv_240x360_0005_0.64351_predictions.csv_directly-train_preprocessed.csv'
                ],threshold=0.001,show=False,score_threshold=0.4)
