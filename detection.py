import numpy as np
import cv2, tqdm
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')

import keras.backend as K
K.set_image_data_format('channels_last')

from PIL import Image
# retinanet
from libs.keras_retinanet.keras_retinanet.bin.train import *
from libs.keras_retinanet.keras_retinanet import models
from libs.keras_retinanet.keras_retinanet.utils.image import preprocess_image, resize_image
from libs.keras_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from libs.keras_retinanet.keras_retinanet.utils.gpu import setup_gpu
from libs.yolo3.train import yolo_det
from libs.keras_retinanet.keras_retinanet.utils.colors import label_color
from libs.EfficientDet.train import efficientdet_train, efficientdet_parse_args
from libs.segmentation import Segmentation
import matplotlib.pyplot as plt


class Detection(object):
    def __init__(self, img_dir, label_csv_path, split_rate=0.2, batch_size=32,
                 resized_shape=(480, 720), base='detection',yolo=False):
        """

        :param img_dir:
        :param label_csv_path:
        :param split_rate:
        :param batch_size:
        :param resized_shape:
        :param base:
        """
        self.img_dir = img_dir
        self.resized_shape = resized_shape
        self.label_csv_path = label_csv_path
        self.base = base
        self.yolo=yolo
        if base.lower() == 'detection':
            if not yolo:
                self.split_rate = split_rate
                self.batch_size = batch_size
                self.train_path, self.val_path, self.cls_path = self.process()
                self.train_steps = len(open(self.train_path, 'r', encoding='utf-8').readlines()) // batch_size
                self.val_steps = len(open(self.val_path, 'r', encoding='utf-8').readlines()) // batch_size
            else:
                self.app=yolo_det()
        elif base.lower() == 'segmentation':
            self.app = Segmentation(img_dir=self.img_dir,
                                    label_csv_path=self.label_csv_path,
                                    batch_size=self.batch_size, split_rate=self.split_rate,
                                    shrink=True, use_aug=False)

    def process(self):
        """

        :return:
        """
        shape = self.resized_shape
        os.makedirs('labels/detection/', exist_ok=True)
        os.makedirs('saved_models/detection', exist_ok=True)
        if os.path.exists(
                'labels/detection/anno_detection_train_{}x{}.csv'.format(shape[0], shape[1])) and os.path.exists(
            'labels/detection/anno_detection_val_{}x{}.csv'.format(shape[0], shape[1])) and os.path.exists(
            'labels/detection/detection_class.csv'):
            return 'labels/detection/anno_detection_train_{}x{}.csv'.format(shape[0], shape[
                1]), 'labels/detection/anno_detection_val_{}x{}.csv'.format(shape[0], shape[
                1]), 'labels/detection/detection_class.csv'
        else:
            f = open(self.label_csv_path, 'r', encoding='utf-8').readlines()
            train_writer = open('labels/detection/anno_detection_train_{}x{}.csv'.format(shape[0], shape[1]), 'w',
                                encoding='utf-8')
            val_writer = open('labels/detection/anno_detection_val_{}x{}.csv'.format(shape[0], shape[1]), 'w',
                              encoding='utf-8')
            class_writer = open('labels/detection/detection_class.csv', 'w', encoding='utf-8')
            all_index = np.array(range(len(f)))
            val_index = np.random.choice(all_index, size=int(self.split_rate * len(f)), replace=False)
            for idx, line in tqdm.tqdm(enumerate(f), total=len(f)):
                line = line.strip('\n').split(',')
                img_path = os.path.join(self.img_dir, line[0])
                img = None
                try:
                    img = Image.open(img_path)
                except Exception as E:
                    print(E)
                if img is None:
                    continue
                original_w, original_h = img.size
                h_rate = shape[0] / original_h
                w_rate = shape[1] / original_w
                line[1], line[3] = int(int(line[1]) * w_rate), int(int(line[3]) * w_rate)
                line[2], line[4] = int(int(line[2]) * h_rate), int(int(line[4]) * h_rate)
                if idx not in val_index:
                    train_writer.write('{},{},{},{},{},{}\n'.format(
                        img_path, line[1], line[2], line[3], line[4], 'face'
                    ))
                else:
                    val_writer.write('{},{},{},{},{},{}\n'.format(
                        img_path, line[1], line[2], line[3], line[4], 'face'))
            class_writer.write('face,0\n')
            class_writer.close()
            val_writer.close()
            train_writer.close()
            return 'labels/detection/anno_detection_train_{}x{}.csv'.format(shape[0], shape[
                1]), 'labels/detection/anno_detection_val_{}x{}.csv'.format(shape[0], shape[
                1]), 'labels/detection/detection_class.csv'

    def train_model(self, gpu=3, directly_train=True, backbone='resnet152', method='retinanet',
                    model_path=None, augmentation=True, gpu_fraction=0.3):
        """

        :param gpu:
        :param directly_train:
        :param backbone:
        :param method:
        :param model_path:
        :param augmentation:
        :param gpu_fraction:
        :return:
        """

        setup_gpu(gpu)
        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        session = tf.Session(config=config)
        KTF.set_session(session)
        if method.lower() == 'retinanet' and directly_train and self.base.lower() == 'detection' and not self.yolo:
            args = parse_args()
            args.dataset_type = 'csv'
            args.batch_size = self.batch_size
            args.backbone = backbone
            args.gpu = gpu
            args.annotations = self.train_path
            args.classes = self.cls_path
            args.val_annotations = self.val_path
            args.evaluation = False
            args.snapshot_path = 'saved_models/detection'
            args.random_transform = True
            args.lr = 1e-3
            args.no_resize = True
            args.augmentation = augmentation
            args.width = self.resized_shape[1]
            args.height = self.resized_shape[0]
            args.compute_val_loss = True
            if model_path is None:
                # args.lr = 1e-3
                # args.freeze_backbone = True
                # args.snapshot = None
                # args.weights = None
                # args.imagenet_weights = True
                # args.epochs = 50
                # detection_main(args, train_steps=self.train_steps, val_steps=self.val_steps)

                args.lr = 1e-4
                args.freeze_backbone = False
                args.epochs = 100
                args.snapshot = '{dir}/retinanet_{backbone}_{dataset_type}_{height}x{width}.h5'.format(
                    dir=args.snapshot_path,
                    backbone=args.backbone,
                    dataset_type=args.dataset_type,
                    height=args.height, width=args.width)
                detection_main(args, train_steps=self.train_steps, val_steps=self.val_steps)
            else:
                args.lr = 1e-5
                args.epochs = 100
                args.freeze_backbone = False
                args.snapshot = model_path
                detection_main(args, train_steps=self.train_steps, val_steps=self.val_steps)

        elif method.lower() == 'efficientdet' and directly_train and self.base.lower() == 'detection' and not self.yolo:

            args = efficientdet_parse_args()
            args.dataset_type = 'csv'
            args.batch_size = self.batch_size
            args.backbone = backbone
            args.gpu = gpu
            args.annotations_path = self.train_path
            args.classes_path = self.cls_path
            args.val_annotations_path = self.val_path
            args.evaluation = False
            args.snapshot_path = 'saved_models/detection'
            args.detect_quadrangle = False
            args.detect_text = False
            args.no_resize = True
            args.augmentation = augmentation
            args.width = self.resized_shape[1]
            args.height = self.resized_shape[0]
            args.compute_val_loss = True
            args.random_transform = True
            args.phi = int(backbone[-1])

            if model_path is None:
                # step-1:
                args.lr = 1e-3
                args.freeze_backbone = True
                args.snapshot = 'imagenet'
                args.epochs = 50
                efficientdet_train(args, train_steps=self.train_steps, val_steps=self.val_steps)

                # step-2
                args.lr = 1e-4
                args.freeze_bn = True
                args.epochs = 100
                args.snapshot = '{dir}/efficientdet_{backbone}_{dataset_type}_{height}x{width}.h5'.format(
                    dir=args.snapshot_path,
                    backbone=args.backbone,
                    dataset_type=args.dataset_type,
                    height=args.height, width=args.width)
                efficientdet_train(args, train_steps=self.train_steps, val_steps=self.val_steps)

            else:
                args.lr = 1e-5
                args.freeze_bn = True
                args.epochs = 100
                args.snapshot = model_path
                efficientdet_train(args, train_steps=self.train_steps, val_steps=self.val_steps)

        elif self.base.lower() == 'segmentation':
            fine_tune = True if model_path else False
            self.app.train(model_name=method, backbone=backbone,
                           fine_tune=fine_tune, model_path=model_path,
                           opt='adam', lr=1e-3, shape=self.resized_shape)

        elif self.yolo:
            self.app.train_yolov3(gpu_id=1)

    def prediction(self,
                   gpu_id=0,
                   directly_train=True,
                   backbone='resnet50',
                   method='retinanet',
                   resized=True,
                   preprocess=True,
                   test_dir='data/personai_icartoonface_detval',
                   model_path='/data/shuai_li/FaceTask/saved_models/detection/resnet50_csv_16.h5',
                   show=False, write_prediction=True, gpu_fraction=0.2,test_on_val=False):
        """

        :param gpu_id:
        :param directly_train:
        :param backbone:
        :param method:
        :param resized:
        :param preprocess:
        :param test_dir:
        :param model_path:
        :param show:
        :param write_prediction:
        :param gpu_fraction:
        :param test_on_val:
        :return:
        """
        name = model_path.split('/')[-1].replace('.h5', '')
        full_name = [method,name]
        if directly_train:
            full_name.append('directly-train')
        if resized:
            full_name.append('resized-by-myself')
        if preprocess:
            full_name.append('preprocessed')
        if test_on_val:
            full_name.append('val-for-mAP-compute')

        setup_gpu(gpu_id)
        import tensorflow as tf
        import keras.backend.tensorflow_backend as KTF
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        session = tf.Session(config=config)
        KTF.set_session(session)
        res_name = 'predictions/{}.csv'.format('_'.join(full_name))
        if directly_train and method.lower() == 'retinanet' and self.base.lower() == 'detection':
            if write_prediction:
                result = open('predictions/{}.csv'.format('_'.join(full_name)), 'w', encoding='utf-8')
            model = models.load_model(model_path, backbone_name=backbone)
            model = models.convert_model(model, nms=True, class_specific_filter=True)
            model.summary()
            label_to_name = {0: 'face'}
            files = os.listdir(test_dir)
            for img_name in tqdm.tqdm(files, total=len(files)):
                img_path = os.path.join(test_dir, img_name)
                image = cv2.imread(img_path)
                original_shape = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                draw = image.copy()
                if resized:
                    image = cv2.resize(image, (720, 480), interpolation=cv2.INTER_AREA)
                if preprocess:
                    image = preprocess_image(image)
                scale = 1
                if not resized:
                    image, scale = resize_image(image)
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
                if not resized:
                    boxes /= scale
                if resized:
                    h_rate = original_shape[0] / 480
                    w_rate = original_shape[1] / 720
                    boxes[:, 0] *= w_rate
                    boxes[:, 2] *= w_rate
                    boxes[:, 1] *= h_rate
                    boxes[:, 3] *= h_rate
                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    if score < 0.5:
                        break
                    color = label_color(label)
                    b = box.astype(int)
                    if write_prediction:
                        result.write('{},{},{},{},{},{},{}\n'.format(
                            img_name, b[0], b[1], b[2], b[3], 'face', score
                        ))
                    if show:
                        draw_box(draw, b, color=color)
                        caption = "{} {:.3f}".format(label_to_name[label], score)
                        draw_caption(draw, b, caption)
                if show:
                    plt.figure(figsize=(15, 15))
                    plt.axis('off')
                    plt.imshow(draw)
                    plt.show()
        elif self.base.lower() == 'segmentation':
            res_name = self.app.predict(
                model_path=model_path,
                test_img_dir=test_dir,
                shape=self.resized_shape,
            )
        elif directly_train and method.lower() == 'efficientdet' and self.base.lower() == 'detection':
            from libs.EfficientDet.model import efficientdet
            from libs.EfficientDet.utils import preprocess_image as pre_process, postprocess_boxes
            from libs.EfficientDet.utils.draw_boxes import draw_boxes
            if write_prediction:
                result = open('predictions/{}.csv'.format('_'.join(full_name)), 'w', encoding='utf-8')
            phi = int(backbone[-1])
            image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
            image_size = image_sizes[phi]
            classes = {0: 'face'}
            score_threshold = 0.2
            _, model = efficientdet(phi=phi,
                                    weighted_bifpn=False,
                                    num_classes=1,
                                    score_threshold=score_threshold)
            model.load_weights(model_path, by_name=True)
            for img_name in tqdm.tqdm(os.listdir(test_dir), total=len(os.listdir(test_dir))):
                img_path = os.path.join(test_dir, img_name)
                image = cv2.imread(img_path)
                image = image[:, :, ::-1]
                src_image = image.copy()
                h, w = image.shape[:2]
                if resized:
                    h_rate = h / self.resized_shape[0]
                    w_rate = w / self.resized_shape[1]
                    image = cv2.resize(image, (self.resized_shape[1], self.resized_shape[0]),
                                       interpolation=cv2.INTER_AREA)
                    image, scale = pre_process(image, image_size)
                    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
                    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
                    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
                    boxes[:, 0] *= w_rate
                    boxes[:, 2] *= w_rate
                    boxes[:, 1] *= h_rate
                    boxes[:, 3] *= h_rate
                    boxes = boxes.astype(dtype='int8')
                    indices = np.where(scores[:] > score_threshold)[0]
                    boxes = boxes[indices]
                    labels = labels[indices]
                    if write_prediction:
                        for b, l, s in zip(boxes, labels, scores):
                            xmin, ymin, xmax, ymax = list(map(int, b))
                            result.write('{},{},{},{},{},{},{}\n'.format(
                                img_name, xmin, ymin, xmax, ymax, 'face', s
                            ))
                    if show:
                        draw_boxes(src_image, boxes, scores, labels, [[255, 0, 0]], classes)
                        plt.figure(figsize=(15, 15))
                        plt.axis('off')
                        plt.imshow(src_image)
                        plt.show()

                else:
                    image, scale = pre_process(image, image_size)
                    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
                    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
                    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
                    indices = np.where(scores[:] > score_threshold)[0]
                    boxes = boxes[indices]
                    labels = labels[indices]
                    if write_prediction:
                        for b, l, s in zip(boxes, labels, scores):
                            xmin, ymin, xmax, ymax = list(map(int, b))
                            result.write('{},{},{},{},{},{},{}\n'.format(
                                img_name, xmin, ymin, xmax, ymax, 'face', s
                            ))
                    if show:
                        draw_boxes(src_image, boxes, scores, labels, [[255, 0, 0]], classes)
                        plt.figure(figsize=(15, 15))
                        plt.axis('off')
                        plt.imshow(src_image)
                        plt.show()

        return res_name

    def compute_mAP_on_val_data(self,
                                gpu_id=0, directly_train=True,
                                backbone='resnet50',method='retinanet',
                                resized=True,preprocess=True,
                                model_path='/data/shuai_li/FaceTask/saved_models/detection/resnet50_csv_16.h5',
                                show=False, write_prediction=True, gpu_fraction=0.2):
        pass


if __name__ == '__main__':
    # (240,360) (480,720) (512,768) (720,1080) multi-scale training
    app = Detection(img_dir='/data/shuai_li/FaceTask/data/personai_icartoonface_dettrain/icartoonface_dettrain',
                    label_csv_path='/data/shuai_li/FaceTask/data/personai_icartoonface_dettrain_anno_updatedv1.0.csv',
                    batch_size=10, resized_shape=(240, 360), base='detection')
    # app.train_model(gpu=3, directly_train=True, method='retinanet', backbone='resnet152',
    #                 model_path=None, augmentation=True, gpu_fraction=0.1)  # the last ms, use augmentation
    app.prediction(preprocess=True,resized=False,show=False,write_prediction=True,backbone='b0',method='efficientdet',gpu_id=0,
                   gpu_fraction=0.2,
                   model_path='/data/shuai_li/FaceTask/saved_models/detection/efficientdet_b0_csv_240x360_0011_0.62161.h5')
