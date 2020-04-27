import numpy as np
import cv2, tqdm

# retinanet
from libs.keras_retinanet.keras_retinanet.bin.train import *


class Detection(object):
    def __init__(self, img_dir, label_csv_path, split_rate=0.2, batch_size=32):
        self.img_dir = img_dir
        self.label_csv_path = label_csv_path
        self.split_rate = split_rate
        self.batch_size = batch_size
        self.train_path, self.val_path, self.cls_path = self.process()
        self.train_steps = len(open(self.train_path, 'r', encoding='utf-8').readlines()) // batch_size
        self.val_steps = len(open(self.val_path, 'r', encoding='utf-8').readlines()) // batch_size

    def process(self, shape=(480, 720)):
        os.makedirs('labels/detection/', exist_ok=True)
        os.makedirs('saved_models/detection', exist_ok=True)
        if os.path.exists('labels/detection/anno_detection_train.csv') and os.path.exists(
                'labels/detection/anno_detection_val.csv') and os.path.exists('labels/detection/detection_class.csv'):
            return 'labels/detection/anno_detection_train.csv', 'labels/detection/anno_detection_val.csv', 'labels/detection/detection_class.csv'
        else:
            f = open(self.label_csv_path, 'r', encoding='utf-8').readlines()
            train_writer = open('labels/detection/anno_detection_train.csv', 'w', encoding='utf-8')
            val_writer = open('labels/detection/anno_detection_val.csv', 'w', encoding='utf-8')
            class_writer = open('labels/detection/detection_class.csv', 'w', encoding='utf-8')
            all_index = np.array(range(len(f)))
            val_index = np.random.choice(all_index, size=int(self.split_rate * len(f)), replace=False)
            for idx, line in tqdm.tqdm(enumerate(f), total=len(f)):
                line = line.strip('\n').split(',')
                img_path = os.path.join(self.img_dir, line[0])
                img = cv2.imread(img_path)
                original_h, original_w = img.shape[:2]
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
            return 'labels/detection/anno_detection_train.csv', 'labels/detection/anno_detection_val.csv', 'labels/detection/detection_class.csv'

    def train_model(self, directly_train=True, backbone='resnet152', method='retinanet'):
        if method == 'retinanet' and directly_train:
            args = parse_args()
            args.dataset_type = 'csv'
            args.batch_size = self.batch_size
            args.backbone = backbone
            args.gpu = 3
            args.annotations = self.train_path
            args.classes = self.cls_path
            args.val_annotations = self.val_path
            args.snapshot_path = 'saved_models/detection'
            args.random_transform = True
            args.weighted_average = True
            args.no_resize=True
            detection_main(args=args, train_steps=self.train_steps, val_steps=self.val_steps)


if __name__ == '__main__':
    app = Detection(img_dir='/data/shuai_li/FaceTask/data/personai_icartoonface_dettrain/icartoonface_dettrain',
                    label_csv_path='/data/shuai_li/FaceTask/data/personai_icartoonface_dettrain/icartoonface_dettrain.csv',
                    batch_size=10)
    app.train_model(directly_train=True,backbone='resnet50')
