"""
Retrain the YOLO model for your own dataset.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import keras.backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


class yolo_det(object):
    def __init__(self,
                 all_file_path='/data/shuai_li/Thai-OCR-Dev/Models/all_index/all_file.npy',
                 ):
        self.all_file = np.load(all_file_path)
        self.bad_cases = [
            '/data/TH_ID/yezi_image/fd87e860dd6406b83a6306def091330e07000000000f892600000000020200a4.jpeg',
            '/data/TH_ID/yezi_image/ab100cd6a06a1925089d0a726e2886280700000000225b2600000000020200e1.jpeg',
            '/data/TH_ID/yezi_image/beb05d687a56ed4d3fb6d42347c003070700000000265c290000000002020035.jpeg',
            '/data/TH_ID/yezi_image/bea43cfb264711b81fe0f49a358294e2070000000023857200000000020200bd.jpeg',
            '/data/TH_ID/yezi_image/bcc900f84ed38e831cef76fc2bec6af707000000002631360000000002020033.jpeg',
            '/data/TH_ID/yezi_image/beda194099d0e3a5265b8c98406f21600700000000456bd500000000020200e2.jpeg',
            '/data/TH_ID/yezi_image/c27e1f836801414b07def809c78de13307000000003a1db2000000000202004a.jpeg',
            '/data/TH_ID/yezi_image/be89e4e6ccfbce649201f60d40a1515007000000001e47a000000000020200f5.jpeg',
            '/data/TH_ID/yezi_image/c6000ce57f232eacba3553a0dec2aacd0700000000663c3400000000020200b4.jpeg',
            '/data/TH_ID/yezi_image/909fe765a7e5f00e0e9d26e386cb54ae07000000003da7660000000002020000.jpeg',
            '/data/TH_ID/yezi_image/b549617924ae64a20bb5eb67bc2327050700000000371a460000000002020060.jpeg',
            '/data/TH_ID/yezi_image/0634f63236f44d9304b077e83a2bc27307000000004794dc0000000002020042.jpeg',
            '/data/TH_ID/yezi_image/c5c146e1bfc0c9b32f3aa8976cf29b5107000000000e55260000000002020089.jpeg',
            '/data/TH_ID/yezi_image/d5c81669ec7b814eb251a1404581742607000000003e908300000000020200af.jpeg',
            '/data/TH_ID/yezi_image/beefe0586919c7aa03ac76cc3ed94a2607000000005305e800000000020200f8.jpeg',
            '/data/TH_ID/yezi_image/bc1bbb89472d3b380558159c87431f7507000000002498e900000000020200ff.jpeg',
            '/data/TH_ID/yezi_image/c7ec8fec506ff0386c6e598542f5aaef070000000042aae900000000020200be.jpeg',
            '/data/TH_ID/yezi_image/1995c377464e5e5df62f372487ac125c0700000000271f3c00000000020200b4.jpeg',
            '/data/TH_ID/yezi_image/8a0ce2aee94518bbe6360c6db5b1e8f607000000004f673b0000000002020051.jpeg',
            '/data/TH_ID/yezi_image/e6918b940c1b9c2ebdd1dfee0a885aad07000000002e62cb000000000202004e.jpeg',
            '/data/TH_ID/yezi_image/be7292a0ac02dafda11a32189297a2b10700000000420a3c00000000020200a8.jpeg',
            '/data/TH_ID/yezi_image/a440e562e074be208ee853f4b0d53f7f070000000028ed770000000002020013.jpeg',
        ]
        self.anno_path=self.process()

    def process(self):
        if os.path.exists('model_data/annotations.txt'):
            return 'model_data/annotations.txt'
        f=open('model_data/annotations.txt','w',encoding='utf-8')
        for line in self.all_file:
            img_path = line.split('\t')[0]
            if img_path.split('/')[2]!='TH_ID':continue
            if img_path in self.bad_cases:continue
            coords = np.array(line.split('\t')[-1].strip('\n').strip('\t').split(' '),dtype=np.int).reshape((-1,2))
            xmin,ymin,xmax,ymax=np.min(coords[:4,0]),np.min(coords[:4,1]),np.max(coords[:4,0]),np.max(coords[:4,1])
            f.write('{} {},{},{},{},{}\n'.format(
                img_path,xmin,ymin,xmax,ymax,0
            ))
        f.close()
        return 'model_data/annotations.txt'

    def train_yolov3(self):
        classes_path = 'model_data/classes.txt'
        anchors_path = 'model_data/yolo_anchors.txt'
        class_names = get_classes(classes_path)
        num_classes = len(class_names)
        anchors = get_anchors(anchors_path)
        input_shape = (416,416) # multiple of 32, hw
        is_tiny_version = len(anchors)==6 # default setting
        if is_tiny_version:
            model = create_tiny_model(input_shape, anchors, num_classes,
                                      freeze_body=2, weights_path='model_data/yolov3_trained_for_th_card_detection.h5')
        else:
            model = create_model(input_shape, anchors, num_classes,
                                 freeze_body=2, weights_path='model_data/yolov3_trained_for_th_card_detection.h5') # make sure you know what you freeze

        logging = TensorBoard(log_dir='logs')
        checkpoint = ModelCheckpoint('model_data/yolov3_trained_for_th_card_detection.h5',
                                     monitor='val_loss', save_weights_only=False, save_best_only=True, period=1,
                                     verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)
        val_split = 0.1
        if os.path.exists('model_data/train.npy') and os.path.exists('model_data/val.npy'):
            train_data=np.load('model_data/train.npy')
            val_data=np.load('model_data/val.npy')
        else:
            with open(self.anno_path) as f:
                lines = f.readlines()
            np.random.seed(10101)
            np.random.shuffle(lines)
            np.random.seed(None)
            num_val = int(len(lines)*val_split)
            num_train = len(lines) - num_val
            train_data=np.array(lines[:num_train])
            val_data=np.array(lines[num_train:])
            np.save('model_data/train.npy',train_data)
            np.save('model_data/val.npy',val_data)

        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Stage 1: Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_data), len(val_data), batch_size))
        model.fit_generator(data_generator_wrapper(train_data, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, len(train_data)//batch_size),
                            validation_data=data_generator_wrapper(val_data, batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, len(val_data)//batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint,reduce_lr,early_stopping])

        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change

        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Stage 2: Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_data), len(val_data), batch_size))
        model.fit_generator(data_generator_wrapper(train_data, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, len(train_data)//batch_size),
                            validation_data=data_generator_wrapper(val_data, batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, len(val_data)//batch_size),
                            epochs=300,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])




if __name__ == '__main__':
    app=yolo_det()
    app.train_yolov3()

