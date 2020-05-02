import os, cv2, tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import keras, numpy as np
from collections import defaultdict
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
KTF.set_session(session)
import segmentation_models as sm
from keras.optimizers import Adam, Adadelta, SGD
import keras.backend as K
import matplotlib.pyplot as plt
K.set_image_data_format('channels_last')

from albumentations import (
    IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,PadIfNeeded,ElasticTransform,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,RandomGamma
)


class DataLoader(object):
    def __init__(self,img_dir,label_csv_path,split_rate=0.2,batch_size=32):
        self.img_dir = img_dir
        self.label_csv_path = label_csv_path
        self.split_rate=split_rate
        self.batch_size=batch_size
        self.train_index,self.val_index,self.data=self.process()
        self.train_steps=len(self.train_index)//self.batch_size
        self.val_steps=len(self.val_index)//self.batch_size

    def process(self):
        data=defaultdict(list)
        f = open(self.label_csv_path,'r',encoding='utf-8').readlines()
        for line in tqdm.tqdm(f,total=len(f)):
            line=line.strip('\n').split(',')
            img_path=os.path.join(self.img_dir,line[0])
            xmin,ymin,xmax,ymax=int(line[1]),int(line[2]),int(line[3]),int(line[4])
            data[img_path].append([xmin,ymin,xmax,ymax])
        total_number=len(data.keys())
        val_number=int(self.split_rate*total_number)
        all_index=np.array(range(total_number))
        val_index=np.random.choice(all_index,size=val_number,replace=False)
        train_index=np.array([i for i in all_index if i not in val_index])
        np.random.shuffle(train_index),np.random.shuffle(val_index)
        print('total number:{} train number:{} val number:{}'.format(
            total_number,len(train_index),len(val_index)
        ))
        return train_index,val_index,data

    def generator(self,is_train=True,shape=(256,256),
                  augmentation=None,coord_augmentation=None,shrink=None):
        """
        don't play with augmentation at the beginning
        :param is_train:
        :param shape:
        :param augmentation:
        :param coord_augmentation:
        :param shrink:
        :return:
        """
        index=self.train_index if is_train else self.val_index
        start=0
        while True:
            inputs = np.zeros(shape=(self.batch_size, shape[0], shape[1], 3))
            labels = np.zeros(shape=(self.batch_size, shape[0],shape[1],1))
            if start + self.batch_size < len(index):
                batch_index = index[start:start + self.batch_size]
            else:
                batch_index = np.hstack((index[start:], index[:(start + self.batch_size) % len(index)]))
            np.random.shuffle(batch_index)
            batch_keys=np.array(list(self.data.keys()))[batch_index]
            for idx,k in enumerate(batch_keys):
                img=cv2.imread(k)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img_shape=img.shape
                resized_img = cv2.resize(img,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
                h,w=img_shape[:2]
                resized_h,resized_w=shape
                h_rate=resized_h/h
                w_rate=resized_w/w
                mask=np.zeros(shape=(shape[0],shape[1]))
                for coord in self.data[k]:
                    xmin,ymin,xmax,ymax = coord
                    xmin=int(xmin*w_rate)
                    xmax=int(xmax*w_rate)
                    ymin=int(ymin*h_rate)
                    ymax=int(ymax*h_rate)
                    if shrink:
                        """
                        segmentation may detect larger area, so I shrink the 
                        mask area to improve the IoU 
                        """
                        face_w=xmax-xmin
                        face_h=ymax-ymin
                        center_y=(ymin+ymax)/2
                        center_x=(xmin+xmax)/2
                        shrink_w=face_w*(1-shrink)
                        shrink_h=face_h*(1-shrink)
                        xmin=int(center_x-shrink_w/2)
                        xmax=int(center_x+shrink_w/2)
                        ymin=int(center_y-shrink_h/2)
                        ymax=int(center_y+shrink_h/2)
                    mask[ymin:ymax,xmin:xmax] = 1
                labels[idx,:,:,:]=np.expand_dims(mask,axis=-1)
                inputs[idx,:,:,:]=resized_img/255.0
            yield inputs,labels
            start = (start + self.batch_size) % len(index)


class Segmentation(object):
    def __init__(self,img_dir='data/personai_icartoonface_dettrain/icartoonface_dettrain',
                 label_csv_path='data/personai_icartoonface_dettrain/icartoonface_dettrain.csv',
                 split_rate=0.2,batch_size=32,
                 shrink=None,
                 use_aug=False):
        self.data_loader = DataLoader(img_dir,label_csv_path,split_rate,batch_size)
        self.shrink=shrink
        self.use_aug=use_aug

    def train(self,model_name='unet', backbone='resnet50',
              fine_tune=False, model_path=None, opt='adam', lr=0.001,
              shape=(256, 256)):
        os.makedirs('saved_models/segmentation',exist_ok=True)
        if fine_tune: lr = lr / 10
        opt_dict = {
            'adam': Adam(lr),
            'sgd': SGD(lr),
            'adadelta': Adadelta(lr)
        }
        if fine_tune and model_path:
            new_name = model_path.strip('.h5') + '_fine-tune_{}.h5'.format(opt)
            model = keras.models.load_model(model_path, compile=False)
            model.compile(optimizer=opt_dict[opt.lower()], loss=sm.losses.bce_jaccard_loss,
                          metrics=['acc', sm.metrics.iou_score, sm.metrics.f1_score])
            model.summary()
            model.fit_generator(
                generator=self.data_loader.generator(is_train=True, shape=shape,shrink=self.shrink),
                steps_per_epoch=self.data_loader.train_steps,
                validation_data=self.data_loader.generator(is_train=False, shape=shape,shrink=self.shrink),
                validation_steps=self.data_loader.val_steps,
                verbose=1, initial_epoch=0, epochs=300, callbacks=[
                    keras.callbacks.TensorBoard('logs'),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=7, verbose=1),
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=38, verbose=1),
                    keras.callbacks.ModelCheckpoint(monitor='val_loss', verbose=1,
                                                    save_weights_only=False, save_best_only=True,
                                                    filepath='saved_models/segmentation/' + new_name
                                                    )
                ]

            )
        else:
            if model_name.lower() == 'unet':
                model = sm.Unet(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1,
                                input_shape=(shape[0], shape[1], 3),
                                decoder_use_batchnorm=True)
            elif model_name.lower() == 'pspnet':
                model = sm.PSPNet(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1,
                                  input_shape=(shape[0], shape[1], 3))
            elif model_name.lower() == 'fpn':
                model = sm.FPN(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1,
                               input_shape=(shape[0], shape[1], 3))
            elif model_name.lower() == 'linknet':
                model = sm.Linknet(backbone, encoder_weights='imagenet', activation='sigmoid', classes=1,
                                   input_shape=(shape[0], shape[1], 3))
            else:
                raise NotImplementedError
            model.compile(optimizer=opt_dict[opt.lower()], loss=sm.losses.bce_jaccard_loss,
                          metrics=['acc', sm.metrics.iou_score, sm.metrics.f1_score])
            model.summary()
            name_list=[model_name, backbone, opt,'init-training',
                       'none' if shape is None else str(shape[0])+'x'+str(shape[1])]
            if self.shrink is None:
                name_list.append('without-shrink')
            new_name = '_'.join(name_list) + '.h5'
            model.fit_generator(
                generator=self.data_loader.generator(is_train=True, shape=shape,shrink=self.shrink),
                steps_per_epoch=self.data_loader.train_steps,
                validation_data=self.data_loader.generator(is_train=False, shape=shape,shrink=self.shrink),
                validation_steps=self.data_loader.val_steps,
                verbose=1, initial_epoch=0, epochs=300, callbacks=[
                    keras.callbacks.TensorBoard('logs'),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=7, verbose=1),
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=38, verbose=1),
                    keras.callbacks.ModelCheckpoint(monitor='val_loss', verbose=1,
                                                    save_weights_only=False, save_best_only=True,
                                                    filepath='saved_models/segmentation/' + new_name
                                                    )
                ]

            )

    def predict(self,model_path,shape,
                test_img_dir='data/personai_icartoonface_detval'):
        model = keras.models.load_model(model_path,compile=False)
        os.makedirs('../predictions', exist_ok=True)
        files=os.listdir(test_img_dir)
        name=model_path.split('/')[-1].replace('.h5','_predictions.csv')
        result=open('predictions/{}'.format(name),'w',encoding='utf-8')
        for img_name in tqdm.tqdm(files,total=len(files)):
            img_path=os.path.join(test_img_dir,img_name)
            img=cv2.imread(img_path)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            original_shape=img.shape
            resized_img=cv2.resize(img,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
            h_rate,w_rate=original_shape[0]/shape[0],original_shape[1]/shape[1]
            x=resized_img/255.0
            x=np.expand_dims(x,axis=0)
            res=model.predict(x)[0]
            res = np.squeeze(res, axis=-1) * 255
            rgb_mask = np.zeros(shape=(256, 256, 3), dtype=np.uint8)
            rgb_mask[:, :, 0] = res
            rgb_mask[:, :, 1] = res
            rgb_mask[:, :, 2] = res
            gray = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
            contours, hier = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                minAreaRect = cv2.minAreaRect(cnt)
                rectCnt = np.int64(cv2.boxPoints(minAreaRect))
                xmin=int(min(rectCnt[:,0])*w_rate)
                xmax=int(max(rectCnt[:,0])*w_rate)
                ymin=int(min(rectCnt[:,1])*h_rate)
                ymax=int(max(rectCnt[:,1])*h_rate)
                result.write('{},{},{},{},{},{},{}\n'.format(
                    img_name,xmin,ymin,xmax,ymax,'face',0.90
                ))
        result.close()


if __name__=='__main__':
   app = Segmentation(batch_size=20,shrink=0.1,use_aug=False)
   app.predict(model_path='saved_models/segmentation/unet_resnet50_adam_init-training_256x256_without-shrink.h5',
               shape=(256,256))
   app.predict(model_path='saved_models/segmentation/unet_resnet50_adam_init-training_256x256.h5',
               shape=(256,256))