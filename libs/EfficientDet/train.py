"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from datetime import date
import os
import sys
import tensorflow as tf

# import keras
# # import keras.preprocessing.image
# import keras.backend as K
# from keras.optimizers import Adam, SGD

from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD

from .augmentor.color import VisualEffect
from .augmentor.misc import MiscEffect
from .model import efficientdet
from .losses import smooth_l1, focal, smooth_l1_quad
from .efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


# def get_session():
#     """
#     Construct a modified tf session.
#     """
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     return tf.Session(config=config)


def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    # if args.tensorboard_dir:
    #     # if tf.version.VERSION > '2.0.0':
    #     #     file_writer = tf.summary.create_file_writer(args.tensorboard_dir)
    #     #     file_writer.set_as_default()
    #     tensorboard_callback = keras.callbacks.TensorBoard(
    #         log_dir=args.tensorboard_dir
    #     )
    #     callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from .eval.coco import Evaluate
            # use prediction model for evaluation
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        else:
            from .eval.pascal import Evaluate
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        callbacks.append(evaluation)

    # save the model
    makedirs(args.snapshot_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            args.snapshot_path,
            f'efficientdet_{args.backbone}_{args.dataset_type}_{args.height}x{args.width}_{{epoch:04d}}_{{val_loss:.5f}}.h5'
        ),
        verbose=1,
        save_weights_only=False,
        # save_best_only=True,
        monitor='val_loss',
        mode='min',
        # save_freq=1,
        # save_best_only=True,
        # monitor="mAP",
        # mode='max'
    )
    callbacks.append(checkpoint)
    print('append checkpoint')


    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
    ))
    print('append reducelronplateau')
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,patience=20,
        mode='min',
    ))
    print('append earlystopping')

    return callbacks


def create_generators(args):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
        'detect_text': args.detect_text,
        'detect_quadrangle': args.detect_quadrangle
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        misc_effect = MiscEffect()
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None

    if args.dataset_type == 'pascal':
        from .generators.pascal import PascalVocGenerator
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            skip_difficult=True,
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'val',
            skip_difficult=True,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'csv':
        from .generators.csv_ import CSVGenerator
        train_generator = CSVGenerator(
            args.annotations_path,
            args.classes_path,
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            resized_w=args.width,
            resized_h=args.height,
            with_aug=args.augmentation,
            **common_args
        )

        if args.val_annotations_path:
            print(args.val_annotations_path)
            validation_generator = CSVGenerator(
                args.val_annotations_path,
                args.classes_path,
                shuffle_groups=False,
                resized_w=args.width,
                resized_h=args.height,
                with_aug=False,
                **common_args
            )
        else:
            validation_generator = None
    elif args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from .generators.coco import CocoGenerator
        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            misc_effect=misc_effect,
            visual_effect=visual_effect,
            group_method='random',
            **common_args
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            shuffle_groups=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.gpu and parsed_args.batch_size < len(parsed_args.gpu.split(',')):
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             len(parsed_args.gpu.split(
                                                                                                 ','))))
    return parsed_args


def efficientdet_parse_args():
    """
    Parse the arguments.
    """
    today = str(date.today())
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--annotations-path', help='Path to CSV file containing annotations for training.')
    parser.add_argument('--classes-path', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('--val-annotations-path',
                            help='Path to CSV file containing annotations for validation (optional).')
    parser.add_argument('--detect-quadrangle', help='If to detect quadrangle.', action='store_true', default=False)
    parser.add_argument('--detect-text', help='If is text detection task.', action='store_true', default=False)

    parser.add_argument('--snapshot', help='Resume training from a snapshot.')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--freeze-bn', help='Freeze training of BatchNormalization layers.', action='store_true')
    parser.add_argument('--weighted-bifpn', help='Use weighted BiFPN', action='store_true')

    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).',default=3)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training',
                        default='checkpoints/{}'.format(today))
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output',
                        default='logs/{}'.format(today))
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                        action='store_false')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss',
                        action='store_true')
    parser.add_argument('--height',           default=480)
    parser.add_argument('--width',            default=720)
    parser.add_argument('--augmentation',     default=True)

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers', help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.', type=int,
                        default=10)
    args = parser.parse_args()
    return args


def efficientdet_train(args,train_steps=None,val_steps=None):
    # parse arguments
    print('*'*100)
    print(args)
    print('*'*100)
    def make_sure_only_save_best():
        dir = args.snapshot_path
        scores = {}
        min_loss = 10000000
        prefix= f'efficientdet_{args.backbone}_{args.dataset_type}_{args.height}x{args.width}'
        for name in os.listdir(dir):
            if name.startswith(prefix):
                cur_loss=float(name.strip('.h5').split('_')[-1])
                scores[os.path.join(dir,name)]=cur_loss
                if cur_loss < min_loss:
                    min_loss = cur_loss
        keys = sorted(scores)
        flag = True
        for k in keys:
            if scores[k] > min_loss:
                os.remove(k)
            if scores[k] == min_loss:
                if flag:
                    flag = False
                    print('only saved this model:{}'.format(k))
                else:
                    os.remove(k)

    # create the generators
    train_generator, validation_generator = create_generators(args)

    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors


    model, prediction_model = efficientdet(args.phi,
                                           num_classes=num_classes,
                                           num_anchors=num_anchors,
                                           weighted_bifpn=args.weighted_bifpn,
                                           freeze_bn=args.freeze_bn,
                                           detect_quadrangle=args.detect_quadrangle
                                           )
    # load pretrained weights
    if args.snapshot:
        if args.snapshot == 'imagenet':
            model_name = 'efficientnet-b{}'.format(args.phi)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
        else:
            print('Loading model, this may take a second...')
            model.load_weights(args.snapshot, by_name=True)

    # freeze backbone layers
    if args.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi]):
            model.layers[i].trainable = False

    # if args.gpu and len(str(args.gpu).split(',')) > 1:
    #     model = keras.utils.multi_gpu_model(model, gpus=list(map(int, args.gpu.split(','))))

    # compile model
    model.compile(optimizer=Adam(lr=args.lr), loss={
        'regression': smooth_l1_quad() if args.detect_quadrangle else smooth_l1(),
        'classification': focal()
    }, )

    # model.summary()

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None
    elif args.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    # start training
    if validation_generator is None:
        raise EnvironmentError
    his = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps if train_steps else args.steps,
        initial_epoch=0,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator,
        validation_steps=val_steps,
    )
    print(his.history)
    make_sure_only_save_best()