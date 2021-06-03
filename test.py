import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input

from models import *


def test(model_name, weight_file, image_size, image_list, data_dir, label_list, label_dir, return_results=True,
         save_dir=None,
         label_suffix='.png',
         data_suffix='.jpg'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    batch_shape = (1,) + image_size + (3,)
    save_path = os.path.join(current_dir, 'Models/' + model_name)
    model_path = os.path.join(save_path, "model.json")
    checkpoint_path = os.path.join(save_path, weight_file)


    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    model = globals()[model_name](batch_shape=batch_shape, input_shape=(512, 512, 3))
    model.load_weights(checkpoint_path, by_name=True)

    model.summary()
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 21] = np.array([[0, 0, 0],
                                 [128, 0, 0],
                                 [0, 128, 0],
                                 [128, 128, 0],
                                 [0, 0, 128],
                                 [128, 0, 128],
                                 [0, 128, 128],
                                 [128, 128, 128],
                                 [64, 0, 0],
                                 [192, 0, 0],
                                 [64, 128, 0],
                                 [192, 128, 0],
                                 [64, 0, 128],
                                 [192, 0, 128],
                                 [64, 128, 128],
                                 [192, 128, 128],
                                 [0, 64, 0],
                                 [128, 64, 0],
                                 [0, 192, 0],
                                 [128, 192, 0],
                                 [0, 64, 128]], dtype='uint8').flatten()
    results = []
    total = 0
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total, img_num))
        image = Image.open('%s/%s%s' % (data_dir, img_num, data_suffix))
        image = img_to_array(image)  # , data_format='default')


        img_h, img_w = image.shape[0:2]

        pad_w = max(image_size[1] - img_w, 0)
        pad_h = max(image_size[0] - img_h, 0)
        image = np.lib.pad(image, (
            (int(pad_h / 2), math.ceil(pad_h - pad_h / 2)), (int(pad_w / 2), math.ceil(pad_w - pad_w / 2)),
            (0, 0)), 'constant',
                           constant_values=0.)

        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        result = model.predict(image, batch_size=1)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)
        result_img = Image.fromarray(result, mode='P')
        result_img.putpalette(palette)
        result_img = result_img.crop((pad_w / 2, pad_h / 2, pad_w / 2 + img_w, pad_h / 2 + img_h))
        if return_results:
            results.append(result_img)
        if save_dir:
            result_img.save(os.path.join(save_dir, img_num + '.png'))
    return results


if __name__ == '__main__':
    model_name = 'AtrousFCN_Resnet50_16s'
    # model_name = 'Atrous_DenseNet'
    # model_name = 'DenseNet_FCN'
    weight_file = 'checkpoint_weights.hdf5'
    image_size = (512, 512)
    nb_classes = 21
    batch_size = 1
    test_file_path = '/home/mist/cv3/test.txt'
    label_file_path = '/home/mist/cv3/val.txt'
    data_dir = os.path.expanduser('/home/mist/cv3/images')
    label_dir = os.path.expanduser('/home/mist/cv3/annotations_trainval')
    save_dir = '/home/mist/cv3/test/'
    fp = open(test_file_path)
    test_list = fp.readlines()
    fp.close()
    fp = open(label_file_path)
    label_list = fp.readlines()
    fp.close()
    test(model_name, weight_file, image_size, test_list, data_dir, label_list, label_dir, return_results=False,
         save_dir=save_dir, label_suffix='.png', data_suffix='.jpg')
