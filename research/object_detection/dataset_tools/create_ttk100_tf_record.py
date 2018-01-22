r"""Convert raw Microsoft ttk dataset to TFRecord for object_detection.
Attention Please!!!

This script converts all the ttk data to tfrecord files:
All but 8000 images from val set are used for trainig too.

1)For easy use of this script, Your ttk dataset directory struture should like this :
    +Your ttk dataset root
        +images/train2014
        +images/val2014
        +annotations
            -instances_train2014.json
            -instances_val2014.json
2)To use this script, you should download python ttk tools from "http://msttk.org/dataset/#download" and make it.
After make, copy the pyttktools directory to the directory of this "create_ttk_tf_record.py"
or add the pyttktools path to  PYTHONPATH of ~/.bashrc file.

Example usage:
    py create_ttk_tf_record.py --data_dir=/data/chercheurs/agarwals/ttk/ \
        --output_dir=/data/chercheurs/agarwals/ttk/tfrecords/ \
        --shuffle_imgs=True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import os, sys
import numpy as np
import tensorflow as tf
import logging
import random
from sklearn.utils import shuffle
import json
from pprint import pprint
    
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/data/chercheurs/agarwals/TTK100/data', 'Root directory to ttk dataset.')
flags.DEFINE_string('output_dir', '/data/chercheurs/agarwals/TTK100/tfrecords', 'Path to output TFRecord directory')
flags.DEFINE_bool('shuffle_imgs',True,'whether to shuffle images of ttk')
FLAGS = flags.FLAGS

def create_label_map(data_dir, output_name='label_map.pbtxt'):
    """ Creates a proto label map based on the labels file from
        the Flickr dataset.

        name: is the code that the class is referred in the dataset
        id: is the id we will use inside our TF model (starts at 1)
        display_name: name to be displayed
    """

    type45="i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
    type45 = type45.split(',')
    print(len(type45))
    print(os.path.join(data_dir, output_name))
    with open(os.path.join(data_dir, output_name), 'a') as text_file:
        for idx, category in enumerate(type45):
            # print(category)
            # print(idx)
            text_file.write("item {\n")
            text_file.write("  id: {}\n".format(idx+1))
            text_file.write("  display_name: {}\n".format("\""+category+"\""))
            text_file.write("}\n")
    print("finished")


def write_ttk_dection_dataset(imgs_dir, annotations_filepath, shuffle_img = True, datatype='train'):
    """Load data from dataset by pyttktools.
    Args:
        imgs_dir: directories of ttk images
        annotations_filepath: file paths of ttk annotations file
        shuffle_img: whether to shuffle images order
    Return:
        ttk_data: list of dictionary format information of each image
    """
    data = json.load(open(annotations_filepath))
    img_ids = data['imgs'].keys()
    # print(data['types'])
    type45="i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"
    type45 = type45.split(',')

    print('Total number of images = ', len(img_ids)) #Should be 16811
    nb_imgs = len(img_ids)
    ttk_data = []
    counter = 0
    # TFRECORD FOR IMAGES
    for index, img_id in enumerate(img_ids):
        if not data['imgs'][img_id]['path'].startswith(datatype):
            continue
        if counter % 2000 == 0:
            print("Reading images: %d / %d "%(counter, nb_imgs))
        img_info = {}
        bboxes = []
        labels = []

        img_path = os.path.join(imgs_dir, data['imgs'][img_id]['path'])
        
        pic_height = 2048
        pic_width = 2048
        anns = data['imgs'][img_id]['objects']
        for ann in anns:
            if ann['category'] in type45:
                labels.append(type45.index(ann['category']) + 1)
                bboxes_data = ann['bbox']
                bboxes_data = [bboxes_data['xmax']/float(pic_width), bboxes_data['xmin']/float(pic_width),\
                                    bboxes_data['ymax']/float(pic_height), bboxes_data['ymin']/float(pic_height)]
                            # the format of ttk bounding boxes is [Xmax, Xmin, Ymax, Ymin]
                bboxes.append(bboxes_data)
            else:
                continue
        img_bytes = tf.gfile.FastGFile(img_path,'rb').read()

        img_info['img_id'] = str(img_id)
        img_info['pixel_data'] = img_bytes
        img_info['height'] = pic_height
        img_info['width'] = pic_width
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels

        ttk_data.append(img_info)
        counter += 1
        if datatype=='train':
            outfile_prefix = 'train_'
        elif datatype=='other':
            outfile_prefix = 'other_'
        elif datatype=='test':
            outfile_prefix = 'test_'        # write ttk data to tf record
        if counter > 1 and counter % 2000 == 0:
            i = counter//2000
            filename = outfile_prefix + str(i) + '.record'
            output_filepath = os.path.join(FLAGS.output_dir, filename)
            print('Writing images to: ', output_filepath)
            with tf.python_io.TFRecordWriter(output_filepath) as tfrecord_writer:
                for _, img_data in enumerate(ttk_data):
                    example = dict_to_ttk_example(img_data)
                    tfrecord_writer.write(example.SerializeToString())
            ttk_data = []

    filename = outfile_prefix + str(i+1) + '.record'
    output_filepath = os.path.join(FLAGS.output_dir, filename)
    print('Number of images in final file: ', len(ttk_data))
    print('Writing images to: ', output_filepath)
    with tf.python_io.TFRecordWriter(output_filepath) as tfrecord_writer:
        for _, img_data in enumerate(ttk_data):
            example = dict_to_ttk_example(img_data)
            tfrecord_writer.write(example.SerializeToString())


def dict_to_ttk_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, include img_id, bounding boxes, labels of bounding boxes,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    # these 4 variables are normalised
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmax.append(bbox[0])
        xmin.append(bbox[1])
        ymax.append(bbox[2])
        ymin.append(bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/source_id': dataset_util.bytes_feature(img_data['img_id'].encode('utf-8')),
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
        'image/encoded': dataset_util.bytes_feature(img_data['pixel_data']),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example

def main(_):
    data_dir = FLAGS.data_dir
    annotations_filepath  = os.path.join(data_dir,'annotations.json')
    # create_label_map(FLAGS.output_dir)
    # Write train data in tfrecords
    write_ttk_dection_dataset(data_dir,annotations_filepath,shuffle_img=FLAGS.shuffle_imgs, datatype='train')
    # Write test data in tfrecords
    write_ttk_dection_dataset(data_dir,annotations_filepath,shuffle_img=FLAGS.shuffle_imgs, datatype='test')
    # Write other data in tfrecords
    write_ttk_dection_dataset(data_dir,annotations_filepath,shuffle_img=FLAGS.shuffle_imgs, datatype='other')

if __name__ == "__main__":
    tf.app.run()