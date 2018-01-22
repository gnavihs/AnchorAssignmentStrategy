# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Flickr Logos dataset to TFRecord for object_detection.

See: Stefan Romberg, Lluis Garcia Pueyo, Rainer Lienhart, Roelof van Zwol
     Scalable Logo Recognition in Real-World Images
     http://www.multimedia-computing.de/flickrlogos/

Example usage:
    ./python create_flickrlogos_tf_record \
        --label_map_path=PATH_TO_DATASET_LABELS \
        --data_dir=PATH_TO_DATA_FOLDER \
        --output_path=PATH_TO_OUTPUT_FILE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/data/chercheurs/agarwals/MunichDataset', 'Root directory to raw Flickr dataset.')
flags.DEFINE_string('set', 'Test', 'Convert training set or merged set.')
flags.DEFINE_string('output_path', '/data/chercheurs/agarwals/MunichDataset/tfrecords', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'models/labels/flickrlogos47_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

SETS = ['Train', 'Test']


def parse_annotations(annos):
    annotations = []
    for anno in annos:
        a = {}
        a['xmin'] = int(anno[2])
        a['ymin'] = int(anno[3])
        a['xmax'] = int(anno[4])
        a['ymax'] = int(anno[5])
        a['class'] = str(anno[1])
        a['angle'] = str(anno[6])
        annotations.append(a)

    return annotations


def get_label_map_text(label_map_path):
    """Reads a label map and returns a dictionary of label names to display name.

    Args:
      label_map_path: path to label_map.

    Returns:
      A dictionary mapping label names to id.
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    label_map_text = {}
    for item in label_map.item:
        label_map_text[item.name] = item.display_name
    return label_map_text

def data_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       label_map_text,
                       ignore_difficult_instances=False):
    """Convert TXT derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: TO REPLACE - dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = data['img_path']
    img_bytes = tf.gfile.FastGFile(img_path,'rb').read()

    width = 5616
    height = 3744

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for obj in data['annos']:
        xmin.append(float(obj['xmin']) / width)
        ymin.append(float(obj['ymin']) / height)
        xmax.append(float(obj['xmax']) / width)
        ymax.append(float(obj['ymax']) / height)
        classes_text.append(label_map_text[obj['class']].encode('utf8'))
        classes.append(label_map_dict[obj['class']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['img_path'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['img_path'].encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(img_bytes),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    label_map_text = get_label_map_text(FLAGS.label_map_path)

    print('Reading from Munich dataset.')
    examples_path = os.path.join(data_dir, FLAGS.set)
    for root, dirs, files in os.walk(examples_path):
        #Get one image file at a time and all its annotation and make tf record
        for file in files:
            if file.endswith('.JPG'):
                print("Reading", file)
                lines = []
                #Read all the annotations corresponding to an image
                for anno_file in files:
                    if anno_file.endswith('.samp') and anno_file.startswith(file[:-4]):
                        print("Reading anno file", anno_file)
                        with open(os.path.join(root, anno_file)) as f:
                            for line in f:
                                if line.startswith('@') or line.startswith('#'):
                                    continue
                                lines.append(line.strip().split(' '))
                print(len(lines))
                #Write the image and annotaions in the tf record
                data = {}
                data['img_path'] = os.path.join(root, file)
                data['annos'] = parse_annotations(lines)
                tf_example = data_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                                label_map_text)

                writer.write(tf_example.SerializeToString())

    writer.close()



if __name__ == '__main__':
    tf.app.run()
