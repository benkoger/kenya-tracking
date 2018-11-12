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

#Modified by Ben Koger
# ==============================================================================

"""Convert gazelle dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_gazelle_tf_record --data_dir=/home/user/gazelle \
        --output_dir=/home/user/gazelle/output
"""

import hashlib
import io
import logging
import os
import random
import re
from scipy import ndimage
import matplotlib.pyplot as plt

from lxml import etree
import PIL.Image
import tensorflow as tf
import glob
from collections import Counter

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS




def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []

  for obj in data['object']:
    #ignore polygons
    if 'polygon' in obj:
      print(obj)
      print('---------------------------')
      continue

    class_name = str(obj['name'])

    try:
      classes.append(label_map_dict[class_name])
    except:
      print('The class "' + class_name + '" is not currently being used.')
      continue

    classes_text.append(class_name.encode('utf8'))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(0),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
    path = os.path.join(annotations_dir, example + '.xml')

    if not os.path.exists(path):
      logging.warning('Could not find %s, ignoring example.', path)
      continue
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
    writer.write(tf_example.SerializeToString())

  writer.close()
  
def create_train_val_list(species_lists, frac_eval):
  species_names = ['buffalo', 'zebra', 'gazelle', 'wbuck', 'other']
  train_list = []
  val_list = []
  random.seed(42)
  for species_num, species_list in enumerate(species_lists):
    random.shuffle(species_list)
    num_examples = len(species_list)
    frac_train = 1 - frac_eval
    num_train = int(frac_train * num_examples)
    train_list = train_list + species_list[:num_train]
    val_list = val_list + species_list[num_train:]
    print('{} training and {} validation examples for {}.'.format(
      len(species_list[:num_train]), len(species_list[num_train:]), species_names[species_num]))
  return train_list, val_list
    
  
#So species are evenly distributed in train and eval set
def sort_files_by_species(examples_list, annotations_dir, frac_eval):
  
  
    buffalo_annotations = []
    zebra_annotations = []
    gazelle_annotations = []
    other_annotations = []
    wbuck_annotations = []
    
    print('sorting ', len(examples_list), 'examples.')
    
    for example in examples_list:
        path = os.path.join(annotations_dir, example + '.xml')
      
        if not os.path.exists(path):
            logging.warning('Could not find %s, ignoring example.', path)
            continue
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      
        objects_in_image = []
      

        for obj in data['object']:
            #print(obj['name'])
            #ignore polygons
            if 'polygon' in obj:
                print(obj)
                print('---------------------------')
                continue

            class_name = str(obj['name'])
            objects_in_image.append(class_name)
    
        most_common_object, num_objects = Counter(objects_in_image).most_common(1)[0]
        #print(most_common_object, ':', num_objects)
        if most_common_object == 'buffalo':
            buffalo_annotations.append(example)
        elif most_common_object == 'zebra':
            zebra_annotations.append(example)
        elif most_common_object == 'gazelle':
            gazelle_annotations.append(example)
        elif most_common_object == 'wbuck':
            wbuck_annotations.append(example)
        else:
            other_annotations.append(example)
    
    print('total:', len(buffalo_annotations) + len(zebra_annotations) + len(gazelle_annotations) + len(wbuck_annotations) + len(other_annotations))
    print('buffalo:', len(buffalo_annotations))
    print('zebra:', len(zebra_annotations))
    print('gazelle:', len(gazelle_annotations))
    print('wbuck:', len(wbuck_annotations))
    print('other:', len(other_annotations))
    
    species_lists = [buffalo_annotations, zebra_annotations, gazelle_annotations, wbuck_annotations, other_annotations]
    
    return create_train_val_list(species_lists, frac_eval)
    
    


# TODO: Add test for pet/PASCAL main files.
def create_all_tf_records(data_dir, label_map_path, output_dir, frac_eval):

  label_map_dict = label_map_util.get_label_map_dict(label_map_path)

  logging.info('Reading from kenya-tracking dataset.')
  image_dir = data_dir
  annotations_dir = os.path.join(image_dir, 'annotations')
  examples_path = os.path.join(annotations_dir, 'trainval.txt')
  print(image_dir)
  examples_list = dataset_util.read_examples_list(examples_path)
  

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.

  train_examples, val_examples = sort_files_by_species(examples_list, annotations_dir, frac_eval)
  #val_examples = examples_list[num_train:]
  print('{} training and {} validation examples.'.format(
    len(train_examples), len(val_examples)))

  train_output_path = os.path.join(output_dir, 'train.record')
  val_output_path = os.path.join(output_dir, 'val.record')
  create_tf_record(train_output_path, label_map_dict, annotations_dir,
                   image_dir, train_examples)
  create_tf_record(val_output_path, label_map_dict, annotations_dir,
                   image_dir, val_examples)

if __name__ == '__main__':
  tf.app.run()
