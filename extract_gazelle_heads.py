import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from tensorflow.python.platform import gfile
import zipfile


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#imports from the object detection module
from utils import label_map_util
from utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    image_array = np.array(image.getdata())

    #make 2d bw images 3d rgb (still grayscale)
    if len(image_array.shape) == 1:
        try:
            image_array = image_array.reshape((im_height, im_width)).astype(np.uint8)
        except:
            return None
        image_array = np.expand_dims(image_array, 2)
        image_array = np.tile(image_array, (1, 1, 3))
    else:
        try:
            image_array = image_array.reshape((im_height, im_width, 3)).astype(np.uint8)
        except:
            return None
    
    return image_array

class ExtractGazelleHeads():

    #Any model exported using the export_inference_graph.py 
    #tool can be loaded here simply by changing PATH_TO_CKPT 
    #to point to a new .pb file.

    # Path to frozen detection graph. This is the actual model 
    #that is used for the object detection.
    PATH_TO_CKPT = '/home/golden/Projects/gazelle_identification/gazelle/output_graph.pb' 

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = '/home/golden/Projects/gazelle_identification/gazelle/data/gazelle_head_label_map.pbtxt'

    NUM_CLASSES = 2

    #Create folder for extracted images
    PATH_TO_EXTRACTED_IMAGES = 'data/extracted/test/'
    os.makedirs(os.path.dirname(PATH_TO_EXTRACTED_IMAGES), exist_ok=True)

    def __init__(verbose=False):
        self.verbose = verbose


    def _load_graph(self):
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def _load_label_map(self):
        #Loading label map
        #Label maps map indices to category names, so that when our convolution 
        #network predicts 5, we know that this corresponds to airplane. 
        #Here we use internal utility functions, but anything that returns a 
        #dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def _get_files(self):
        PATH_TO_IMAGES_DIR = '/home/golden/Projects/gazelle_identification/gazelle/raw_images'

        dir_name = os.path.dirname(PATH_TO_IMAGES_DIR)

        if not gfile.Exists(PATH_TO_IMAGES_DIR):
            if self.verbose:
                print("Image directory '" + dir_name + "' not found.")
            return False
        else:
            result = {}

            extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
            file_list = []
            if self.verbose:
                print("Looking for images in '" + dir_name + "'")
            for extension in extensions:
                file_glob = os.path.join(PATH_TO_IMAGES_DIR, '*.' + extension)
                file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                if self.verbose:
                    print('No files found')
                return False
            else:
                if self.verbose:
                    print(len(file_list), ' files found')
                return file_list

    def _extract(self, file_list):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                if self.verbose:
                    print('extraction begun...')
                for image_count, image_path in enumerate(file_list):
                    if image_count % 100 == 0:
                        if verbose:
                            print(image_count, ' images processed')
                    image = Image.open(image_path)
                    # the array based representation of the image will be used 
                    #later in order to prepare the result image with boxes 
                    #and labels on it.
                    image_np = load_image_into_numpy_array(image)
                    if image_np is None:
                        if self.verbose:
                            print('image failed to load: \n' , image_path)
                        continue
                        
                    # Expand dimensions since the model expects images 
                    #to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a 
                    #particular object was detected.
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                      [boxes, scores, classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
                    
                    boxes = np.squeeze(boxes)
                    classes = np.squeeze(classes)
                    score = np.squeeze(scores)

                    boxes = boxes[np.where(score > .6)]
                    classes = classes[np.where(score > .6)]
                    
                    if boxes.shape[0] > 0:
                        boxes = boxes[np.where(score > .6)]
                        for box_num in range(boxes.shape[0]):
                            box = boxes[box_num]
                            height = image_np.shape[0]
                            width = image_np.shape[1]

                            box_scaled = box.copy()
                            box_scaled[0] = box[0] * height
                            box_scaled[1] = box[2] * height
                            box_scaled[2] = box[1] * width
                            box_scaled[3] = box[3] * width

                            box_scaled = np.floor(box_scaled).astype(int)

                            image_crop = (
                                image_np[box_scaled[0]:box_scaled[1], box_scaled[2]:box_scaled[3]])
                            
                            if classes[box_num] in self.category_index.keys():
                                class_name = self.category_index[classes[box_num]]['name']
                            else:
                                class_name = 'no_label'
                            
                            img = Image.fromarray(image_crop)
                            path = os.path.join(PATH_TO_EXTRACTED_IMAGES, class_name)
                            os.makedirs(path, exist_ok=True)
                            file_name = os.path.splitext(os.path.basename(image_path))[0]
              
                            img.save(os.path.join(path, '{}.{}.jpg'.format(file_name, box_num)))

            
        if self.verbose:
            print('Image extraction finished')



    def extract_heads():
        self._load_graph()
        self._load_label_map() 
        file_list = self._get_files()
        self._extract(file_list)

