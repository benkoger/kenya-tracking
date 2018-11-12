import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from tensorflow.python.platform import gfile
import zipfile
import cv2


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#imports from the object detection module
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from queue import Queue
from threading import Thread
import sys
import time

class ImageReader:
    def __init__(self, image_file_list, queue_size=100):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.image_files = image_file_list
        self.total_frames = len(image_file_list)
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        time.sleep(1)
        return self

    def update(self):
        # keep looping infinitely
        frame_num = 0
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                image_np = cv2.imread(self.image_files[frame_num])
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                # Expand dimensions since the model expects images 
                #to have shape: [1, None, None, 3]
                image_np = np.expand_dims(image_np, axis=0)
                # add the frame to the queue
                self.Q.put(image_np)
                
                frame_num += 1
                
                if frame_num >= self.total_frames:
                    self.stop()
                    return
                
                

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def read_batch(self, n_frames, asarray=False):
        frames = []
        for idx in range(n_frames):
            frames.append(self.read())
        return frames


    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        print('stopping')
        self.stopped = True
        
    def close(self):
        self.stop()

class KenyaTracking():

    #Any model exported using the export_inference_graph.py 
    #tool can be loaded here simply by changing PATH_TO_CKPT 
    #to point to a new .pb file.


    def __init__(self, images_dir, path_to_checkpoint, path_to_labels, num_classes, verbose=False, create_video=False, 
                video_file_output=None, position_file=None, video_width=None, video_height=None, 
                 label_every_n_frames = 1, save_still_frames=False, save_positions=False, extract=False):
        self.verbose = verbose
        self.create_video = create_video
        self.video_file_output = video_file_output
        self.position_file = position_file
        self.video_width = video_width
        self.video_height = video_height
        self.images_dir = images_dir
        # Path to frozen detection graph. This is the actual model 
        #that is used for the object detection (.pb).
        self.path_to_checkpoint = path_to_checkpoint
        # List of the strings that is used to add correct label for each box (.pbtxt).
        self.path_to_labels = path_to_labels
        self.num_classes = num_classes
        self.label_every_n_frames = label_every_n_frames 
        self.save_still_frames = save_still_frames
        self.save_positions = save_positions
        self.extract = extract

    def _load_graph(self):
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_checkpoint, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

    def _load_label_map(self):
        #Loading label map
        #Label maps map indices to category names, so that when our convolution 
        #network predicts 5, we know that this corresponds to airplane. 
        #Here we use internal utility functions, but anything that returns a 
        #dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def _get_files(self):
        PATH_TO_IMAGES_DIR = self.images_dir 

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
                file_list.sort()
                return file_list

    def _process_and_extract(self, file_list):
        #extracting images within bounding boxes.  Gazelle heads, for instance
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                if self.verbose:
                    print('extraction begun...')
                for image_count, image_path in enumerate(file_list):
                    if image_count % 100 == 0:
                        if self.verbose:
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

    def _initialize_video(self, video_file_name, video_width, video_height):
        fourcc = cv2.VideoWriter_fourcc(*'ffv1')
        out = cv2.VideoWriter(filename=video_file_name, fourcc=fourcc, fps=30, 
                              frameSize=(int(video_width), int(video_height)), isColor=True)
        return out
    
    def _get_boxes_center(self, boxes):
        center = np.ones((boxes.shape[0], 2))
        #need to convert from top right to bottom right origin
        center[:, 0] = (self.video_height - 
                        (self.video_height * (boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2.0))) 
        center[:, 1] = self.video_width * (boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2.0)
        return center
   
    def _extract(self, boxes, classes, image, image_name):
         for box_num in range(boxes.shape[0]):
            box = boxes[box_num]
            height = image.shape[0]
            width = image.shape[1]

            box_scaled = box.copy()
            box_scaled[0] = box[0] * height
            box_scaled[1] = box[2] * height
            box_scaled[2] = box[1] * width
            box_scaled[3] = box[3] * width

            box_scaled = np.floor(box_scaled).astype(int)

            image_crop = (
                image[box_scaled[0]:box_scaled[1], box_scaled[2]:box_scaled[3]])

            if classes[box_num] in self.category_index.keys():
                class_name = self.category_index[classes[box_num]]['name']
            else:
                class_name = 'no_label'

            img = Image.fromarray(image_crop)
            path = os.path.join(self.images_dir, 'extracts')
            path = os.path.join(path, image_name, class_name)
            os.makedirs(path, exist_ok=True)

            img.save(os.path.join(path, '{}.{}.{}.jpg'.format(image_name, class_name, box_num)))

        
        
    

    def _label_images(self, file_list):
        if self.save_positions:
            positions_list = []
            boxes_list = []
            classes_list = []
            scores_list = []
            
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                if self.verbose:
                    print('extraction begun...')
                for image_count, image_path in enumerate(file_list):
                    if image_count % self.label_every_n_frames == 0:
                        if image_count % 100 == 0:
                            if self.verbose:
                                print(image_path)
                        if image_count % 1000 == 0:
                            if self.verbose:
                                print(image_count, ' images processed')
                        image = Image.open(image_path)
                        
                        image_name = os.path.splitext(os.path.basename(image_path))[0]
                        
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
                        classes = np.squeeze(classes).astype(np.int32)
                        scores = np.squeeze(scores)

                        boxes = boxes[np.where(scores > .6)]
                        classes = classes[np.where(scores > .6)]
                        scores = scores[np.where(scores > .6)]
                        
                        if self.extract:
                            self._extract(boxes, classes, image_np, image_name)
                            
                            
                        
#                         positions = self._get_boxes_center(boxes)
                        
#                         positions_list.append(positions)
                        
                        boxes_list.append(boxes)
                        classes_list.append(classes)
                        scores_list.append(scores)
                        
                        
                        if self.create_video or self.save_still_frames:
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                  image_np,
                                  boxes,
                                  classes,
                                  scores,
                                  self.category_index,
                                  use_normalized_coordinates=True,
                                  line_thickness=8)
                        if self.create_video:
                            frame = image_np.astype('u1')
                            self.out.write(frame)
                        if self.save_still_frames:
                            file_root, file_ext = os.path.splitext(image_path)
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(file_root + '_labeled' + file_ext, image_np)

            
        np.save(os.path.join(self.position_file,'boxes.npy'), boxes_list)
        np.save(os.path.join(self.position_file,'classes.npy'), classes_list)
        np.save(os.path.join(self.position_file,'scores.npy'), scores_list)
        if self.verbose:
            print('Image extraction finished')



    def track(self):
        
        self._load_graph()
        self._load_label_map() 
        file_list = self._get_files()
        if self.create_video:
            self.out = self._initialize_video(self.video_file_output, self.video_width, self.video_height)
        self._label_images(file_list)

