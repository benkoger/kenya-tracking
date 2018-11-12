import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from tensorflow.python.platform import gfile
import zipfile
import cv2
import glob


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

import matplotlib.pyplot as plt

class ImageReader:
    def __init__(self, image_file_list, queue_size=100):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.image_files = image_file_list
        self.total_frames = len(image_file_list)
        print('Processing ', self.total_frames, 'frames.')
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
                #image name
                image_name = os.path.splitext(os.path.basename(self.image_files[frame_num]))[0]
                # read the next frame from the file
                image_np = cv2.imread(self.image_files[frame_num])
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                # Expand dimensions since the model expects images 
                #to have shape: [1, None, None, 3]
                image_np = np.expand_dims(image_np, axis=0)
                # add the frame to the queue
                self.Q.put([image_name, image_np])
                
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

class ObjectDetection():

    #Any model exported using the export_inference_graph.py 
    #tool can be loaded here simply by changing PATH_TO_CKPT 
    #to point to a new .pb file.


    def __init__(self, output_directory, tf_record_file_list, path_to_checkpoint, path_to_labels, num_classes, verbose=False, 
                label_every_n_frames = 1, save_positions=False, extract=False,
                record_drone_movement=False, first_frame=0, last_frame=None):
        
        self.tf_record_file_list = tf_record_file_list
        # make the nessisary folders to save outputs and image extracts
        try:
          file_name = self.tf_record_file_list[0]
          file_name = file_name.split('/')[-2]
          video_folder = os.path.join(output_directory, file_name)
          os.makedirs(video_folder, exist_ok=True)
          print('outputs saved at', video_folder)
          self.position_folder = os.path.join(video_folder, 'localizations')
          os.makedirs(self.position_folder, exist_ok=True)
          self.extracts_folder = os.path.join(video_folder, 'extracts')
          os.makedirs(self.extracts_folder, exist_ok=True)
        except:
          print('failed to make folders.  TFRecords may have unexpected name')
          
        self.verbose = verbose

        # Path to frozen detection graph. This is the actual model 
        #that is used for the object detection (.pb).
        self.path_to_checkpoint = path_to_checkpoint
        # List of the strings that is used to add correct label for each box (.pbtxt).
        self.path_to_labels = path_to_labels
        self.num_classes = num_classes
        self.label_every_n_frames = label_every_n_frames 
        self.save_positions = save_positions
        self.extract = extract
        self.record_drone_movement = record_drone_movement
        self.check_drone_movement = 1 
        #used for drone movement calculations
        self.last_saved_movement_reference_image = None
        #If this is set to a number then the program will process up to this number of images 
        #(if labeling every frame. Otherwise will process num_files_return / label_n_frames files)
        self.first_frame = first_frame
        self.last_frame = last_frame

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
        #self.images_dir 
        
        if not gfile.Exists(self.images_dir):
            if self.verbose:
                print("Image directory '" + self.images_dir + "' not found.")
            return False
        else:
            result = {}

            file_list = []
            if self.verbose:
                print("Looking for images in '" + self.images_dir + "'")

            file_list = glob.glob(self.images_dir + '**/*.jpg', recursive=True)
            if not file_list:
                if self.verbose:
                    print('No files found')
                return False
            else:
                if self.verbose:
                    print(len(file_list), ' files found')
                #file_list.sort()
                example_split_file = file_list[0].split('.')
                #multiple files per frame -> extracted object images
                if len(example_split_file) >= 3:
                    print('These are extracted images from frame -> Two level sort')
                    file_list.sort(key=lambda file: (int(file.split('.')[0].split('_')[-1]), int(file.split('.')[2])))
                else:
                    print('These are raw video frames -> Single level sort')
                    file_list.sort(key=lambda file: int(file.split('.')[-2].split('_')[-1]))
                length_file_list = len(file_list)

                if self.first_frame:
                    if length_file_list < self.first_frame:
                        print('first frame is greater than number of frames in folder')
                        return False
                    else:
                        print('valid firstframe value of ', self.first_frame, 'being used.')
                        file_list = file_list[self.first_frame:]
                if self.last_frame:
                    if length_file_list < self.last_frame:
                        print('last frame is greater than number of frames in folder')
                        print('returning the max number of frames')
                        return file_list
                    else:
                        print('valid lastframe value of ', self.last_frame, 'being used.')
                        file_list = file_list[:self.last_frame - self.first_frame]

                return file_list[::self.label_every_n_frames]
            


    def _initialize_video(self, video_file_name, video_width, video_height):
        fourcc = cv2.VideoWriter_fourcc(*'ffv1')
        out = cv2.VideoWriter(filename=video_file_name, fourcc=fourcc, fps=30, 
                              frameSize=(int(video_width), int(video_height)), isColor=True)
        return out
   
    def _extract(self, boxes, classes, image, image_name):
      
      #will crop out and save a crop_size x crop_size box around the center of the 
      #detected box
      crop_size = 160
      
      for box_num in range(boxes.shape[0]):
        box = boxes[box_num].copy()
        height = image.shape[0]
        width = image.shape[1]

#        box_scaled = box.copy()
#        box_scaled[0] = box[0] * height
#        box_scaled[1] = box[2] * height
#        box_scaled[2] = box[1] * width
#        box_scaled[3] = box[3] * width
        
        #below is equivelent: box_center_height = height * (box[0] + (box[1] - box[0]) / 2)
        box_center_height = height / 2 * (box[0] + box[2])
    
        #below is equivelent: box_center_width = width * (box[2] + (box[3] - box[2]) / 2)
        box_center_width = width / 2 * (box[1] + box[3])

        at_edge = {}
        
        box[0] = box_center_height - crop_size / 2
        if box[0] < 0:
          at_edge['height'] = box[0]
          box[0] = 0
        box[1] = box_center_height + crop_size / 2
        #Make sure bounding box fits within the height of the image
        if box[1] >= height:
          at_edge['height'] = box[1]
          box[1] = height - 1
        box[2] = box_center_width - crop_size / 2
        if box[2] < 0:
          at_edge['width'] = box[2]
          box[2] = 0
        box[3] = box_center_width + crop_size / 2
        if box[3] >= width:
          at_edge['width'] = box[3]
          box[3] = width - 1
          
      
        box_scaled = np.floor(box).astype(int)

        image_crop = (
          image[box_scaled[0]:box_scaled[1], box_scaled[2]:box_scaled[3]])
        
        if bool(at_edge):
          
          patch_start_height =np.random.randint(height - crop_size)
          patch_start_width =np.random.randint(width - crop_size)
          if 'height' in at_edge:
            if at_edge['height'] > 0:
              patch = image[patch_start_height:int(1 + patch_start_height + (at_edge['height']-height)),
                            patch_start_width:(patch_start_width + box_scaled[3]-box_scaled[2])]
              image_crop = np.vstack((image_crop, patch))

            else:
              #at_edge is negative 
              #need the +1 to account for zero position
              patch = image[patch_start_height:int(patch_start_height - at_edge['height'] + 1), 
                            patch_start_width:(patch_start_width + box_scaled[3]-box_scaled[2])]
              image_crop = np.vstack((patch, image_crop))
          if 'width' in at_edge:
            if at_edge['width'] > 0:
              patch = image[patch_start_height:int(patch_start_height+ crop_size), 
                            patch_start_width:int(1+ patch_start_width + at_edge['width'] - width)]
              image_crop = np.hstack((image_crop, patch))

            else:
              #at_edge is negative 
              #need the +1 to account for zero position
              patch = image[patch_start_height:int(patch_start_height+ crop_size), 
                            patch_start_width:int(patch_start_width - at_edge['width'] + 1)]
              image_crop = np.hstack((patch, image_crop))
              


        if classes[box_num] in self.category_index.keys():
          class_name = self.category_index[classes[box_num]]['name']
        else:
          class_name = 'no_label'
        try:
          img = Image.fromarray(image_crop)
          path = os.path.join(self.extracts_folder, image_name, class_name)
          os.makedirs(path, exist_ok=True)
          if bool(at_edge):
            edge_name = 'edge'
          else:
            edge_name = 'full'
          img.save(os.path.join(path, '{}.{}.{}.{}.png'.format(image_name, class_name, box_num, edge_name)))
        except Exception as e:
          print(e)
          

        
        
    def _measure_drone_movement(self, image, drone_movement_list):
        #When the drone is expected to be static
        long_wait = 10
        #When the drone is expected to be moving
        short_wait = 2
        if self.last_saved_movement_reference_image is not None:
            last_im_gray = self.last_saved_movement_reference_image
            new_im_gray = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            shift, _ = cv2.phaseCorrelate(last_im_gray, new_im_gray)
            shift_per_frame = np.array(shift) / self.check_drone_movement
            for frame in range(self.check_drone_movement):
                drone_movement_list.append(shift_per_frame)
            self.last_saved_movement_reference_image = new_im_gray
            # This is meant to be a quicker aproximation of distance drone moved
            if np.abs(np.sum(shift_per_frame)) > .1:
                return short_wait
            else:
                return long_wait
        else:
            self.last_saved_movement_reference_image = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            drone_movement_list.append(np.zeros(2))
            return short_wait
          
    def _old_label_images(self, file_list):
        if self.save_positions:
            positions_list = []
            boxes_list = []
            classes_list = []
            scores_list = []
            drone_movement_list = []
            
        
        #create then initialize the image reader queue
        image_reader = ImageReader(file_list, queue_size=1000)
        image_reader.start()  
            
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                if self.verbose:
                    print('extraction begun...')
                    
                last_movement_record = 0
                for image_count, image_path in enumerate(file_list):
                        
                    # the array based representation of the image will be used 
                    #later in order to prepare the result image with boxes 
                    #and labels on it.
                    image_name, image_np = image_reader.read()

                    if image_count % 500 == 0:
                        if self.verbose:
                            print(image_name)
                    if image_count % 10000 == 0:
                        if self.verbose:
                            print(image_count, ' images processed')

                    if image_np is None:
                        if self.verbose:
                            print('image failed to load: \n' , image_path)
                        continue

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
                      feed_dict={image_tensor: image_np})

                    #get normal image dimensions for post processing
                    image_np = np.squeeze(image_np)

                    boxes = np.squeeze(boxes)
                    classes = np.squeeze(classes).astype(np.int32)
                    scores = np.squeeze(scores)

                    boxes = boxes[np.where(scores > .6)]
                    classes = classes[np.where(scores > .6)]
                    scores = scores[np.where(scores > .6)]

                    if self.extract:
                        self._extract(boxes, classes, image_np, image_name)


                    boxes_list.append(boxes)
                    classes_list.append(classes)
                    scores_list.append(scores)

                    if self.record_drone_movement:
                        #This is the last image being processed
                        if image_count == len(file_list) - 1:
                            self.check_drone_movement = image_count - last_movement_record
                            self._measure_drone_movement(image_np, drone_movement_list)
                            print('drone movement finished')
                            
                        elif (image_count - last_movement_record) % self.check_drone_movement == 0:
                            self.check_drone_movement = self._measure_drone_movement(
                                image_np, drone_movement_list)
                            last_movement_record = image_count



            
        np.save(os.path.join(self.position_folder,'boxes.npy'), boxes_list)
        np.save(os.path.join(self.position_folder,'classes.npy'), classes_list)
        np.save(os.path.join(self.position_folder,'scores.npy'), scores_list)
        # np.save(os.path.join(self.position_folder,'drone_movement.npy'), drone_movement_list)
        if self.verbose:
            print('Image extraction finished')
            
            
            

    def _label_images(self):
        score_thresh = .1
        if self.save_positions:
            positions_list = []
            boxes_list = []
            classes_list = []
            scores_list = []
            drone_movement_list = []
            
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                if self.verbose:
                    print('extraction begun...')
                    
                last_movement_record = 0
                image_count = 0
                
                #for n in tf.get_default_graph().as_graph_def().node:
                #  print(n.name)
                
                
                filenames_tensor = self.tf_record_file_list


                dataset = tf.data.TFRecordDataset(filenames_tensor)
                
                def _normalize(image):
                  epsilon = .00001
                  image = tf.cast(image, tf.float32)
                  mean, variance = tf.nn.moments(image, axes=[0, 1, 2])
                  return (image - mean) / (variance + epsilon)
                  

                def _parse_tf_record(example_proto):
                  features = {
                    'image/height': tf.FixedLenFeature((), tf.int64),
                    'image/width': tf.FixedLenFeature((), tf.int64),
                    'image/colorspace': tf.FixedLenFeature((), tf.string),
                    'image/channels': tf.FixedLenFeature((), tf.int64),
                    'image/format': tf.FixedLenFeature((), tf.string),
                    'image/filename':tf.FixedLenFeature((), tf.string),
                    'image/encoded': tf.FixedLenFeature([], tf.string)}
                  parsed_features = tf.parse_single_example(example_proto, features)
                  parsed_features['image'] = tf.image.decode_jpeg(
                    parsed_features['image/encoded'], channels=3, ratio=1)
                  # parsed_features['image'] = tf.image.adjust_contrast(parsed_features['image'], 1.5)
                  return parsed_features['image'], parsed_features['image/filename']
                  # return (parsed_features['image/filename'])


                dataset = dataset.map(_parse_tf_record, num_parallel_calls=2)
                dataset = dataset.batch(16)
                dataset = dataset.prefetch(1)

                # iterator = dataset.make_initializable_iterator()
                iterator = dataset.make_one_shot_iterator()
                next_image, next_filename = iterator.get_next(name='raw_image_batch')

                

                while True:
                    try:

                        if image_count % 300 == 0:
                            if self.verbose:
                                if self.save_positions:   
                                    np.save(os.path.join(self.position_folder,'boxes.npy'), boxes_list)
                                    np.save(os.path.join(self.position_folder,'classes.npy'), classes_list)
                                    np.save(os.path.join(self.position_folder,'scores.npy'), scores_list)

                                print(image_count, ' images processed')
                        
                        
                                

                        #filenames_tensor = self.detection_graph.get_tensor_by_name('filenames:0')
                        # Each box represents a part of the image where a 
                        #particular object was detected.
                        boxes_batch = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                        # Each score represent how level of confidence for each of the objects.                                                                                               # Score is shown on the result image, together with the class label.
                        scores_batch = self.detection_graph.get_tensor_by_name('detection_scores:0')
                        classes_batch = self.detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections_batch = self.detection_graph.get_tensor_by_name('num_detections:0')
                        #raw_images_batch = self.detection_graph.get_tensor_by_name('raw_image_batch:0')
                        #filenames_batch = self.detection_graph.get_tensor_by_name('raw_image_batch:1')
                        
                        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                        
                        raw_images_batch, filenames_batch = sess.run([next_image, next_filename]) 
              
                        # Actual detection.
                        (boxes_batch, scores_batch, classes_batch, num_detections_batch) = sess.run(
                          [boxes_batch, scores_batch, classes_batch, num_detections_batch], 
                          feed_dict={image_tensor: raw_images_batch})
                  
                        #(boxes, scores, classes, num_detections) = sess.run(
                        #  [boxes, scores, classes, num_detections])

                        #get normal image dimensions for post processing
                        #image_np = np.squeeze(image_np)
                        
                        
                        image = raw_images_batch[0]

                        
                        for batch_ind in range(boxes_batch.shape[0]):
                          boxes = boxes_batch[batch_ind]
                          classes = classes_batch[batch_ind].astype(np.int32)
                          scores = scores_batch[batch_ind]
                          raw_image = raw_images_batch[batch_ind]
                          filename = str(filenames_batch[batch_ind])[2:-1]
                          filename = filename.split('.')[-2]
                          
                          boxes = boxes[np.where(scores > score_thresh)]
                          classes = classes[np.where(scores > score_thresh)]
                          scores = scores[np.where(scores > score_thresh)]



                          if self.extract:
                              self._extract(boxes, classes, raw_image, filename)

                          if self.save_positions:
                            boxes_list.append(boxes)
                            classes_list.append(classes)
                            scores_list.append(scores)

                          if self.record_drone_movement:
                              #This is the last image being processed
                             # if image_count == len(file_list) - 1:
                             #     self.check_drone_movement = image_count - last_movement_record
                             #     self._measure_drone_movement(image_np, drone_movement_list)
                             #     print('drone movement finished')

                              if (image_count - last_movement_record) % self.check_drone_movement == 0:
                                  self.check_drone_movement = self._measure_drone_movement(
                                      raw_image, drone_movement_list)
                                  last_movement_record = image_count

                          image_count += 1
               
                    except tf.errors.OutOfRangeError:
                        print('finshed')
                        break

        if self.save_positions:   
          np.save(os.path.join(self.position_folder,'boxes.npy'), boxes_list)
          np.save(os.path.join(self.position_folder,'classes.npy'), classes_list)
          np.save(os.path.join(self.position_folder,'scores.npy'), scores_list)
        if self.verbose:
            print('Image extraction finished')
            print('image_count:', image_count)
            print('boxes saved to ', os.path.join(self.position_folder,'boxes.npy'))



    def track(self):
      
        self.use_dataset = True
        
        self._load_graph()
        self._load_label_map() 
        if not self.use_dataset:
            file_list = self._get_files()
        if not self.use_dataset:
            self._old_label_images(file_list)
        else:
            self._label_images()

