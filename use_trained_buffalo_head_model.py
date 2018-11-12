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

class ImageReader:
    def __init__(self, image_file_list, queue_size=100):
        # initializ be the file video stream along with the boolean
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


    def __init__(self, images_dir, path_to_checkpoint, path_to_labels, num_classes, verbose=False, create_video=False, 
                video_file_output=None, position_file=None, video_width=None, video_height=None, 
                 label_every_n_frames = 1, save_still_frames=False, save_positions=False, extract=False,
                record_drone_movement=False, first_frame=0, last_frame=None):
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
            
            
            

    def _label_images(self, file_list):
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
        np.save(os.path.join(self.position_file,'drone_movement.npy'), drone_movement_list)
        if self.verbose:
            print('Image extraction finished')



    def track(self):
        
        self._load_graph()
        self._load_label_map() 
        file_list = self._get_files()
        if self.create_video:
            self.out = self._initialize_video(self.video_file_output, self.video_width, self.video_height)
        self._label_images(file_list)

