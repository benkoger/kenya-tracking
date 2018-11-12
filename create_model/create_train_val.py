'''
takes in a directory and creates a train_val.txt file like one
gets in the pets data set and is nessisary for create_~_tf_record.py
in tensorflow/models/object_detection 
'''

import os
from tensorflow.python.platform import gfile




def create_train_val(data_dir):
    output_file = os.path.join(data_dir, 'annotations', 'trainval.txt') 
    print(output_file)
#    for data_dir  in data_dirs:
    print(data_dir)

    files = [x[2] for x in gfile.Walk(data_dir)]

    with open(output_file, 'w') as f:
        for file_name in files[0]:
            f.write(os.path.splitext(file_name)[0] + ' \n')


