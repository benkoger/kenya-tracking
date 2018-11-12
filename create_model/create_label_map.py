import os

def create_label_map(folder, class_list):
    file_path = os.path.join(folder, 'label_map.pbtxt')
    with open(file_path, 'w') as out_file:
        for class_num, class_name in enumerate(class_list):
            print('item {', file=out_file)
            print('    id: {}'.format(class_num + 1), file=out_file)
            print('    name: \'{}\''.format(class_name), file=out_file)
            print('}', file=out_file)
            print(' ', file=out_file)
    return file_path
