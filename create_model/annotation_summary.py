import os
import glob
import xml.etree.ElementTree as ET

def get_annotation_summary(folder):
    animal_count = {}

    training_xmls = glob.glob(folder + '/annotations/*.xml')
    print('There are', len(training_xmls), 
        '.xml annotation files in', folder, '.')

    num_other_frames = 0
    for xml in training_xmls:
        tree = ET.parse(xml)
        root = tree.getroot()

        for animal in root.findall('object'):
            name = animal.find('name').text
    #         name = name.split('_')
    #         animal_name = name[0]
            animal_name = name
            if animal_name == 'other':
                num_other_frames += 1
            if animal_name in animal_count:
                animal_count[animal_name] += 1
            else:
                animal_count[animal_name] = 1
    print(animal_count)
    print('other', num_other_frames)
    print(animal_count)
    return(animal_count)


