import os
import cv2
import numpy as np
from xml.etree import ElementTree as ET
from model import OcclusionDetection_ICASSP


def LoadData(file_dir):

    buffer = []
    
    for frame in sorted(os.listdir(file_dir)):
        
        buffer.append(cv2.imread(os.path.join(file_dir, frame)).astype(np.float64))

    return buffer

def LoadLabel(file_dir):
    
    labels = []

    for xml in sorted(os.listdir(file_dir)):   
        
        tree = ET.parse(os.path.join(file_dir, xml))
        root = tree.getroot()

        if len(root.findall('object')) == 0:
            labels.append(0)
        else:
            labels.append(1)
    
    return labels


if __name__ == '__main__':

    batch_size = 8
    entire_label = []
    entire_predict = []
    dataset_path = 'data/OcclusionDataSet-MM20/'

    Data_folder = os.path.join(dataset_path, 'Data', 'test')
    Annotation_folder = os.path.join(dataset_path, 'Annotations', 'test')

    data_list = []
    for data in os.listdir(Data_folder):
        data_list.append(os.path.join(Data_folder, data))
    data_list = sorted(data_list)

    label_list = []
    for label in os.listdir(Annotation_folder):
        label_list.append(os.path.join(Annotation_folder, label))
    label_list = sorted(label_list)

    method = OcclusionDetection_ICASSP(model_path = 'weights/ICASSP_Model.pth.tar')
    
    for index in range(len(data_list)):

        print('{}: {}/{}'.format(data_list[index], index + 1, len(data_list)))
        
        inputs = LoadData(data_list[index])
        labels = LoadLabel(label_list[index])

        entire_label = entire_label + labels

        while len(inputs) >= batch_size:

            model_input = inputs[:batch_size]
            result = method.occlusion_detection(model_input)

            for i in range(batch_size):
                if result[i] >= 0.5:
                    entire_predict.append(1)
                else:
                    entire_predict.append(0)
            
            model_input.clear()
            del inputs[:batch_size]

        if len(inputs) != 0:
            temp_length = len(inputs)
            while len(inputs) != batch_size:
                inputs.append(inputs[-1])

            result = method.occlusion_detection(inputs)

            for i in range(temp_length):
                if result[i] >= 0.5:
                    entire_predict.append(1)
                else:
                    entire_predict.append(0)

            inputs.clear()


    print(sum(np.array(entire_predict) == np.array(entire_label)) / len(entire_label))
