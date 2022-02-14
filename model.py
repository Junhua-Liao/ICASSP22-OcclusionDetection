import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class OcclusionDetection_ICASSP():

    def __init__(self, model_path):
        self.__device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.__occlusion_resize_height = 128
        self.__occlusion_resize_width = 171
        self.__occlusion_detection_model = ICASSP()
        checkpoint = torch.load(model_path, map_location = lambda storage, loc: storage)
        self.__occlusion_detection_model.load_state_dict(checkpoint['state_dict'])
        self.__occlusion_detection_model.to(self.__device)
        self.__occlusion_detection_model.eval()
        
    def pretreatment(self, frames):
        buffer = np.empty((len(frames), self.__occlusion_resize_height, self.__occlusion_resize_width, 3), np.dtype('float32'))

        for i, frame in enumerate(frames):
            buffer[i] = cv2.resize(frame, (self.__occlusion_resize_width, self.__occlusion_resize_height)) / 255.
        
        return torch.from_numpy(buffer.transpose((3, 0, 1, 2))) 

    def occlusion_detection(self, frames):
        inputs = self.pretreatment(frames)
        inputs = torch.reshape(inputs, (1, 3, len(frames), self.__occlusion_resize_height, self.__occlusion_resize_width))
        inputs = Variable(inputs).to(self.__device)

        with torch.no_grad():
            outputs = self.__occlusion_detection_model(inputs)
        
        return outputs.tolist()
        

class SAT_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAT_Module, self).__init__()

        self.inter_channels = out_channels // 8
        self.relu = nn.ReLU()

        self.route_s_1_1 = nn.Conv3d(in_channels, self.inter_channels, kernel_size = (1, 1, 1), padding = (0, 0, 0))
        self.route_s_1_2 = nn.Conv3d(self.inter_channels, out_channels // 4, kernel_size = (1, 3, 3), padding = (0, 1, 1))

        self.route_s_2_1 = nn.Conv3d(in_channels, self.inter_channels, kernel_size = (1, 1, 1), padding = (0, 0, 0))
        self.route_s_2_2 = nn.Conv3d(self.inter_channels, out_channels // 4, kernel_size = (1, 5, 5), padding = (0, 2, 2))

        self.route_t_3_1 = nn.Conv3d(in_channels, self.inter_channels, kernel_size = (1, 1, 1), padding = (0, 0, 0))
        self.route_t_3_2 = nn.Conv3d(self.inter_channels, out_channels // 4, kernel_size = (3, 1, 1), padding = (1, 0, 0))

        self.route_t_4_1 = nn.Conv3d(in_channels, self.inter_channels, kernel_size = (1, 1, 1), padding = (0, 0, 0))
        self.route_t_4_2 = nn.Conv3d(self.inter_channels, out_channels // 4, kernel_size = (5, 1, 1), padding = (2, 0, 0))


    def forward(self, x):

        # Path I
        x1 = self.relu(self.route_s_1_1(x))
        x1 = self.relu(self.route_s_1_2(x1))

        # Path II
        x2 = self.relu(self.route_s_2_1(x))
        x2 = self.relu(self.route_s_2_2(x2))

        # Path III
        x3 = self.relu(self.route_t_3_1(x))
        x3 = self.relu(self.route_t_3_2(x3))

        # Path IV
        x4 = self.relu(self.route_t_4_1(x))
        x4 = self.relu(self.route_t_4_2(x4))

        x_final = torch.cat([x1, x2, x3, x4], 1)
        
        return x_final


class ICASSP(nn.Module):
    def __init__(self):
        super(ICASSP, self).__init__()
        
        self.conv1 = SAT_Module(3, 64)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))

        self.conv2 = SAT_Module(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        
        self.conv3 = SAT_Module(128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))

        self.conv4 = SAT_Module(256, 512)
        self.pool4 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))

        self.conv5 = SAT_Module(512, 1024)
        self.pool5 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))

        self.conv6 = SAT_Module(1024, 2048)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc6 = nn.Linear(2048, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 2)
        
        self.dropout = nn.Dropout(p = 0.5)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
            

    def forward(self, x):

        x = self.relu(self.conv1(x))  # 1st SAT Module
        x = self.pool1(x)

        x = self.relu(self.conv2(x))  # 2nd SAT Module
        x = self.pool2(x)
        
        x = self.relu(self.conv3(x))  # 3rd SAT Module
        x = self.pool3(x)
        
        x = self.relu(self.conv4(x))  # 4th SAT Module
        x = self.pool4(x)

        x = self.relu(self.conv5(x))  # 5th SAT Module
        x = self.pool5(x)

        x = self.relu(self.conv6(x))  # 6th SAT Module
        x = x.squeeze()
        x = self.maxpool(x)
        x = torch.transpose(x, 0, 1)
        x = torch.reshape(x, (-1, 2048))

        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        x = nn.functional.softmax(x, dim = 1)
        
        return x[:,1]