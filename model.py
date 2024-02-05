import torch.nn as nn
from functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self,layers):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))

        for ii in range(layers-2):
            self.feature.add_module(f'f_conv{2+ii}', nn.Conv2d(64,64,padding = 2,kernel_size=5))
            self.feature.add_module(f'f_bn{2+ii}', nn.BatchNorm2d(64))
            self.feature.add_module(f'f_relu{2+ii}', nn.ReLU(True))
 
        self.feature.add_module(f'f_conv{layers}', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module(f'f_bn{layers}', nn.BatchNorm2d(50))
        self.feature.add_module(f'f_drop{layers}', nn.Dropout2d())
        self.feature.add_module(f'f_pool{layers}', nn.MaxPool2d(2))
        self.feature.add_module(f'f_relu{layers}', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        for ii in range(layers-2): 
            self.domain_classifier.add_module(f'd_fc{2+ii}', nn.Linear(100, 100)) 
            self.domain_classifier.add_module(f'd_bn{2+ii}', nn.BatchNorm1d(100))
            self.domain_classifier.add_module(f'd_relu{2+ii}', nn.ReLU(True))
        self.domain_classifier.add_module(f'd_fc{layers}', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

class CNNModel_old(nn.Module):

    def __init__(self,layers):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        for ii in range(layers-2):
            self.feature.add_module(f'f_conv{2+ii}', nn.Conv2d(64,64,padding = 2,kernel_size=5))
            self.feature.add_module(f'f_bn{2+ii}', nn.BatchNorm2d(64))
            self.feature.add_module(f'f_relu{2+ii}', nn.ReLU(True))
        self.feature.add_module(f'f_conv{layers}', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module(f'f_bn{layers}', nn.BatchNorm2d(50))
        self.feature.add_module(f'f_drop{layers}', nn.Dropout2d())
        self.feature.add_module(f'f_pool{layers}', nn.MaxPool2d(2))
        self.feature.add_module(f'f_relu{layers}', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 5 * 5, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 5 * 5, 100)) # bak buralar değişiyor 50*4*4 ten 50*5*5 oluyorlar.
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        for ii in range(layers-2): 
            self.domain_classifier.add_module(f'd_fc{2+ii}', nn.Linear(100, 100)) 
            self.domain_classifier.add_module(f'd_bn{2+ii}', nn.BatchNorm1d(100))
            self.domain_classifier.add_module(f'd_relu{2+ii}', nn.ReLU(True))
        self.domain_classifier.add_module(f'd_fc{layers}', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, gamma, ms_or_mt):
        bs = input_data.data.shape[0]
        Ms_or_Mt = int(bs*ms_or_mt)

        input_data = input_data.expand(bs, 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 5 * 5) # bak burada da var bir değişikliki
        reverse_feature = ReverseLayerF.apply(feature, gamma)
        class_output = self.class_classifier(feature[0:Ms_or_Mt,:])
