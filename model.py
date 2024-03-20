import torch.nn as nn
from functions import ReverseLayerF
import numpy as np



class CNNModel(nn.Module):

    def __init__(self,layers, dimchange_multiplier,m_or_n_change):
        super(CNNModel, self).__init__()

        dim_conv = int(64 * np.sqrt(dimchange_multiplier))
        # dim_conv = 64
        # ks = int( 5 *dimchange_multiplier)
        # padding_conv1 = int((ks-5)/2)
        # padding_convs = int((ks-1)/2)
        ks = 5
        padding_convs = 2
        padding_conv1 = 0
        layers_feature = layers
        if m_or_n_change == "mchange":
            dim_lin_class = int(100 * dimchange_multiplier)
            layers_class = layers

            dim_lin_domain = 100
            layers_domain = 2

        if m_or_n_change == "nchange":
            dim_lin_class = 100
            layers_class = 3

            dim_lin_domain = int(100 * dimchange_multiplier)
            layers_domain = layers            
        

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, dim_conv, kernel_size=ks,padding=padding_conv1))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(dim_conv))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        for ii in range(layers_feature-2):
            self.feature.add_module(f'f_conv{2+ii}', nn.Conv2d(dim_conv,dim_conv,padding = padding_convs,kernel_size=ks))
            self.feature.add_module(f'f_bn{2+ii}', nn.BatchNorm2d(dim_conv))
            self.feature.add_module(f'f_relu{2+ii}', nn.ReLU(True))
        self.feature.add_module(f'f_conv{layers_feature}', nn.Conv2d(dim_conv, 50, kernel_size=ks,padding=padding_conv1))
        self.feature.add_module(f'f_bn{layers_feature}', nn.BatchNorm2d(50))
        self.feature.add_module(f'f_drop{layers_feature}', nn.Dropout2d())
        self.feature.add_module(f'f_pool{layers_feature}', nn.MaxPool2d(2))
        self.feature.add_module(f'f_relu{layers_feature}', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, dim_lin_class))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(dim_lin_class))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        for ii in range(layers_class-2):
            self.class_classifier.add_module(f'c_fc{2+ii}', nn.Linear(dim_lin_class, dim_lin_class))
            self.class_classifier.add_module(f'c_bn{2+ii}', nn.BatchNorm1d(dim_lin_class))
            self.class_classifier.add_module(f'c_relu{2+ii}', nn.ReLU(True))
        self.class_classifier.add_module(f'c_fc{layers_class}', nn.Linear(dim_lin_class, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, dim_lin_domain))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(dim_lin_domain))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        for ii in range(layers_domain-2): 
            self.domain_classifier.add_module(f'd_fc{2+ii}', nn.Linear(dim_lin_domain, dim_lin_domain)) 
            self.domain_classifier.add_module(f'd_bn{2+ii}', nn.BatchNorm1d(dim_lin_domain))
            self.domain_classifier.add_module(f'd_relu{2+ii}', nn.ReLU(True))
        self.domain_classifier.add_module(f'd_fc{layers_domain}', nn.Linear(dim_lin_domain, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
