# Create Date    : 2025/09/15
# Modify Date    : 2025/12/16
# Python Version : 3.12.7-amd64
# Cuda   Version : 12.4.0_551.61_windows
# Cudnn  Version : 9.1.1_windows
# CPU    Version : Intel(R) Core(TM) i7-14700F 2.10 GHz
# GPU    Version : NVIDIA GeForce RTX 4090 D
############################################################################################################################
import numpy as np              # numpy: 1.26.3
import os
import _pickle as pickle
import FSNanalysis as fsn       # custom module: FSNanalysis.py
import matplotlib.pyplot as plt # matplotlib: 3.9.2
import sklearn                  # sklearn2: 0.0.13
import torch                    # torch: 2.6.0+cu124
from torch import nn
import time

def flip(data, label, tick):
    data_flip                   = []
    label_flip                  = []
    for d, l in zip(data, label):
        data_flip.append([feature for feature in d])
        data_flip[-1][tick]    *= 0.5
        data_flip.append([feature for feature in d])
        data_flip[-1][tick]     = (1 - data_flip[-1][tick]) * 0.5 + 0.5
        label_flip.append(l)
        label_flip.append(l)
    return np.array(data_flip, dtype = np.float32), np.array(label_flip, dtype = np.float32)

class FCmodel2(nn.Module):
    def __init__(self, nodes):
        super(FCmodel2, self).__init__()
        self.fc1                = nn.Linear(nodes[0], nodes[1])
        self.fc2                = nn.Linear(nodes[1], nodes[1])
        self.fc3                = nn.Linear(nodes[1], nodes[2])
        self.elu                = nn.ELU()
    def forward(self, input):
        tem                     = self.fc1(input)
        tem                     = self.elu(tem)
        tem                     = self.fc2(tem)
        tem                     = self.elu(tem)
        tem                     = self.fc3(tem)
        output                  = self.elu(tem)
        return output

class FCmodel3(nn.Module):
    def __init__(self, nodes):
        super(FCmodel3, self).__init__()
        self.fc1                = nn.Linear(nodes[0], nodes[1])
        self.fc2                = nn.Linear(nodes[1], nodes[1])
        self.fc3                = nn.Linear(nodes[1], nodes[1])
        self.fc4                = nn.Linear(nodes[1], nodes[2])
        self.elu                = nn.ELU()
    def forward(self, input):
        tem                     = self.fc1(input)
        tem                     = self.elu(tem)
        tem                     = self.fc2(tem)
        tem                     = self.elu(tem)
        tem                     = self.fc3(tem)
        tem                     = self.elu(tem)
        tem                     = self.fc4(tem)
        output                  = self.elu(tem)
        return output

class FCmodel4(nn.Module):
    def __init__(self, nodes):
        super(FCmodel4, self).__init__()
        self.fc1                = nn.Linear(nodes[0], nodes[1])
        self.fc2                = nn.Linear(nodes[1], nodes[1])
        self.fc3                = nn.Linear(nodes[1], nodes[1])
        self.fc4                = nn.Linear(nodes[1], nodes[1])
        self.fc5                = nn.Linear(nodes[1], nodes[2])
        self.elu                = nn.ELU()
    def forward(self, input):
        tem                     = self.fc1(input)
        tem                     = self.elu(tem)
        tem                     = self.fc2(tem)
        tem                     = self.elu(tem)
        tem                     = self.fc3(tem)
        tem                     = self.elu(tem)
        tem                     = self.fc4(tem)
        tem                     = self.elu(tem)
        tem                     = self.fc5(tem)
        output                  = self.elu(tem)
        return output

if __name__ == '__main__':
    # Load Data
    fileList                    = os.listdir('.')
    Data                        = sklearn.datasets.fetch_california_housing()
    data                        = np.array(Data['data'], dtype = np.float32)
    # MedInc                    [15.0001, 0.4999]
    data[: , 0]                 = data[: , 0] / np.max(data[: , 0])
    # HouseAge                  [52, 1]
    data[: , 1]                 = np.array([value if value < 50 else 50 for value in data[: , 1]])[: ] / 50
    # AveRooms                  [141.9090909090909, 0.8461538461538461]
    data[: , 2]                 = data[: , 2] / np.max(data[: , 2])
    # AveBedrms                 [34.06666666666667, 0.3333333333333333]
    data[: , 3]                 = data[: , 3] / np.max(data[: , 3])
    # Population                [35682.0 3.0]
    data[: , 4]                 = data[: , 4] / np.max(data[: , 4])
    # AveOccup                  [1243.3333333333333, 0.6923076923076923]
    data[: , 5]                 = data[: , 5] / np.max(data[: , 5])
    # Latitude                  [41.95, 32.54]
    data[: , 6]                 = (data[: , 6] - np.min(data[: , 6])) / (np.max(data[: , 6]) - np.min(data[: , 6]))
    # Longitude                 [-114.31, -124.35]
    data[: , 7]                 = (data[: , 7] - np.min(data[: , 7])) / (np.max(data[: , 7]) - np.min(data[: , 7]))
    # Label                     [5.00001, 0.14999]
    Data['target'] = np.array([[value] if value < 5 else [5] for value in Data['target']], dtype=np.float32) / 5
    shape                       = np.shape(data)
    # Slice
    Data['data']                = np.zeros((shape[0], 2), dtype = np.float32)
    Data['data'][: , 0]         = data[: , 6]
    Data['data'][: , 1]         = data[: , 7]
    feature_num                 = np.shape(Data['data'][0])[0]
    domain                      = [[0, 1] for feature in range(feature_num)]
    # Parameters (Modifiable hyperparameters)
    sampleRate                  = [1000 for feature in range(feature_num)]
    blocks                      = [[5 for feature in range(feature_num)] for i in range(3)]
    epochs                      = 20
    epochs_fc                   = {'1000': 200, '2000': 150, '4000': 100}
    iid_num                     = 1
    iid_num_fc                  = 20
    lossFunction                = nn.MSELoss()
    lossFunction_test           = nn.L1Loss()
    nodes_fc                    = [1000, 2000, 4000]
    fc_enable                   = True
    """
    # DataGraph
    fig                         = plt.figure(figsize = (7, 4.5))
    plt.scatter(Data['data'].T[0], Data['data'].T[1], Data['target'].T[0], c = Data['target'].T[0], cmap = plt.cm.seismic)
    plt.title('Geographical Distribution')
    plt.ylabel('Longitude[-114.31, -124.35]')
    plt.xlabel('Latitude[41.95, 32.54]')
    plt.grid(True)
    plt.show()
    time.sleep(0.1)
    """
    fsn_train                   = []
    fsn_test                    = []
    fsn_valid                   = []
    # Acceleration device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Results
    fileList                    = os.listdir('.')
    if 'FSNPerformance' not in fileList:
        os.mkdir('FSNPerformance')
    # i.i.d
    for iid in range(1, iid_num + 1):
        current_time            = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        path                    = 'FSNPerformance/' + current_time
        os.mkdir(path)
        # Divide
        tick                    = np.random.rand(shape[0])
        tick                    = np.argsort(tick)
        data                    = np.zeros((shape[0], feature_num), dtype = np.float32)
        label                   = np.zeros((shape[0], 1), dtype = np.float32)
        for i, t in enumerate(tick):
            data[t, :]          = Data['data'][i, :]
            label[t, 0]         = Data['target'][i, 0]
        data_train              = {'input': data[: int(shape[0] * 0.8)], 'output': label[: int(shape[0] * 0.8)]}
        data_test               = {'input': data[int(shape[0] * 0.8): int(shape[0] * 0.9)], 'output': label[int(shape[0] * 0.8): int(shape[0] * 0.9)]}
        data_valid              = {'input': data[int(shape[0] * 0.9): ], 'output': label[int(shape[0] * 0.9): ]}
        ################################################################################################################
        # Joint
        data_train_fsn          = {'input': [], 'output': []}
        data_test_fsn           = {'input': [], 'output': []}
        data_valid_fsn          = {'input': [], 'output': []}
        for feature in range(feature_num):
            data_train_fsn['input'], data_train_fsn['output']   = flip(data_train['input'], data_train['output'], feature)
            data_test_fsn['input'] , data_test_fsn['output']    = flip(data_test['input'] , data_test['output'], feature)
            data_valid_fsn['input'], data_valid_fsn['output']   = flip(data_valid['input'], data_valid['output'], feature)
        # Complement
        data_complement         = [[]]
        for feature in range(feature_num):
            tem                 = []
            for d in data_complement:
                for i in range(11):
                    tem.append(d + [i / 10])
            data_complement     = tem
        data_complement         = [d for d in data_complement if 0.0 in d or 1.0 in d]
        data_train_fsn['input'] = np.array(list(data_train_fsn['input']) + data_complement, dtype = np.float32)
        data_train_fsn['output']= np.array(list(data_train_fsn['output']) + [[0] for d in data_complement], dtype = np.float32)
        # Generate network
        print(time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time())))
        print('############################################')
        print('#%4d/%4d (FSN)' % (iid, iid_num))
        print('Interpolation...         ', end = ' ')
        map_grid, X             = fsn.dataProcess_insert(data_train_fsn, sampleRate, domain, processes = 8)
        print('Done')
        print('Fast Fourier Transform...', end = ' ')
        spectrum, _             = fsn.dataProcess_FFT(map_grid, sampleRate)
        print('Done')
        print('FSN Generation...        ', end = ' ')
        model                   = fsn.FSN(spectrum, blocks, sampleRate, device, processes = 8)
        print('Done')
        # test
        print('Training...')
        dataloader_train        = fsn.dataTransform(data_train_fsn, batch_size = len(data_train_fsn), shuffle = True, device = device)
        dataloader_test         = fsn.dataTransform(data_test_fsn, batch_size = 1, shuffle = True, device = device)
        dataloader_valid        = fsn.dataTransform(data_valid_fsn, batch_size = 1, shuffle = True, device = device)
        print('#%4d/%4d (Training Set)' % (0, epochs), end = ' ')
        result, _               = fsn.test(model, dataloader_train, device, lossFunction = lossFunction_test, classifyEnable = False)
        fsn_train.append(result)
        print('           ( Validation )', end=' ')
        result, _               = fsn.test(model, dataloader_valid, device, lossFunction = lossFunction_test, classifyEnable = False)
        fsn_valid.append(result)
        print('           (Testing  Set)', end = ' ')
        result, _               = fsn.test(model, dataloader_test, device, lossFunction = lossFunction_test, classifyEnable = False)
        fsn_test.append(result)
        # train
        optimizer               = torch.optim.Adam(model.parameters(), lr = 1e-8)
        for epoch in range(1, epochs + 1):
            print('#%4d/%4d' % (epoch, epochs), end = ' ')
            fsn.train(model, dataloader_train, device, lossFunction, optimizer)
            print('(Training Set)', end = ' ')
            result, _           = fsn.test(model, dataloader_train, device, lossFunction = lossFunction_test, classifyEnable = False)
            fsn_train.append(result)
            print('           ( Validation )', end=' ')
            result, _           = fsn.test(model, dataloader_valid, device, lossFunction = lossFunction_test, classifyEnable = False)
            fsn_valid.append(result)
            print('           (Testing  Set)', end = ' ')
            result, _           = fsn.test(model, dataloader_test, device, lossFunction = lossFunction_test, classifyEnable = False)
            fsn_test.append(result)
        del model, optimizer
        with open(path + '/fsn_train.pkl', 'wb') as file:
            pickle.dump(fsn_train, file)
        with open(path + '/fsn_valid.pkl', 'wb') as file:
            pickle.dump(fsn_valid, file)
        with open(path + '/fsn_test.pkl', 'wb') as file:
            pickle.dump(fsn_test, file)
        ################################################################################################################
        if fc_enable:
            dataloader_train    = fsn.dataTransform(data_train, batch_size = 1024, shuffle = True, device = device)
            dataloader_test     = fsn.dataTransform(data_test, batch_size = 1, shuffle = True, device = device)
            dataloader_valid    = fsn.dataTransform(data_valid, batch_size = 1, shuffle = True, device = device)
            fc_train            = {}
            fc_test             = {}
            fc_valid            = {}
            nodes               = {}
            for node in nodes_fc:
                fc_train['%d'%node]             = []
                fc_test['%d'%node]              = []
                fc_valid['%d'%node]             = []
                nodes['%d'%node]                = [feature_num] + [node] + [1]
            for node in nodes.keys():
                for i in range(iid_num_fc):
                    fc_train[node].append([1])
                    fc_test[node].append([1])
                    fc_valid[node].append([1])
                    print(time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time())))
                    print('############################################')
                    print('#%4d/%4d (FC:%4d/%4d) Node Num: %4d * 2' %(iid, iid_num, i + 1, iid_num_fc, nodes[node][1]))
                    tem_train   = []
                    tem_test    = []
                    tem_valid   = []
                    model       = FCmodel2(nodes[node]).to(device)
                    optimizer   = torch.optim.Adam(model.parameters(), lr = 1e-6)
                    print('#%4d/%4d (Training Set)' % (0, epochs_fc[node]), end = ' ')
                    result, _   = fsn.test(model, dataloader_train, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_train.append(result)
                    print('           ( Validation )', end=' ')
                    result, _   = fsn.test(model, dataloader_valid, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_valid.append(result)
                    print('           (Testing  Set)', end = ' ')
                    result, _   = fsn.test(model, dataloader_test, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_test.append(result)
                    for epoch in range(1, epochs_fc[node] + 1):
                        if epoch % 40 == 0:
                            print('#%4d/%4d' % (epoch, epochs_fc[node]), end=' ')
                            printEnable         = True
                        else:
                            printEnable         = False
                        fsn.train(model, dataloader_train, device, lossFunction, optimizer)
                        if epoch % 40 == 0:
                            print('(Training Set)', end = ' ')
                        result, _               = fsn.test(model, dataloader_train, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_train.append(result)
                        if epoch % 40 == 0:
                            print('           ( Validation )', end = ' ')
                        result, _               = fsn.test(model, dataloader_valid, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_valid.append(result)
                        if epoch % 40 == 0:
                            print('           (Testing  Set)', end = ' ')
                        result, _               = fsn.test(model, dataloader_test, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_test.append(result)
                    del model, optimizer
                    if min(fc_valid[node][-1]) > min(tem_valid):
                        fc_valid[node][-1]      = min(tem_valid)
                        fc_train[node][-1]      = tem_train[np.where(tem_valid == np.min(tem_valid))[0][0]]
                        fc_test[node][-1]       = tem_test[np.where(tem_valid == np.min(tem_valid))[0][0]]
            print(time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time())))
            with open(path + '/fc2_train.pkl', 'wb') as file:
                pickle.dump(fc_train, file)
            with open(path + '/fc2_valid.pkl', 'wb') as file:
                pickle.dump(fc_valid, file)
            with open(path + '/fc2_test.pkl', 'wb') as file:
                pickle.dump(fc_test, file)
            ################################################################################################################
            fc_train            = {}
            fc_test             = {}
            fc_valid            = {}
            nodes               = {}
            for node in nodes_fc:
                fc_train['%d' % node]           = []
                fc_test['%d' % node]            = []
                fc_valid['%d' % node]           = []
                nodes['%d' % node]              = [feature_num] + [node] + [1]
            for node in nodes.keys():
                for i in range(iid_num_fc):
                    fc_train[node].append([1])
                    fc_test[node].append([1])
                    fc_valid[node].append([1])
                    print(time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time())))
                    print('############################################')
                    print('#%4d/%4d (FC:%4d/%4d) Node Num: %4d * 3' % (iid, iid_num, i + 1, iid_num_fc, nodes[node][1]))
                    tem_train   = []
                    tem_test    = []
                    tem_valid   = []
                    model       = FCmodel3(nodes[node]).to(device)
                    optimizer   = torch.optim.Adam(model.parameters(), lr = 1e-6)
                    print('#%4d/%4d (Training Set)' % (0, epochs_fc[node]), end = ' ')
                    result, _   = fsn.test(model, dataloader_train, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_train.append(result)
                    print('           ( Validation )', end=' ')
                    result, _   = fsn.test(model, dataloader_valid, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_valid.append(result)
                    print('           (Testing  Set)', end=' ')
                    result, _   = fsn.test(model, dataloader_test, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_test.append(result)
                    for epoch in range(1, epochs_fc[node] + 1):
                        if epoch % 40 == 0:
                            print('#%4d/%4d' % (epoch, epochs_fc[node]), end = ' ')
                            printEnable         = True
                        else:
                            printEnable         = False
                        fsn.train(model, dataloader_train, device, lossFunction, optimizer)
                        if epoch % 40 == 0:
                            print('(Training Set)', end = ' ')
                        result, _               = fsn.test(model, dataloader_train, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_train.append(result)
                        if epoch % 40 == 0:
                            print('           ( Validation )', end = ' ')
                        result, _               = fsn.test(model, dataloader_valid, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_valid.append(result)
                        if epoch % 40 == 0:
                            print('           (Testing  Set)', end = ' ')
                        result, _               = fsn.test(model, dataloader_test, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_test.append(result)
                    del model, optimizer
                    if min(fc_valid[node][-1]) > min(tem_valid):
                        fc_valid[node][-1]      = min(tem_valid)
                        fc_train[node][-1]      = tem_train[np.where(tem_valid == np.min(tem_valid))[0][0]]
                        fc_test[node][-1]       = tem_test[np.where(tem_valid == np.min(tem_valid))[0][0]]
            print(time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time())))
            with open(path + '/fc3_train.pkl', 'wb') as file:
                pickle.dump(fc_train, file)
            with open(path + '/fc3_valid.pkl', 'wb') as file:
                pickle.dump(fc_valid, file)
            with open(path + '/fc3_test.pkl', 'wb') as file:
                pickle.dump(fc_test, file)
            ################################################################################################################
            fc_train            = {}
            fc_test             = {}
            fc_valid            = {}
            nodes               = {}
            for node in nodes_fc:
                fc_train['%d' % node]           = []
                fc_test['%d' % node]            = []
                fc_valid['%d' % node]           = []
                nodes['%d' % node]              = [feature_num] + [node] + [1]
            for node in nodes.keys():
                for i in range(iid_num_fc):
                    fc_train[node].append([1])
                    fc_test[node].append([1])
                    fc_valid[node].append([1])
                    print(time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time())))
                    print('############################################')
                    print('#%4d/%4d (FC:%4d/%4d) Node Num: %4d * 4' % (iid, iid_num, i + 1, iid_num_fc, nodes[node][1]))
                    tem_train   = []
                    tem_test    = []
                    tem_valid   = []
                    model       = FCmodel4(nodes[node]).to(device)
                    optimizer   = torch.optim.Adam(model.parameters(), lr=1e-6)
                    print('#%4d/%4d (Training Set)' % (0, epochs_fc[node]), end = ' ')
                    result, _   = fsn.test(model, dataloader_train, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_train.append(result)
                    print('           ( Validation )', end = ' ')
                    result, _   = fsn.test(model, dataloader_valid, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_valid.append(result)
                    print('           (Testing  Set)', end = ' ')
                    result, _   = fsn.test(model, dataloader_test, device, lossFunction = lossFunction_test, classifyEnable = False)
                    tem_test.append(result)
                    for epoch in range(1, epochs_fc[node] + 1):
                        if epoch % 40 == 0:
                            print('#%4d/%4d' % (epoch, epochs_fc[node]), end = ' ')
                            printEnable         = True
                        else:
                            printEnable         = False
                        fsn.train(model, dataloader_train, device, lossFunction, optimizer)
                        if epoch % 40 == 0:
                            print('(Training Set)', end = ' ')
                        result, _               = fsn.test(model, dataloader_train, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_train.append(result)
                        if epoch % 40 == 0:
                            print('           ( Validation )', end = ' ')
                        result, _               = fsn.test(model, dataloader_valid, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_valid.append(result)
                        if epoch % 40 == 0:
                            print('           (Testing  Set)', end = ' ')
                        result, _               = fsn.test(model, dataloader_test, device, lossFunction = lossFunction_test, classifyEnable = False, printEnable = printEnable)
                        tem_test.append(result)
                    del model, optimizer
                    if min(fc_valid[node][-1]) > min(tem_valid):
                        fc_valid[node][-1]      = min(tem_valid)
                        fc_train[node][-1]      = tem_train[np.where(tem_valid == np.min(tem_valid))[0][0]]
                        fc_test[node][-1]       = tem_test[np.where(tem_valid == np.min(tem_valid))[0][0]]
            print(time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time())))
            with open(path + '/fc4_train.pkl', 'wb') as file:
                pickle.dump(fc_train, file)
            with open(path + '/fc4_valid.pkl', 'wb') as file:
                pickle.dump(fc_valid, file)
            with open(path + '/fc4_test.pkl', 'wb') as file:
                pickle.dump(fc_test, file)