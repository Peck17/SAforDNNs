# Create Date    : 2025/03/13
# Modify Date    : 2025/09/30
# Python Version : 3.12.7-amd64
# Cuda   Version : 12.4.0_551.61_windows
# Cudnn  Version : 9.1.1_windows
# CPU    Version : Intel(R) Core(TM) i7-14700F 2.10 GHz
# GPU    Version : NVIDIA GeForce RTX 4090 D
########################################################################################################################
import numpy as np              # numpy: 1.26.3
import sympy as sp              # sympy: 1.13.1
import torch                    # torch: 2.6.0+cu124
from torch import nn

import time
import copy
import queue
import threading
import concurrent.futures
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator # scipy: 1.14.1

# Interpolation sampling and zeroing of data
########################################################################################################################
class selfIter(object):
    def __init__(self, obj, length, copyEnable = False):
        self.obj = obj
        self.length             = length
        self.current            = 0
        self.copyEnable         = copyEnable
    def __iter__(self):
        return self
    def __next__(self):
        if self.current < self.length:
            self.current       += 1
            if self.copyEnable:
                return copy.deepcopy(self.obj)
            else:
                return self.obj
        else:
            raise StopIteration
class paraList(object):
    def __init__(self, function, coordinate):
        self.function           = function
        self.coordinate         = coordinate
        self.length             = len(function)
        self.current            = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.current < self.length:
            self.current       += 1
            return self.function[self.current - 1], next(self.coordinate)
        else:
            raise StopIteration
def thread_mark(param):
    result                      = []
    coor_center                 = np.floor(param[0] * param[1]).astype(np.int32)
    for c in param[2]:
        coor                    = coor_center + c
        if all(coor[i] >= param[3][i][0] for i in range(param[4])) and all(coor[i] <= param[3][i][1] for i in range(param[4])):
            result.append(tuple(coor))
    return result
def thread_interpolator(q, label, inputs, outputs, state = 0):
    if state == 0:
        q.put([label, LinearNDInterpolator(inputs, outputs)])
    else:
        q.put([label, NearestNDInterpolator(inputs, outputs)])
    return
def thread_compute(param):
    try:
        result                  = param[1]([param[0]])[0]
    except:
        result                  = 0
    return result
def multiCompute(param):
    num                         = len(param[1])
    epochs                      = int(np.ceil(num / 1024))
    results                     = []
    for epoch in range(epochs):
        if epoch == epochs - 1:
            p                   = param[1][epoch * 1024: ]
        else:
            p                   = param[1][epoch * 1024: epoch * 1024 + 1024]
        with ThreadPool(24) as pool:
            result              = pool.map(thread_compute, paraList(p, selfIter(param[0], len(param[1]), copyEnable = False)), 1)
        results                += [r for r in result]
    return results
def dataProcess_insert(data, sampleRate, domain, radius = 0, processes = None):
    """
    dataProcess_insert(data, sampleRate, domain, radius = 0, processes = None)
    Function Description:
        The data is organized through interpolation to make the sample intervals the same for calculations such as fast Fourier transform.
    Parameter Description:
        data                    Sample                          —— {'input': [Sample size, Input dimension n], 'output': [Sample size, Output dimension m]}
        sampleRate              Sampling frequency(Hz)          —— [sampleRate_1, sampleRate_2, …… , sampleRate_n]
        domain                  Sample domain(s)                —— [[domain_1[0], domain_1[1]], …… ,  [domain_n[0], domain_n[1]]] It is required that the domain of definition includes zero points
        radius                  Neighborhood radius             —— int Determine the range of the difference. Default 0 Global interpolation.
        processes               Number of processes             —— int (Default None - os.cpu_count())
    Return Description:
        map_grid                Time-domain data graph          —— [N_1, N_2, …… , N_n]
        X                       Time-domain coordinates(s)      —— [x_1, x_2, …… , x_n]
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        import matplotlib.pyplot as plt
        if __name__ == '__main__':
            num                 = 1000
            sampleRate          = [1000, 1000]
            domain              = [[0, 0.2], [0, 0.4]]
            x                   = np.random.rand(num) * domain[0][1]
            y                   = np.random.rand(num) * domain[1][1]
            g_1                 = np.sin(2 * x * np.pi / domain[0][1]) + np.cos(2 * y * np.pi / domain[1][1])
            g_2                 = np.cos(2 * x * np.pi / domain[0][1]) + np.sin(2 * y * np.pi / domain[1][1])
            data                = {'input': [], 'output': []}
            for n in range(num):
                data['input'].append([x[n], y[n]])
                data['output'].append([g_1[n], g_2[n]])
            map_grid, X         = fsn.dataProcess_insert(data, sampleRate, domain, radius=20)
            (X, Y)              = np.meshgrid(X[1], X[0])
            fig                 = plt.figure()
            ax                  = plt.axes(projection='3d')
            ax.scatter(y, x, g_1)
            plt.title('Real Value of Output#1')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection='3d')
            ax.plot_surface(X, Y, map_grid[:, :, 0])
            plt.title('Interpolation of Output#1')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection='3d')
            ax.scatter(y, x, g_2)
            plt.title('Real Value of Output#2')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection='3d')
            ax.plot_surface(X, Y, map_grid[:, :, 1])
            plt.title('Interpolation of Output#2')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
    """
    # Constrain the upper limit of the number of tasks in the interpolation thread pool
    para_mark                   = 4096
    input_dimension             = np.shape(data['input'])
    output_dimension            = np.shape(data['output'])
    feature_num                 = len(sampleRate)
    # The number of grids
    grid_num                    = [int((domain[i][1] - domain[i][0]) * sampleRate[i] + 1) for i in range(input_dimension[1])]
    # Initialize the zero-complementing dataset
    data_sup                    = {'input': [], 'output': []}
    for d, label in zip(data['input'], data['output']):
        data_sup['input'].append([i for i in d])
        data_sup['output'].append([i for i in label])
    # Initialize the time-domain graph
    map_grid                    = np.zeros((grid_num[-1], output_dimension[1]), dtype = np.float32)
    for feature in range(2, input_dimension[1] + 1):
        map_grid                = np.array([map_grid], dtype = np.float32).repeat(grid_num[input_dimension[1] - feature], axis = 0)
    # Find the default position
    if radius == 0:
        coordinate              = np.array(np.where(map_grid[..., 0] == 0), dtype = np.int32).T
    else:
        s                       = np.array(sampleRate, dtype = np.float32)
        domain_s                = [[domain[i][0] * sampleRate[i], domain[i][1] * sampleRate[i]] for i in range(feature_num)]
        # Determine the neighborhood
        coor_complement         = [[]]
        for feature in range(feature_num):
            tem                 = []
            for c in coor_complement:
                for tick in range(-radius, radius + 1):
                    coor        = c + [tick]
                    if sum(coor) <= radius:
                        tem.append(c + [tick])
            coor_complement     = tem
        coor_complement         = np.array(coor_complement, dtype=np.int32)
        # Mark the neighborhood
        data_num                = len(data['input'])
        group_num               = int(np.ceil(data_num / para_mark))
        with concurrent.futures.ThreadPoolExecutor(processes) as pool:
            futures             = {}
            # Group to generate thread pools for solution
            for group in range(group_num):
                start           = group * para_mark
                if group == group_num - 1:
                    target      = data['input'][start:]
                else:
                    target      = data['input'][start: start + para_mark]
                for i, sample in enumerate(target):
                    future      = pool.submit(thread_mark, [sample, s, coor_complement, domain_s, feature_num])
                    futures[future]                             = c
                for future in concurrent.futures.as_completed(futures):
                    for c in future.result():
                        map_grid[c]                             = 1
        # Supplement zero samples
        coordinate              = np.array(np.where(map_grid[..., 0] == 0), dtype=np.int32).T
        for c in coordinate:
            data_sup['input'].append([i for i in c])
            data_sup['output'].append([0 for i in data['output'][0]])
        coordinate              = np.array(np.where(map_grid[..., 0] == 1), dtype=np.int32).T
    coordinate_map              = coordinate / np.array([sampleRate], dtype=np.int32).repeat(len(coordinate), axis = 0)
    # Initialize the interpolation equation
    function                    = [0 for i in range(output_dimension[1])]
    # Calculate the interpolation function
    threads                     = []
    q = queue.Queue()
    for label in range(output_dimension[1]):
        threads.append(threading.Thread(target = thread_interpolator, args = (q, label, data['input'], np.array(data['output'])[:, label])))
    for thread in threads:
        thread.start()
    for label in range(output_dimension[1]):
        item                    = q.get()
        function[item[0]]       = item[1]
    for thread in threads:
        thread.join()
    # Assignment of time-domain graphs
    with Pool(processes = processes) as pool:
        results                 = pool.map(multiCompute, paraList(function, selfIter(coordinate_map, len(function), copyEnable = False)))
    for label, result in enumerate(results):
        for i, value in enumerate(result):
            map_grid[tuple(coordinate[i])][label]               = value
    # Calculate the time-domain coordinates
    X                           = [np.linspace(domain[i][0], domain[i][1], grid_num[i]) for i in range(input_dimension[1])]
    return map_grid, X

# Extract the data labels
########################################################################################################################
def valueDivide(data, labels):
    """
    valueDivide(data, labels):
    Function Description:
        Extract labels according to the instructions to form a new dataset.
    Parameter Description:
        data                    Sample                          —— {'input': [Sample size, Input dimension n], 'output': [Sample size, Output dimension m]}
        labels                  Labels Coordinate(Hz)           —— [coor_1, coor_2, …… , coor_k]
    Return Description:
        data                    Sample                          —— {'input': [Sample size, Input dimension n], 'output': [Sample size, Output dimension m]}
    Test Example:
        import FSNanalysis as fsn
        if __name__ == '__main__':
            data                = {'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'output': [[0, 1], [1, 0], [0, 1]]}
            data_slice          = fsn.valueDivide(data, [0])
            print(data_slice['input'])
            print(data_slice['output'])
            data_slice          = fsn.valueDivide(data, [0, 1])
            print(data_slice['input'])
            print(data_slice['output'])
    """
    result                      = {'input': [], 'output': []}
    result['input']             = data['input']
    for value in data['output']:
        result['output'].append([value[label] for label in labels])
    return result

# Generate coordinates from the grid X
########################################################################################################################
def gridCoordinate(X):
    """
    gridCoordinate(X):
    Function Description:
        Generate coordinates from the grid X.
    Parameter Description:
        X                       Time-domain coordinates(s)      —— [x_1, x_2, …… , x_n]
    Return Description:
        data                    Sample                          —— {'input': [Sample size, Input dimension n], 'output': [Sample size, 1]}
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        if __name__ == '__main__':
            X                   = [np.linspace(-1, 1, 3), np.linspace(-2, 2, 5)]
            data                = fsn.gridCoordinate(X)
            print(data)
    """
    data                        = {'input': [[]], 'output': []}
    for grid in X:
        tem                     = []
        for coor in data['input']:
            for c in grid:
                tem.append(coor + [c])
        data['input']           = tem
    for i in range(len(data['input'])):
        data['output'].append(np.zeros((1)))
    data['input']               = np.array(data['input'], dtype = np.float32)
    data['output']              = np.array(data['output'], dtype = np.float32)
    return data

# Find the fast Fourier transform of the data samples
########################################################################################################################
def dataProcess_FFT(map_grid, sampleRate):
    """
    dataProcess_FFT(map_grid, sampleRate):
    Function Description:
        Solve the spectrum of periodic discrete signals.
    Parameter Description:
        map_grid                Time-domain data graph          —— [feature_1, feature_2, …… , feature_n, value]
        sampleRate              Sampling frequency(Hz)          —— [sampleRate_1, sampleRate_2, …… , sampleRate_n]
    Return Description:
        spectrum                Frequency-domain data graph     —— Correspond to map_grid
        F                       Frequency-domain coordinates    —— [f_1, f_2, …… , f_n]
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        import matplotlib.pyplot as plt
        if __name__ == '__main__':
            sampleRate          = [200, 100]
            domain              = [[0.2, 0.6], [0, 0.8]]
            x                   = np.array(range(int(domain[0][0] * sampleRate[0]), int(domain[0][1] * sampleRate[0]))) / sampleRate[0]
            y                   = np.array(range(int(domain[1][0] * sampleRate[1]), int(domain[1][1] * sampleRate[1]))) / sampleRate[1]
            map_grid            = np.zeros((np.shape(x)[0], np.shape(y)[0], 2))
            for i, xx in enumerate(x):
                for j, yy in enumerate(y):
                    map_grid[i, j, 0]               = np.cos(20 * 2 * np.pi * xx ** 2) + np.sin(30 * 2 * np.pi * yy ** 2)
                    map_grid[i, j, 1]               = np.cos(20 * 2 * np.pi * xx) + np.sin(30 * 2 * np.pi * yy)
            spectrum, F         = fsn.dataProcess_FFT(map_grid, sampleRate)
            spectrum            = np.fft.fftshift(spectrum, axes = (0, 1))
            F                   = [np.fft.fftshift(f) for f in F]
            (Fx, Fy)            = np.meshgrid(F[1], F[0])
            (X, Y)              = np.meshgrid(y, x)
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, np.abs(spectrum[: , : , 0]))
            plt.title('Frequency Area of Signal#1(Magnitude)')
            plt.xlabel('Input#2(Hz)')
            plt.ylabel('Input#1(Hz)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, np.angle(spectrum[: , : , 0]))
            plt.title('Frequency Area of Signal#1(Phase)')
            plt.xlabel('Input#2')
            plt.ylabel('Input#1')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, np.abs(spectrum[: , : , 1]))
            plt.title('Frequency Area of Signal#2(Magnitude)')
            plt.xlabel('Input#2(Hz)')
            plt.ylabel('Input#1(Hz)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, np.angle(spectrum[: , : , 1]))
            plt.title('Frequency Area of Signal#2(Phase)')
            plt.xlabel('Input#2')
            plt.ylabel('Input#1')
            plt.show()
            spectrum            = np.fft.ifftshift(spectrum, axes = (0, 1))
            signal              = np.fft.ifftn(spectrum, axes = (0, 1)) * np.prod(np.shape(spectrum)[0: 2])
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, map_grid[: , : , 0])
            plt.title('Original Signal#1')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.real(signal[: , : , 0]))
            plt.title('Restruction of Signal#1')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, map_grid[: , : , 1])
            plt.title('Original Signal#2')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.real(signal[: , : , 1]))
            plt.title('Restruction of Signal#2')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
    """
    shape                       = np.shape(map_grid)
    feature_num                 = np.shape(shape)[0] - 1
    N                           = [[i for i in range(shape[feature])] for feature in range(feature_num)]
    # Calculate the spectrum
    spectrum                    = np.fft.fftn(map_grid, axes = tuple(range(feature_num))) / np.prod(shape[: feature_num])
    # Calculate the coordinates
    F                           = [[i * sampleRate[feature] / shape[feature] for i in range(-shape[feature] // 2, shape[feature] - shape[feature] // 2)] for feature in range(feature_num)]
    F                           = [np.fft.ifftshift(np.array(f)) for f in F]
    return spectrum, F

# Find the quasi-fast Fourier transform of the spectrum
########################################################################################################################
def dataProcess_IFFT(spectrum, sampleRate):
    """
    dataProcess_IFFT(spectrum, sampleRate):
    Function Description:
        Solve the spectrum of periodic discrete signals.
    Parameter Description:
        spectrum                Frequency-domain data graph     —— [frequency_1, frequency_2, …… , frequency_n, value]
        sampleRate              Sampling frequency(Hz)          —— [sampleRate_1, sampleRate_2, …… , sampleRate_n]
    Return Description:
        map_grid                Time-domain data graph          —— Correspond to spectrum
        X                       Time-domain coordinates(s)      —— [x_1, x_2, …… , x_n]
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        import matplotlib.pyplot as plt
        if __name__ == '__main__':
            sampleRate          = [200, 100]
            domain              = [[0.2, 0.6], [0, 0.8]]
            x                   = np.array(range(int(domain[0][0] * sampleRate[0]), int(domain[0][1] * sampleRate[0]))) / sampleRate[0]
            y                   = np.array(range(int(domain[1][0] * sampleRate[1]), int(domain[1][1] * sampleRate[1]))) / sampleRate[1]
            map_grid            = np.zeros((np.shape(x)[0], np.shape(y)[0], 2))
            for i, xx in enumerate(x):
                for j, yy in enumerate(y):
                    map_grid[i, j, 0]               = np.cos(20 * 2 * np.pi * xx ** 2) + np.sin(30 * 2 * np.pi * yy ** 2)
                    map_grid[i, j, 1]               = np.cos(20 * 2 * np.pi * xx) + np.sin(30 * 2 * np.pi * yy)
            spectrum, F         = fsn.dataProcess_FFT(map_grid, sampleRate)
            spectrum            = np.fft.fftshift(spectrum, axes = (0, 1))
            F                   = [np.fft.fftshift(f) for f in F]
            (Fx, Fy)            = np.meshgrid(F[1], F[0])
            (X, Y)              = np.meshgrid(y, x)
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, abs(spectrum[: , : , 0]))
            plt.title('Frequency Area of Signal#1(Magnitude)')
            plt.xlabel('Input#2(Hz)')
            plt.ylabel('Input#1(Hz)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, abs(spectrum[: , : , 1]))
            plt.title('Frequency Area of Signal#2(Magnitude)')
            plt.xlabel('Input#2(Hz)')
            plt.ylabel('Input#1(Hz)')
            plt.show()
            spectrum            = np.fft.ifftshift(spectrum, axes = (0, 1))
            signal, Z           = fsn.dataProcess_IFFT(spectrum, sampleRate)
            (z1, z2)            = np.meshgrid(np.array(Z[1]) + domain[1][0], np.array(Z[0]) + domain[0][0])
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, map_grid[: , : , 0])
            plt.title('Original Signal#1')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(z1, z2, np.real(signal[: , : , 0]))
            plt.title('Restruction of Signal#1')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, map_grid[: , : , 1])
            plt.title('Original Signal#2')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(z1, z2, np.real(signal[: , : , 1]))
            plt.title('Restruction of Signal#2')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
    """
    shape                       = np.shape(spectrum)
    feature_num                 = np.shape(shape)[0] - 1
    # Calculate the time-domain signal
    map_grid                    = np.fft.ifftn(spectrum, axes = tuple(range(feature_num))) * np.prod(shape[: feature_num])
    X                           = [[i / sampleRate[feature] for i in range(0, shape[feature])] for feature in range(feature_num)]
    return map_grid, X

# Reverse solution of single-layer parameters
########################################################################################################################
def singleLayer(fs, T, C, spectrum, solveEnable = True):
    """
    singleLayer(fs, T, C, spectrum, solveEnable = True):
    Function Description:
        Complete the spectrum and parameter conversion of the first layer of the network.
    Parameter Description:
        fs                      Sampling frequency(Hz)          —— [fs_1, fs_2, …… , fs_n]
        T                       Truncation period(s)            —— float
        C                       Fourier series of activation    —— [C_0, C_1, …… , C_m, C_-m, …… , C_-1]
        spectrum                Frequency-domain data graph     —— [frequency_1, frequency_2, …… , frequency_n]
        solveEnable             Solve enable                    —— True(Default) solve the amplifier and direction weights/False
    Return Description:
        direction               Direction weights               —— [n, ?]
        amplifier               Amplification weights           —— [?, 1]
        spectrum_rebuild        Reconstructed spectrum          —— [frequency_1, frequency_2, …… , frequency_n]
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        class ComplexExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                outputs         = (inputs * 2j * np.pi).exp()
                ctx.save_for_backward(outputs)
                return outputs
            @staticmethod
            def backward(ctx, grad_output):
                outputs,        = ctx.saved_tensors
                return grad_output * outputs
        class TestModel(torch.nn.Module):
            def __init__(self, nodes):
                super().__init__()
                self.Linear_1   = nn.Linear(2, nodes, dtype = complex)
                self.Linear_2   = nn.Linear(nodes, 1, dtype = complex)
            def forward(self, inputs):
                layer_1         = self.Linear_1(inputs)
                activation      = ComplexExp.apply(layer_1)
                outputs         = self.Linear_2(activation)
                return outputs
        if __name__ == '__main__':
            T                   = 1
            C                   = [0, 1, 0]
            fs                  = [100, 100]
            x                   = np.array([i / fs[0] for i in range(-int(fs[0] / 2), int(fs[0] / 2))], dtype = complex)
            y                   = np.array([i / fs[1] for i in range(-int(fs[1] / 2), int(fs[1] / 2))], dtype = complex)
            g                   = np.zeros((fs[0], fs[1], 1), dtype = complex)
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    g[i, j, 0]  = np.cos(20 * np.pi * (yj ** 2 + xi ** 2)) + 1
            spectrum, F         = fsn.dataProcess_FFT(g, fs)
            direction, amplifier, spectrum_rebuild = fsn.singleLayer(fs, T, C, spectrum[: , : , 0])
            testModel           = TestModel(len(direction) - 1)
            modelDict           = testModel.state_dict()
            parameter           = [direction[1: , : ], np.array([0 for i in range(len(direction) - 1)], dtype = complex), amplifier.T[: , 1: ], amplifier.T[: , 0]]
            for key, value in zip(modelDict, parameter):
                modelDict[key]  = torch.from_numpy(value)
            testModel.load_state_dict(modelDict)
            xy                  = np.zeros((fs[0], fs[1], 2), dtype = complex)
            for i, xx in enumerate(np.fft.fftshift(x)):
                for j, yy in enumerate(np.fft.fftshift(y)):
                    xy[i, j, 0], xy[i, j, 1]    = (xx, yy)
            testModel.eval()
            with torch.no_grad():
                prediction      = testModel(torch.from_numpy(xy))
            spectrum            = np.fft.fftshift(spectrum, axes = (0, 1))
            spectrum_rebuild    = np.fft.fftshift(spectrum_rebuild, axes = (0, 1))
            F                   = [np.fft.fftshift(f) for f in F]
            (X, Y)              = np.meshgrid(y, x)
            (Fx, Fy)            = np.meshgrid(F[1], F[0])
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(g[: , : , 0]))
            plt.title('Original Signal')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, np.abs(spectrum[: , : , 0]))
            plt.title('Original Spectrum')
            plt.xlabel('Input#2(Hz)')
            plt.ylabel('Input#1(Hz)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(prediction.numpy()[: , : , 0]))
            plt.title('Network Funtion')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, np.abs(spectrum_rebuild))
            plt.title('Network Spectrum')
            plt.xlabel('Input#2(Hz)')
            plt.ylabel('Input#1(Hz)')
            plt.show()
    """
    # Obtain the spectral size. The spectral size determines the lower limit of the number of network parameters
    # The number of spectral sample points = the number of amplification weights +1(bias)
    shape                       = np.shape(spectrum)
    # Obtain the feature dimension
    feature_num                 = np.shape(shape)[0]
    # Calculate the data spectral interval
    omega_s                     = np.array([fs[i] / shape[i] for i in range(feature_num)])
    # Calculate the spectral interval of the activation function
    omega_0                     = 1 / T
    # Calculate the conversion coefficient from the activation function spectrum to the signal spectrum
    unit                        = omega_s / omega_0
    # Initialize and reconstruct the spectrum to check the correctness of the algorithm
    # By saving the remaining terms generated by the calculation, the system of equations for the subsequent terms can be conveniently called up
    spectrum_rebuild            = np.zeros(shape, dtype = complex)
    # Initialize the weights
    direction                   = np.zeros(list(shape) + [feature_num], dtype = complex)
    amplifier                   = np.zeros(shape, dtype = complex)
    if solveEnable:
        # Considering the periodicity of fft, calculate the boundaries of each quadrant
        border                  = []
        for length in shape:
            if length % 2:
                border.append([-(length // 2), length // 2])
            else:
                border.append([-(length // 2), length // 2 - 1])
        # The coefficient matrix of a system of linear equations in two variables
        # Since only fft is considered, it is a constant matrix; otherwise, weighted coefficients are required
        parameter               = np.matrix([[C[1], C[-1]], [C[-1], C[1]]])
        # Obtain the spectral coordinate index (positive)
        coordinate              = [[]]
        for feature in range(feature_num):
            tem                 = []
            for coor in coordinate:
                tem            += [coor + [j] for j in range(-border[feature][0] + 1)]
            coordinate          = tem
        # Assign a value to the mask
        mask                    = [[]]
        for feature in range(feature_num):
            tem                 = []
            for n, m in enumerate(mask):
                tem.append(m + [(-1) ** n])
            for n, m in enumerate(mask):
                tem.append(m + [(-1) ** (n + 1)])
            mask                = tem
        # Sort the coordinates of the spectral points in the order of the spectrum from the inside out
        coordinate_sort         = [[] for i in range(-sum([border[i][0] for i in range(feature_num)]) + 1)]
        for coor in coordinate:
            coordinate_sort[sum(coor)].append(coor)
        coordinate              = []
        for coor in coordinate_sort[1: ]:
            coordinate         += coor
        # Solve the amplification coefficients in pairs
        for coor in coordinate:
            for m in range(len(mask) // 2):
                # Find the coordinates of the symmetrical points
                coor_pair       = [[coor[i] * mask[m * 2][i] for i in range(feature_num)], [coor[i] * mask[m * 2 + 1][i] for i in range(feature_num)]]
                # Calculate the extended term on the right side of the system of equations,
                # which is obtained by subtracting the influence produced by the previous term from the corresponding spectral value
                expand          = np.matrix([[spectrum[tuple(coor_pair[0])] - spectrum_rebuild[tuple(coor_pair[0])]], [spectrum[tuple(coor_pair[1])] - spectrum_rebuild[tuple(coor_pair[1])]]])
                # Solve for the amplification weights
                result          = np.dot(np.linalg.inv(parameter), expand)
                # Calculate the ticks to determine whether the boundaries have been exceeded
                for c in range(0, len(C) // 2 + 1):
                    for d in range(2):
                        tick    = True
                        for feature in range(feature_num):
                            if border[feature][0] > coor_pair[d][feature] * c or border[feature][1] < coor_pair[d][feature] * c:
                                tick            = False
                        # Assignment within the domain
                        if tick:
                            if c == 1:
                                # Assign values to the direction weights
                                direction[tuple(coor_pair[d])]  = np.array(coor_pair[d]) * unit
                                # Assign values to the amplification weights
                                amplifier[tuple(coor_pair[d])] += result[d, 0]
                            # Add the positions of the series coefficients that may have an impact when not exceeding the spectral domain
                            spectrum_rebuild[tuple([i * c for i in coor_pair[d]])] += C[c] * result[d, 0]
                            spectrum_rebuild[tuple([i * c for i in coor_pair[d]])] += C[-c] * result[1 - d, 0]
        # Calculate the bias term
        amplifier[tuple([0 for i in range(feature_num)])]       = spectrum[tuple([0 for i in range(feature_num)])] - spectrum_rebuild[tuple([0 for i in range(feature_num)])]
        # Add the influence of the bias term to obtain the complete spectrum after reconstruction
        spectrum_rebuild[tuple([0 for i in range(feature_num)])]+= amplifier[tuple([0 for i in range(feature_num)])]
    # Adjust the output format
    direction                   = np.reshape(direction, (-1, feature_num))
    amplifier                   = np.reshape(amplifier, (-1, 1))
    return direction, amplifier, spectrum_rebuild

# Multilayer spectrum reverse decomposition
########################################################################################################################
def divide(unit, blocks, spectrum, solveEnable = True): 
    """
    divide(unit, blocks, spectrum):
    Function Description:
        Divide the spectrum into blocks.
    Parameter Description:
        unit                    Sampling interval(Hz)           —— [unit_1, unit_2, …… , unit_n]
        blocks                  Number of Spectral blocks       —— [block_1, block_2, …… , block_n]
        spectrum                Frequency-domain data graph     —— [frequency_1, frequency_2, …… , frequency_n]
        solveEnable             Solve enable                    —— True(Default) solve the amplifier and direction weights/False
    Return Description:
        spectrum_blocks         Spectrum blocks                 —— [blocks_1, blocks_2, …… , blocks_n, block_1, block_2, …… , block_n]
        bias_blocks             Relative displacement(Hz)       —— [bias_1, bias_2, …… , bias_n]
        F_blocks                Spectrum block coordinates(Hz)  —— [f_1, f_2, …… , f_n]
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        import matplotlib.pyplot as plt
        if __name__ == '__main__':
            fs                  = [80, 80]
            block               = [3, 3]
            x                   = np.array([i / fs[0] - 0.5 for i in range(fs[0])], dtype = complex)
            y                   = np.array([i / fs[1] - 0.5 for i in range(fs[1])], dtype = complex)
            g                   = np.zeros((fs[0], fs[1], 1), dtype = complex)
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    g[i, j, 0]  = np.cos(40 * np.pi * (xi ** 2 + yj ** 2))
            spectrum, F, phase  = fsn.dataProcess_FFT(g, fs)
            blocks, bias, F_block               = fsn.divide(fs, block, spectrum[: , : , 0])
            spectrum            = np.fft.fftshift(spectrum, axes = (0, 1))
            blocks              = np.fft.fftshift(blocks, axes = (2, 3))
            F                   = [np.fft.fftshift(f) for f in F]
            F_block             = [np.fft.fftshift(f) for f in F_block]
            (X, Y)              = np.meshgrid(y, x)
            (Fx, Fy)            = np.meshgrid(F[1], F[0])
            (Fxx, Fyy)          = np.meshgrid(F_block[1], F_block[0])
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.real(g[: , : , 0]))
            plt.title('Original Signal')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(Fx, Fy, np.abs(spectrum[: , : , 0]))
            plt.title('Original Spectrum')
            plt.xlabel('Input#2(Hz)')
            plt.ylabel('Input#1(Hz)')
            plt.show()
            for i in range(-(block[0] // 2), block[0] - (block[0] // 2)):
                for j in range(-(block[1] // 2), block[1] - (block[1] // 2)):
                    fig         = plt.figure()
                    ax          = plt.axes(projection = '3d')
                    ax.plot_surface(Fxx, Fyy, np.abs(blocks[i, j, : , : ]))
                    plt.title('Spectrum Blocks(%d, %d) # bias: (%.2f, %.2f)'%(i, j, bias[i, j, 0], bias[i, j, 1]))
                    plt.xlabel('Input#2(Hz)')
                    plt.ylabel('Input#1(Hz)')
                    plt.show()
    """
    shape                       = np.shape(spectrum)
    feature_num                 = np.shape(shape)[0]
    border                      = [[-(i // 2), i - (i // 2) - 1] for i in shape]
    # Calculate the size of spectral blocks and their boundaries
    block                       = [int(np.ceil(shape[i] / blocks[i])) for i in range(feature_num)]
    border_block                = [[-(i // 2), i - (i // 2) - 1] for i in block]
    border_blocks               = [[-(i // 2), i - (i // 2) - 1] for i in blocks]
    # Initialize the spectrum blocks
    spectrum_blocks             = np.zeros(blocks + list(block), dtype = complex)
    bias_blocks                 = np.zeros(blocks + [feature_num])
    # Calculate the relative positions of the spectrum blocks
    coordinate_blocks           = [[]]
    for feature in range(feature_num):
        tem                     = []
        for coor in coordinate_blocks:
            tem                += [coor + [j] for j in range(border_blocks[feature][0], border_blocks[feature][1] + 1)]
        coordinate_blocks       = tem
    # Calculate the spectral block offset
    coordinate_block            = [[]]
    for feature in range(feature_num):
        tem                     = []
        for coor in coordinate_block:
            tem                += [coor + [j] for j in range(border_block[feature][0], border_block[feature][1] + 1)]
        coordinate_block        = tem
    # Calculate the center of the spectrum block and assign the value
    for c1 in coordinate_blocks:
        for feature in range(feature_num):
            center                               = [c1[i] * block[i] for i in range(feature_num)]
            if solveEnable:
                for c2 in coordinate_block:
                    spectrum_blocks[tuple(c1 + c2)]  = spectrum[tuple([center[i] + c2[i] for i in range(feature_num)])]
            # Calculate the phase offset of the spectral block
            bias_blocks[tuple(c1)][: ]           = [-center[i] * unit[i] for i in range(feature_num)]
    # Calculate the coordinates
    F_blocks                    = [[j * unit[i] for j in range(-(block[i] // 2), block[i] - block[i] // 2)] for i in range(feature_num)]
    F_blocks                    = [np.fft.ifftshift(np.array(f)) for f in F_blocks]
    return spectrum_blocks, bias_blocks, F_blocks

# Reverse solution of multi-layer parameters
########################################################################################################################
def dictInit(coor, value = 0):
    result                      = {}
    for c in coor:
        result['-'.join([str(num) for num in c])]               = value
    return result
def dictShift(data, num, shape, coor, feature_num, keys):
    result                      = dictInit(coor)
    for feature in range(feature_num):
        num[feature]           %= shape[feature]
    for i, c in enumerate(coor):
        c_shift                 = [c[feature] + num[feature] for feature in range(feature_num)]
        c_shift                 = [c_shift[feature] - shape[feature] if c_shift[feature] >= shape[feature] else c_shift[feature] for feature in range(feature_num)]
        result['-'.join([str(num) for num in c_shift])]         = data[keys[i]]
    return result
def dictConv(spectrum, kernal, shape, coor, feature_num, keys):
    result                      = dictInit(coor)
    for c in coor:
        kernal_shift            = dictShift(kernal, c, shape, coor, feature_num, keys)
        key                     = '-'.join([str(num) for num in c])
        for k in keys:
            result[key]        += spectrum[k] * kernal_shift[k]
    return result
def multiLayer(spectrum, activation):
    """
    multiLayer(spectrum, activation):
    Function Description:
        Complete the spectrum and parameter conversion of the subsequent layer of the network.
    Parameter Description:
        spectrum                Output spectrum data graph      —— [frequency_1, frequency_2, …… , frequency_n]
        activation              Polynomial coefficients         —— [C_0, C_1, …… , C_m]
    Return Description:
        spectrum_seperate       Direction weights               —— [spectrum_1, spectrum_2, …… ]
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        from scipy.signal import convolve2d
        if __name__ == '__main__':
            kernal              = np.arange(0, 1.2, 0.1)
            kernal[0: 3]        = np.zeros((3))[: ]
            kernal[4: 6]        = np.zeros((2))[: ]
            kernal[9: 12]       = np.zeros((3))[: ]
            kernal              = kernal.reshape((3, 2, 2))
            spectrum            = np.array([convolve2d(kernal[i], kernal[i], mode = 'full') for i in range(3)])
            spectrum            = np.sum(spectrum, axis = 0)
            print('Origin kernal:')
            for i in range(3):
                print(kernal[i])
            result              = fsn.multiLayer(spectrum, activation = [0, 0, 1])
            print('Solution:')
            for i in range(3):
                print(result[i])
    """
    shape                       = np.shape(spectrum)
    feature_num                 = len(shape)
    order                       = len(activation) - 1
    kernal_size                 = [int(np.ceil((shape[i] + order - 1) / order)) for i in range(feature_num)]
    node_num                    = int(np.ceil(np.prod(shape) / np.prod(kernal_size)))
    # Variable index
    coor_var                    = [[]]
    keys_var                    = []
    for feature in range(feature_num):
        tem                     = []
        for i in range(kernal_size[feature]):
            for c in coor_var:
                tem.append(c + [i])
        coor_var                = tem
    for c in coor_var:
        keys_var.append('-'.join([str(num) for num in c]))
    # Variable symbols
    var                         = {}
    for i in range(node_num):
        var[i]                  = dictInit(coor_var)
        for k in keys_var:
            var[i][k]           = sp.symbols('%d-'%i + k)
    # Convolution
    coor                        = [[]]
    keys                        = []
    for feature in range(feature_num):
        tem                     = []
        for i in range(shape[feature]):
            for c in coor:
                tem.append(c + [i])
        coor                    = tem
    for c in coor:
        keys.append('-'.join([str(num) for num in c]))
    spectrum_rebuild            = dictInit(coor)
    spectrum_conv               = [dictInit([[0 for feature in range(feature_num)]], value = 1) for i in range(node_num)]
    shape_new                   = [1 for feature in range(feature_num)]
    coor_new                    = [[]]
    keys_new                    = []
    for feature in range(feature_num):
        tem                     = []
        for i in range(shape_new[feature]):
            for c in coor_new:
                tem.append(c + [i])
        coor_new                = tem
    for c in coor_new:
        keys_new.append('-'.join([str(num) for num in c]))
    for i in range(1, order + 1):
        shape_old               = shape_new
        keys_old                = keys_new
        shape_new               = tuple([shape_old[feature] + kernal_size[feature] - 1 for feature in range(feature_num)])
        coor_new                = [[]]
        for feature in range(feature_num):
            tem                 = []
            for j in range(shape_new[feature]):
                for c in coor_new:
                    tem.append(c + [j])
            coor_new            = tem
        keys_new                = []
        for c in coor_new:
            keys_new.append('-'.join([str(num) for num in c]))
        coor_sub                = [int(shape[feature] // 2 - shape_new[feature] // 2) for feature in range(feature_num)]
        spectrum_rebuild        = dictShift(spectrum_rebuild, [shape[feature] - coor_sub[feature] for feature in range(feature_num)], shape, coor, feature_num, keys)
        for j in range(node_num):
            spectrum_tem        = dictInit(coor_new)
            kernal              = dictInit(coor_new)
            for c in keys_old:
                spectrum_tem[c] = spectrum_conv[j][c]
            spectrum_tem        = dictShift(spectrum_tem, [kernal_size[feature] - 1 for feature in range(feature_num)], shape_new, coor_new, feature_num, keys_new)
            for k, c in zip(keys_var, coor_var):
                kernal[k]       = var[j]['-'.join([str(kernal_size[feature] - 1 - c[feature]) for feature in range(feature_num)])]
            spectrum_conv[j]    = dictConv(spectrum_tem, kernal, shape_new, coor_new, feature_num, keys_new)
            for c in keys_new:
                spectrum_rebuild[c]                            += spectrum_conv[j][c] * activation[i]
        spectrum_rebuild        = dictShift(spectrum_rebuild, coor_sub, shape, coor, feature_num, keys)
    del coor_new, coor_var, keys_new, keys_old, spectrum_tem, spectrum_conv, kernal
    # Solution(quite some time of sp.solve)
    equation                    = dictInit(coor)
    for c, k in zip(coor, keys):
        equation[k]             = sp.Eq(spectrum_rebuild[k], spectrum[tuple(c)])
    equation                    = list(equation.values())
    variable                    = []
    for i in range(node_num):
        variable               += list(var[i].values())
    solution                    = sp.solve(tuple(equation), tuple(variable), dict = True)[0]
    # Constraint
    symbol_constraint           = []
    for c in solution.values():
        for s in c.atoms(sp.Symbol):
            if s not in symbol_constraint:
                symbol_constraint.append(s)
    equation_constraint         = []
    for s in symbol_constraint:
        equation_constraint.append(sp.Eq(s, 0))
    x                           = sp.symbols('x')
    for k, c in zip(solution.keys(), solution.values()):
        solution[k]             = sp.solve(tuple(equation_constraint + [sp.Eq(c, x)]), tuple(symbol_constraint + [x]), dict = True)[0][x]
    # Reconstruct the seperated spectrum
    spectrum_seperate           = np.zeros([node_num] + kernal_size)
    for k, v in zip(solution.keys(), solution.values()):
        c                       = k.split('-')
        i                       = int(c[0])
        c                       = [int(c[feature]) for feature in range(1, feature_num + 1)]
        spectrum_seperate[i][tuple(c)]                          = v
    return spectrum_seperate

# Class of network spectrum component
########################################################################################################################
class SpectralLine:
    def __init__(self):
        """
        __init__(self):
        Function Description:
            Initialize the spectrum class, which is used to record the spectrum components within a certain spectrum.
        Variable Description:(Default)
            self.spectrum       = {'coor': [], 'value': []}
            self.num            = 0
            self.feature_num    = 0
        """
        self.spectrum           = {'coor': [], 'value': []}
        self.num                = 0
        self.feature_num        = 0
        return 
    def add(self, coordinate, value):
        """
        add(self, coordinate, value):
        Function Description:
            Add spectral lines to the spectrum.
        Parameter Description:
            coordinate          Spectral line coordinates(Hz)   —— [f_1, f_2, …… , f_n]
            value               Spectral line value             —— complex
        Test Example:
            import FSNanalysis as fsn
            if __name__ == '__main__':
                spectrum        = fsn.SpectralLine()
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
                spectrum.add([1, 1, 0], 1 + 0j)
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
                spectrum.add([1, 1, 0], 0.5 + 1j)
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
        """
        self.num               += 1
        self.spectrum['coor'].append(coordinate)
        self.spectrum['value'].append(value)
        if self.feature_num == 0:
            self.feature_num    = len(coordinate)
        return
    def translate(self, spectrum, fs):
        """
        translate(self, spectrum, fs):
        Function Description:
            Convert the existing spectrum graph into a spectrum class.
        Parameter Description:
            spectrum            Frequency-domain data graph     —— [f_1, f_2, …… , f_n, value]
            fs                  Sampling frequency(Hz)          —— [fs_1, fs_2, …… , fs_n]
        Test Example:
            import numpy as np
            import FSNanalysis as fsn
            if __name__ == '__main__':
                spectrum        = fsn.SpectralLine()
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
                spectrum.translate(np.reshape(np.array([i for i in range(25)]), (5, 5)), [5, 5])
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
        """
        shape                   = np.shape(spectrum)
        self.feature_num        = np.shape(shape)[0]
        df                      = [fs[i] / shape[i] for i in range(self.feature_num)]
        coordinate              = [[]]
        for i in range(self.feature_num):
            tem                 = []
            for coor in coordinate:
                for j in range(-(shape[i] // 2), shape[i] - shape[i] // 2):
                    tem.append(coor + [j])
            coordinate          = tem
        for coor in coordinate:
            self.add([coor[i] * df[i] for i in range(self.feature_num)], spectrum[tuple(coor)])
        return
    def mul(self, c):
        """
        mul(self, c):
        Function Description:
            Overall scaling of the spectral amplitude.
        Parameter Description:
            c                   Magnification factor            —— float/int
        Return Description:
            spectralLine        Spectrum class                  —— SpectralLine
        Test Example:
            import numpy as np
            import FSNanalysis as fsn
            if __name__ == '__main__':
                spectrum        = fsn.SpectralLine()
                spectrum.translate(np.reshape(np.array([i for i in range(25)]), (5, 5)), [5, 5])
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
                spectrum        = spectrum.mul(-1.)
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
        """
        spectralLine            = SpectralLine()
        spectralLine.feature_num= self.feature_num
        for coor in self.spectrum['coor']:
            spectralLine.spectrum['coor'].append([i for i in coor])
        for value in self.spectrum['value']:
            spectralLine.spectrum['value'].append(c * value)
        spectralLine.num        = self.num
        return spectralLine
    def transform(self):
        """
        transform(self):
        Function Description:
            Transpose the spectrum.
        Return Description:
            spectralLine        Spectrum class                  —— SpectralLine
        Test Example:
            import numpy as np
            import FSNanalysis as fsn
            if __name__ == '__main__':
                spectrum        = fsn.SpectralLine()
                spectrum.translate(np.reshape(np.array([i for i in range(25)]), (5, 5)), [5, 5])
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
                spectrum        = spectrum.transform()
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
        """
        spectralLine            = SpectralLine()
        spectralLine.feature_num= self.feature_num
        for coor in self.spectrum['coor']:
            spectralLine.spectrum['coor'].append([-i for i in coor])
        for value in self.spectrum['value']:
            spectralLine.spectrum['value'].append(value)
        spectralLine.num        = self.num
        return spectralLine
    def copy(self):
        """
        copy(self):
        Function Description:
            Make a copy of the spectrum class.
        Return Description:
            spectralLine        Spectrum class                  —— SpectralLine
        Test Example:
            import numpy as np
            import FSNanalysis as fsn
            if __name__ == '__main__':
                spectrum        = fsn.SpectralLine()
                spectrum.translate(np.reshape(np.array([i for i in range(25)]), (5, 5)), [5, 5])
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
                spectrum_copy   = spectrum.copy()
                del spectrum
                print('Num      : ', spectrum_copy.num)
                print('Dimension: ', spectrum_copy.feature_num)
                print(spectrum_copy.spectrum)
        """
        spectralLine            = SpectralLine()
        spectralLine.feature_num= self.feature_num
        for coor in self.spectrum['coor']:
            spectralLine.spectrum['coor'].append([i for i in coor])
        for value in self.spectrum['value']:
            spectralLine.spectrum['value'].append(value)
        spectralLine.num        = self.num
        return spectralLine
    def combine(self, spectralLine):
        """
        combine(self, spectralLine):
        Function Description:
            Merge the two spectrum classes.
        Parameter Description:
            spectralLine        Spectrum class                  —— SpectralLine
        Return Description:
            result              Spectrum class                  —— SpectralLine
        Test Example:
            import numpy as np
            import FSNanalysis as fsn
            if __name__ == '__main__':
                spectrum        = fsn.SpectralLine()
                spectrum.translate(np.reshape(np.array([i for i in range(25)]), (5, 5)), [5, 5])
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
                spectrum        = spectrum.combine(spectrum.mul(-1))
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
        """
        result                  = self.copy()
        for coor in spectralLine.spectrum['coor']:
            result.spectrum['coor'].append([i for i in coor])
        for value in spectralLine.spectrum['value']:
            result.spectrum['value'].append(value)
        result.num             += spectralLine.num
        if self.feature_num == 0:
            self.feature_num    = spectralLine.feature_num
        return result
    def correlate(self, spectralLine):
        """
        correlate(self, spectralLine):
        Function Description:
            Perform relevant operations on the two spectra.
        Parameter Description:
            spectralLine        Spectrum class                  —— SpectralLine
        Return Description:
            result              Spectrum class                  —— SpectralLine
        Test Example:
            import numpy as np
            import FSNanalysis as fsn
            if __name__ == '__main__':
                spectrum        = fsn.SpectralLine()
                spectrum.translate(np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]), [5, 5])
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
                spectrum        = spectrum.correlate(spectrum)
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                print(spectrum.spectrum)
        """
        result                  = SpectralLine()
        result.feature_num      = self.feature_num
        for i in range(self.num):
            for j in range(spectralLine.num):
                coor            = [self.spectrum['coor'][i][k] - spectralLine.spectrum['coor'][j][k] for k in range(self.feature_num)]
                value           = self.spectrum['value'][i] * spectralLine.spectrum['value'][j]
                if value != 0:
                    result.add(coor, value)
        return result
    def compute(self, inputs):
        """
        compute(self, inputs):
        Function Description:
            Solve the time-domain output based on the spectral components in the spectrum class.
        Parameter Description:
            inputs              Time-domain coordinate(s)       —— [x1, x2, …… , xn]
        Return Description:
            outputs             Time-domain value               —— complex
        Test Example:
            import numpy as np
            import FSNanalysis as fsn
            import matplotlib.pyplot as plt
            if __name__ == '__main__':
                fs              = [10, 10]
                x               = [i for i in range(-fs[0], fs[0])]
                y               = [i for i in range(-fs[1], fs[1])]
                g               = np.zeros((fs[0] * 2, fs[1] * 2, 1))
                for i in x:
                    for j in y:
                        g[(i, j, 0)]            = np.cos(2 * np.pi * 2 * (i / fs[0] + j / fs[1]))
                spec, F         = fsn.dataProcess_FFT(g, fs)
                F               = [np.fft.fftshift(f) for f in F]
                (Fx, Fy)        = np.meshgrid(F[1], F[0])
                fig             = plt.figure()
                ax              = plt.axes(projection = '3d')
                ax.plot_surface(Fx, Fy, np.real(np.fft.fftshift(spec, axes = (0, 1))[: , : , 0]))
                plt.title('Original Spectrum')
                plt.xlabel('Input#2(Hz)')
                plt.ylabel('Input#1(Hz)')
                plt.show()
                spectrum        = fsn.SpectralLine()
                spectrum.translate(spec[: , : , 0], fs)
                print('Num      : ', spectrum.num)
                print('Dimension: ', spectrum.feature_num)
                for i in range(spectrum.num):
                    if abs(spectrum.spectrum['value'][i]) > 0.1:
                        print('coor: ', spectrum.spectrum['coor'][i], ' value: ', spectrum.spectrum['value'][i])
                spec            = np.zeros((fs[0] * 2, fs[1] * 2))
                for i in x:
                    for j in y:
                        spec[(i, j)]            = spectrum.compute([i / fs[0], j / fs[1]])
                (X, Y)          = np.meshgrid([i / fs[1] for i in y], [i / fs[0] for i in x])
                fig             = plt.figure()
                ax              = plt.axes(projection = '3d')
                ax.plot_surface(X, Y, g[: , : , 0])
                plt.title('Original Signal')
                plt.xlabel('Input#2(s)')
                plt.ylabel('Input#1(s)')
                plt.show()
                fig             = plt.figure()
                ax              = plt.axes(projection = '3d')
                ax.plot_surface(X, Y, np.real(spec))
                plt.title('Network Function')
                plt.xlabel('Input#2(s)')
                plt.ylabel('Input#1(s)')
                plt.show()
        """
        outputs                 = 0
        for i in range(self.num):
            value               = self.spectrum['value'][i]
            for feature in range(self.feature_num):
                value          *= np.exp(2j * np.pi * self.spectrum['coor'][i][feature] * inputs[feature])
            outputs            += value
        return outputs

# Forward verification module
########################################################################################################################
def analysis(model, C, T, sampleRate = [], domain = [], enable = True):
    """
    analysis(model, sampleRate, domain, C, T, enable = True):
    Function Description:
        Verify the principle of the spectrum analysis method and present the output and error of the network and the equivalent model.
        Meanwhile, the network parameters of any fully connected neural network can be analyzed to obtain the spectrum
    Parameter Description:
        model                   Network model                   —— Any fully connected neural network
        sampleRate              Sampling frequency(Hz)          —— [sampleRate_1, sampleRate_2, …… , sampleRate_n]
        domain                  Sample domain(s)                —— [[domain_1[0], domain_1[1]], …… ,  [domain_n[0], domain_n[1]]], the domain of definition includes zero points
        C                       Fourier series coefficients of activation       —— [C1, C2, …… ,  Cm], C1 is the Fourier series coefficient of the first layer, and the rest are the Taylor series coefficients of the corresponding layers
        T                       Truncation period(s)            —— float
        enable                  Enable time-domain computing    —— True/False
    Return Description:
        spectrum                Equivalent spectrum             —— SpectralLine()
        result_net              Time-domain output of network   —— [x_1, x_2, …… , x_n, result_net]
        result_spec             Time-domain output of equivalent spectrum       —— [x_1, x_2, …… , x_n, result_spec] According to the sampling frequency and domain, the values of the sampling points output by the network are taken
        loss                    Model error                     —— [x_1, x_2, …… , x_n, loss]
        x                       Time-domain coordinates         —— [x1, x2, …… , xn]
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        class ComplexExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                outputs         = (inputs * 2j * np.pi).exp()
                ctx.save_for_backward(outputs)
                return outputs
            @staticmethod
            def backward(ctx, grad_output):
                outputs,        = ctx.saved_tensors
                return grad_output * outputs
        class x3(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                outputs         = torch.mul(torch.mul(inputs, inputs), inputs)
                ctx.save_for_backward(outputs)
                return outputs
            @staticmethod
            def backward(ctx, grad_output):
                outputs,        = ctx.saved_tensors
                return grad_output * outputs
        class TestModel(torch.nn.Module):
            def __init__(self, nodes):
                super().__init__()
                self.Linear_1   = nn.Linear(2, nodes[0], dtype = complex)
                self.Linear_2   = nn.Linear(nodes[0], nodes[1], dtype = complex)
                self.Linear_3   = nn.Linear(nodes[1], nodes[2], dtype = complex)
                self.Linear_4   = nn.Linear(nodes[2], 1, dtype = complex)
            def forward(self, inputs):
                layer_1         = self.Linear_1(inputs)
                activation_1    = ComplexExp.apply(layer_1)
                layer_2         = self.Linear_2(activation_1)
                activation_2    = x3.apply(layer_2)
                layer_3         = self.Linear_3(activation_2)
                activation_3    = x3.apply(layer_3)
                layer_4         = self.Linear_4(activation_3)
                return layer_4
        if __name__ == '__main__':
            T                   = 1
            C                   = [[0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
            domain              = [[0, 1], [0, 1]]
            sampleRate          = [200, 200]
            testModel           = TestModel([1, 1, 1])
            modelDict           = testModel.state_dict()
            for i, key in enumerate(modelDict):
                modelDict[key]  = torch.from_numpy(np.real(modelDict[key].numpy()))
            testModel.load_state_dict(modelDict)
            spec, result_net, result_spec, loss, x  = fsn.analysis(testModel, C, T, sampleRate, domain)
            (X, Y)              = np.meshgrid(x[1], x[0])
            maximum             = np.max(np.abs(result_net[: , : , 0]))
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(result_net[: , : , 0]))
            plt.title('Network Signal(Magnitude)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(result_spec[: , : , 0]))
            plt.title('Restruct Signal(Magnitude)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.angle(result_net[: , : , 0]))
            plt.title('Network Signal(Phase)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.angle(result_spec[: , : , 0]))
            plt.title('Restruct Signal(Phase)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(loss[: , : , 0]))
            ax.set_zlim(0, maximum)
            plt.title('Loss')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
    """
    weights                     = []
    bias                        = []
    C_num                       = [[-(len(c) // 2), len(c) - len(c) // 2] for c in C]
    # Extract parameters
    modelDict                   = model.state_dict()
    for i, key in enumerate(modelDict):
        if i % 2 == 0:
            weights.append(modelDict[key].numpy())
        else:
            bias.append(modelDict[key].numpy())
    layer_num                   = len(weights)
    # Construct the spectrum class
    shape                       = [np.shape(weights[i]) for i in range(layer_num)]
    spectrum                    = []
    for i in range(shape[1][0]):
        spectrum.append(SpectralLine())
    # Add the first-layer spectral components
    for i in range(shape[0][0]):
        for c in range(C_num[0][0], C_num[0][1]):
            coor                = [weights[0][i, k] * c / T for k in range(shape[0][1])]
            for j in range(shape[1][0]):
                value           = C[0][c] * np.exp(2j * np.pi * c * bias[0][i] / T) * weights[1][j, i]
                if value != 0:
                    spectrum[j].add(coor, value)
    # Correct the DC component
    for i in range(shape[1][0]):
        if bias[1][i] != 0:
            spectrum[i].add([0 for k in range(shape[0][1])], bias[1][i])
    # Spectrum conduction
    label                       = [1, 2]
    while label[0] < len(C_num) - 1 or label[1] < layer_num:
        # Activate
        if label[0] < len(C_num):
            for i, s in enumerate(spectrum):
                tem             = [SpectralLine(), SpectralLine()]
                for j, c in enumerate(C[label[0]]):
                    if c != 0:
                        if j == 0:
                            tem[0].add([0 for k in range(shape[0][1])], c)
                        else:
                            tem[1]              = s.copy()
                            core                = s.transform()
                            for k in range(j - 1):
                                tem[1]          = tem[1].correlate(core)
                            tem[0]              = tem[0].combine(tem[1].mul(c))
                spectrum[i]     = tem[0].copy()
            label[0]           += 1
        # Weighted sum
        if label[1] < layer_num:
            tem                 = []
            for i in range(shape[label[1]][0]):
                tem.append(SpectralLine())
                for j in range(shape[label[1]][1]):
                    tem[-1]     = tem[-1].combine(spectrum[j].mul(weights[label[1]][i, j]))
                tem[-1].add([0 for k in range(shape[0][1])], bias[label[1]][i])
            spectrum            = tem.copy()
            label[1]           += 1
    # Calculate the time-domain coordinates
    if enable:
        x                       = []
        shape_x                 = []
        for i, fs in enumerate(sampleRate):
            x.append(np.array([i / fs for i in range(domain[i][0] * fs, domain[i][1] * fs)], dtype = complex))
            shape_x.append(len(x[-1]))
        feature_num             = len(shape_x)
        coordinate              = [[]]
        for feature in range(feature_num):
            tem                 = []
            for c in coordinate:
                for i in range(shape_x[feature]):
                    tem.append(c + [i])
            coordinate          = tem
        # Calculation time-domain result
        result_net              = np.zeros(tuple(shape_x + [shape[-1][0]]), dtype = complex)
        result_spec             = np.zeros(tuple(shape_x + [shape[-1][0]]), dtype = complex)
        loss                    = np.zeros(tuple(shape_x + [shape[-1][0]]), dtype = complex)
        model.eval()
        with torch.no_grad():
            for c in coordinate:
                coor                            = [x[i][c[i]] for i in range(feature_num)]
                result_net[tuple(c)]            = model(torch.tensor(coor)).numpy()
                for i in range(shape[-1][0]):
                    result_spec[tuple(c + [i])] = spectrum[i].compute(coor)
                loss[tuple(c)]                  = result_net[tuple(c)] - result_spec[tuple(c)]
    else:
        result_net, result_spec, loss, x        = None, None, None, None
    return spectrum, result_net, result_spec, loss, x

# Equivalence verification of single-layer and multi-layer
########################################################################################################################
def multi_to_single(model, C, T, f_border):
    """
    multi_to_single(model, C, T):
    Function Description:
        Obtain a single-layer network consistent with the functions of the multi-layer network.
    Parameter Description:
        model                   Network model                   —— Any fully connected neural network
        C                       Fourier series coefficients of activation       —— [C1, C2, …… ,  Cm], C1 is the Fourier series coefficient of the first layer, and the rest are the Taylor series coefficients of the corresponding layers
        T                       Truncation period(s)            —— float
        f_border                Frequency boundary(Hz)          —— [f1, f2, …… , fn]
    Return Description:
        direction               Direction weights               —— [d1, d2, …… , dk]
        amplifier               Amplification weights           —— [a1, a2, …… , ak]
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        class ComplexExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                outputs         = (inputs * 2j * np.pi).exp()
                ctx.save_for_backward(outputs)
                return outputs
            @staticmethod
            def backward(ctx, grad_output):
                outputs,        = ctx.saved_tensors
                return grad_output * outputs
        class x3(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                outputs         = torch.mul(torch.mul(inputs, inputs), inputs) + torch.mul(inputs, inputs) + inputs + 1
                ctx.save_for_backward(outputs)
                return outputs
            @staticmethod
            def backward(ctx, grad_output):
                outputs,        = ctx.saved_tensors
                return grad_output * outputs
        class TestModel(torch.nn.Module):
            def __init__(self, nodes):
                super().__init__()
                self.Linear_1   = nn.Linear(2, nodes[0], dtype = complex)
                self.Linear_2   = nn.Linear(nodes[0], nodes[1], dtype = complex)
                self.Linear_3   = nn.Linear(nodes[1], nodes[2], dtype = complex)
                self.Linear_4   = nn.Linear(nodes[2], 1, dtype = complex)
            def forward(self, inputs):
                layer_1         = self.Linear_1(inputs)
                activation_1    = ComplexExp.apply(layer_1)
                layer_2         = self.Linear_2(activation_1)
                activation_2    = x3.apply(layer_2)
                layer_3         = self.Linear_3(activation_2)
                activation_3    = x3.apply(layer_3)
                layer_4         = self.Linear_4(activation_3)
                return layer_4
        class TestModel_single(torch.nn.Module):
            def __init__(self, nodes):
                super().__init__()
                self.Linear_1   = nn.Linear(2, nodes, dtype = complex)
                self.Linear_2   = nn.Linear(nodes, 1, dtype = complex)
            def forward(self, inputs):
                layer_1         = self.Linear_1(inputs)
                activation      = ComplexExp.apply(layer_1)
                outputs         = self.Linear_2(activation)
                return outputs
        if __name__ == '__main__':
            T                   = 1
            C                   = [[0, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
            fs                  = [100, 100]
            domain              = [[0, 1], [0, 1]]
            sampleRate          = [200, 200]
            f_border            = [0, 0]
            testModel           = TestModel([1, 1, 1])
            modelDict           = testModel.state_dict()
            for i, key in enumerate(modelDict):
                modelDict[key]  = torch.from_numpy(np.real(modelDict[key].numpy()))
            testModel.load_state_dict(modelDict)
            dir, amp, bias      = fsn.multi_to_single(testModel, C, T, f_border)
            nodes               = len(amp)
            singleModel         = TestModel_single(nodes)
            modelDict           = singleModel.state_dict()
            parameter           = [dir, np.array([0 for i in range(nodes)], dtype = complex), amp.T, bias]
            for key, value in zip(modelDict, parameter):
                modelDict[key]  = torch.from_numpy(value)
            singleModel.load_state_dict(modelDict)
            x                   = np.array([i / fs[0] for i in range(fs[0])], dtype = complex)
            y                   = np.array([i / fs[1] for i in range(fs[1])], dtype = complex)
            xy                  = np.zeros((fs[0], fs[1], 2), dtype = complex)
            for i, xx in enumerate(x):
                for j, yy in enumerate(y):
                    xy[i, j, 0], xy[i, j, 1]        = (xx, yy)
            testModel.eval()
            with torch.no_grad():
                prediction_1    = testModel(torch.from_numpy(xy)).numpy()
                prediction_2    = singleModel(torch.from_numpy(xy)).numpy()
            maximum             = np.max(np.abs(prediction_1[: , : , 0]))
            (X, Y)              = np.meshgrid(y, x)
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            loss                = prediction_1[: , : , 0] - prediction_2[: , : , 0]
            ax.plot_surface(X, Y, np.abs(prediction_1[: , : , 0]))
            plt.title('MultiLayer Funtion(Magnitude)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(prediction_2[: , : , 0]))
            plt.title('SingleLayer Funtion(Magnitude)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.angle(prediction_1[: , : , 0]))
            plt.title('MultiLayer Funtion(Phase)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.angle(prediction_2[: , : , 0]))
            plt.title('SingleLayer Funtion(Phase)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(loss))
            ax.set_zlim(0, maximum)
            plt.title('Loss')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
    """
    # Calculate the equivalent spectrum
    spec, _, _, _, _            = analysis(model, C, T, enable = False)
    # Traverse different outputs
    direction                   = []
    amplifier                   = []
    bias                        = np.array([0 for i in range(len(spec))], dtype = complex)
    for out, s in enumerate(spec):
        tem                     = []
        # Solve the nodes in the one-dimensional direction in a loop
        for line in range(s.num):
            # Determine whether it is a bias term
            if np.sum(s.spectrum['coor'][line]) == 0:
                bias[out]      += s.spectrum['value'][line]
            else:
                # Calculate the number of additional zeros that need to be supplemented
                N               = int(np.min([f_border[i] // np.real(s.spectrum['coor'][line][i]) for i in range(s.feature_num)])) + 3
                # Construct one-dimensional spectrum
                specLine        = np.array([0, s.spectrum['value'][line], 0] + [0 for i in range(N - 3)], dtype = complex)
                # Calculation parameters
                d, a, _         = singleLayer([N], T, C[0], specLine)
                # Parameter coordinate transformation
                for i in range(-(N // 2), N - N // 2):
                    if d[i, 0] != 0:
                        if out == 0:
                            direction.append([d[i, 0] * s.spectrum['coor'][line][j] for j in range(s.feature_num)])
                        tem.append(a[i , : ])
                    else:
                        bias[out]      += a[i, 0]
        amplifier.append(tem)
    amplifier                   = np.concatenate(tuple(amplifier), axis = 1)
    direction                   = np.array(direction)
    return direction, amplifier, bias

# The complex exponential activation network module
########################################################################################################################
class ComplexExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        outputs                 = (inputs * 2j * np.pi).exp()
        ctx.save_for_backward(outputs)
        return outputs
    @staticmethod
    def backward(ctx, grad_output):
        outputs,                = ctx.saved_tensors
        return grad_output * outputs

# Sinusoidal activation network module
########################################################################################################################
class ComplexSin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        outputs                 = (inputs * 2 * np.pi).sin()
        ctx.save_for_backward(outputs)
        return outputs
    @staticmethod
    def backward(ctx, grad_output):
        outputs,                = ctx.saved_tensors
        return grad_output * outputs

# Cosine activation network module
########################################################################################################################
class ComplexCos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        outputs                 = (inputs * 2 * np.pi).cos()
        ctx.save_for_backward(outputs)
        return outputs
    @staticmethod
    def backward(ctx, grad_output):
        outputs,                = ctx.saved_tensors
        return grad_output * outputs

# Frequency-shift layer (plural) class
########################################################################################################################
class Shift_Complex(torch.nn.Module):
    def __init__(self, nodes_num, omega):
        """
        __init__(self, nodes_num, omega):
        Function Description:
            It is used to implement the functions of spectrum translation and merging.
        Variable Description:
            omega               Offset                         —— [omega1, omega2, …… , omegan]
            nodes_num           Number of nodes                —— int
        """
        super(Shift_Complex, self).__init__()
        self.chunk_num          = int(nodes_num / np.shape(omega)[0])
        self.omega              = nn.Parameter(torch.tensor(omega, dtype = complex).t(), requires_grad = False)
    def forward(self, inputs, nodes):
        # Phase-weighted summation
        omega                   = torch.matmul(inputs, self.omega)
        # Activate
        activation              = ComplexExp.apply(omega)
        # Split nodes
        chunks                  = torch.chunk(nodes, chunks = self.chunk_num, dim = -1)
        # Spectrum shifting and combination
        shift                   = [torch.sum(chunk * activation, dim = -1, keepdim = True) for chunk in chunks]
        # Tensor splicing
        outputs                 = torch.cat(shift, dim = -1)
        return outputs

# FSN network (plural) class
########################################################################################################################
class FSNModel_Complex(torch.nn.Module):
    def __init__(self, blocks, omega, direction, amplifier, classifyEnable = False):
        """
        __init__(self, blocks, omega, direction, amplifier, classifyEnable = False):
        Function Description:
            Define the FSN network.
        Variable Description:
            blocks              Spectral block dimension         —— [block_1, block_2, …… , block_n, [output_num]]
            omega               Frequency offset                 —— [omega_1, omega_2, …… , omega_n]
            direction           Direction weights                —— [out_dim, in_dim]
            amplifier           Amplification weights            —— [out_dim, in_dim]
            classifyEnable      Classify Enable                  —— Specify classification tasks: False(Default)/True
        """
        super(FSNModel_Complex, self).__init__()
        self.complexEnable      = True
        self.classifyEnable     = classifyEnable
        self.feature_num        = np.shape(blocks[0])[0]
        self.omega_num          = 0
        # Calculate the number of nodes in each layer
        self.nodes              = [np.shape(blocks[0])[0], np.shape(direction)[-1], np.shape(amplifier)[-1]]
        for b in blocks[::-1]:
            self.nodes         += [int(self.nodes[-1] / np.prod(b))]
            self.omega_num     += np.prod(b)
        # Direction weights
        self.direction          = nn.Parameter(torch.from_numpy(direction.astype(complex)))
        # Amplification weights
        self.amplifier          = nn.Parameter(torch.from_numpy(amplifier.astype(complex)))
        # Spectrum shifting
        self.shift              = [Shift_Complex(self.nodes[i], omega[i - 2]) for i in range(2, len(self.nodes) - 1)]
        # softmax
        if self.classifyEnable:
            self.softmax        = nn.Softmax(dim = -1)
        # Weight sort
        self.shape              = np.shape(amplifier)
        self.order              = np.argsort(np.reshape(np.abs(amplifier), -1))
    def forward(self, inputs):
        direction               = torch.matmul(inputs, self.direction)
        spectrum_line           = ComplexExp.apply(direction)
        outputs                 = torch.matmul(spectrum_line, self.amplifier)
        for i in range(len(self.shift)):
            outputs             = self.shift[i](inputs, outputs)
        if self.classifyEnable:
            outputs             = self.softmax(outputs.abs())
        return outputs

# Frequency-shifted layer (real) class
########################################################################################################################
class Shift(torch.nn.Module):
    def __init__(self, nodes_num, omega):
        """
        __init__(self, nodes_num, omega):
        Function Description:
            It is used to implement the functions of spectrum translation and merging.
        Variable Description:
            omega               Offset                         —— [omega1, omega2, …… , omegan]
            nodes_num           Number of nodes                —— int
        """
        super(Shift, self).__init__()
        self.chunk_num          = int(nodes_num / np.shape(omega)[0])
        self.omega              = nn.Parameter(torch.tensor(omega).t(), requires_grad = False)
    def forward(self, inputs, nodes_real, nodes_image):
        # Phase-weighted summation
        omega                   = torch.matmul(inputs, self.omega)
        # Activate
        activation_real         = ComplexCos.apply(omega)
        activation_image        = ComplexSin.apply(omega)
        # Split nodes
        chunks_real             = torch.chunk(nodes_real, chunks = self.chunk_num, dim = -1)
        chunks_image            = torch.chunk(nodes_image, chunks = self.chunk_num, dim = -1)
        # Spectrum shifting and combination
        shift_real              = [torch.sum(chunks_real[i] * activation_real - chunks_image[i] * activation_image, dim = -1, keepdim = True) for i in range(self.chunk_num)]
        shift_image             = [torch.sum(chunks_real[i] * activation_image + chunks_image[i] * activation_real, dim = -1, keepdim = True) for i in range(self.chunk_num)]
        # Tensor splicing
        outputs_real            = torch.cat(shift_real, dim = -1)
        outputs_image           = torch.cat(shift_image, dim = -1)
        return outputs_real, outputs_image

# FSN network (real) class
########################################################################################################################
class FSNModel(torch.nn.Module):
    def __init__(self, blocks, omega, direction, amplifier, classifyEnable = False):
        """
        __init__(self, blocks, omega, direction, amplifier, classifyEnable = False):
        Function Description:
            Define the FSN network.
        Variable Description:
            blocks              Spectral block dimension         —— [block_1, block_2, …… , block_n, [output_num]]
            omega               Frequency offset                 —— [omega_1, omega_2, …… , omega_n]
            direction           Direction weights                —— [out_dim, in_dim]
            amplifier           Amplification weights            —— [out_dim, in_dim]
            classifyEnable      Classify Enable                  —— Specify classification tasks: False(Default)/True
        """
        super(FSNModel, self).__init__()
        self.complexEnable      = False
        self.classifyEnable     = classifyEnable
        self.feature_num        = np.shape(blocks[0])[0]
        self.omega_num          = 0
        # Calculate the number of nodes in each layer
        self.nodes              = [np.shape(blocks[0])[0], np.shape(direction)[-1], np.shape(amplifier)[-1]]
        for b in blocks[::-1]:
            self.nodes         += [int(self.nodes[-1] / np.prod(b))]
            self.omega_num     += np.prod(b)
        # Direction weights
        self.direction          = nn.Parameter(torch.from_numpy(np.real(direction).astype(np.float32)))
        # Amplification weights
        self.amplifier_real     = nn.Parameter(torch.from_numpy(np.real(amplifier).astype(np.float32)))
        self.amplifier_image    = nn.Parameter(torch.from_numpy(np.imag(amplifier).astype(np.float32)))
        # bias
        self.omega              = [o.astype(np.float32) for o in omega]
        # Spectrum shifting
        self.shift              = [Shift(self.nodes[i], self.omega[i - 2]) for i in range(2, len(self.nodes) - 1)]
        # softmax
        if self.classifyEnable:
            self.softmax        = nn.Softmax(dim = -1)
        # Weight sort
        self.shape              = np.shape(amplifier)
        self.order              = np.argsort(np.reshape(np.abs(amplifier), -1))
    def forward(self, inputs):
        direction               = torch.matmul(inputs, self.direction)
        spectrum_real           = ComplexCos.apply(direction)
        spectrum_image          = ComplexSin.apply(direction)
        amplifier_real          = self.amplifier_real
        amplifier_image         = self.amplifier_image
        outputs_real            = torch.matmul(spectrum_real, amplifier_real) - torch.matmul(spectrum_image, amplifier_image)
        outputs_image           = torch.matmul(spectrum_real, amplifier_image) - torch.matmul(spectrum_image, amplifier_real)
        for i in range(len(self.shift)):
            outputs_real, outputs_image         = self.shift[i](inputs, outputs_real, outputs_image)
        outputs                 = (outputs_real.pow(2) + outputs_image.pow(2)).pow(0.5)
        if self.classifyEnable:
            outputs             = self.softmax(outputs)
        return outputs

# FSN Network generation
########################################################################################################################
def singleLayer_process(param):
    direction, amp, _           = singleLayer(param[0][0], 1, [0, 1, 0], param[0][1], solveEnable = param[1])
    return direction, amp
def FSN(spectrum, blocks, fs, device = torch.device('cpu'), complexEnable = False, solveEnable = True, classifyEnable = False, processes = None):
    """
    FSN(spectrum, blocks, fs, device = torch.device('cpu'), complexEnable = False, solveEnable = True, classifyEnable = False):
    Function Description:
        Generate the corresponding FSN network from the spectrum based on the block instructions.
    Parameter Description:
        spectrum                Frequency-domain data graph      —— [f_1, f_2, …… , f_n, value]
        blocks                  Block instruction                —— [[int_1, int_2, …… , int_n], [int_1, int_2, …… , int_n], ……]
        fs                      Sampling frequency(Hz)           —— [fs1, fs2, …… , fsn]
        device                  Model location                   —— CPU (Default) / GPU
        complexEnable           Real/plural network              —— False(Default): Real network/True: Plural network
        solveEnable             Solve enable                     —— True(Default) solve the amplifier and direction weights/False
        classifyEnable          Classify Enable                  —— Specify classification tasks: False(Default)/True
        processes               Number of processes             —— int (Default None - os.cpu_count())
    Return Description:
        model                   FSN network class                —— FSNModel()
    Test Example:
        import numpy as np
        import FSNanalysis as fsn
        import matplotlib.pyplot as plt
        import torch
        if __name__ == '__main__':
            fs                  = [400, 400]
            x                   = np.array([i / fs[0] for i in range(fs[0])], dtype = np.float32)
            y                   = np.array([i / fs[1] for i in range(fs[1])], dtype = np.float32)
            g                   = np.zeros((fs[0], fs[1], 2), dtype = np.float32)
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    g[i, j, 0]  = np.sin(10 * np.pi * (xi ** 2 + 2 * yj ** 2)) + 1
                    g[i, j, 1]  = np.cos(20 * np.pi * xi ** 2) + np.sin(10 * np.pi * yj ** 2) + 2
            spectrum, F         = fsn.dataProcess_FFT(g[..., 0: 2], fs)
            model               = fsn.FSN(spectrum, [[3, 3], [3, 3]], fs, complexEnable = False)
            xy                  = np.zeros((fs[0], fs[1], 2), dtype = np.float32)
            for i, xx in enumerate(x):
                for j, yy in enumerate(y):
                    xy[i, j, 0], xy[i, j, 1]    = (xx, yy)
            model.eval()
            with torch.no_grad():
                prediction      = model(torch.from_numpy(xy)).numpy()
            maximum             = [np.max(np.abs(prediction[: , : , 0])), np.max(np.abs(prediction[: , : , 1]))]
            (X, Y)              = np.meshgrid(y, x)
            loss                = g - prediction
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(g[: , : , 0]))
            plt.title('Original Funtion #1 (Magnitude)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(prediction[: , : , 0]))
            plt.title('Network Funtion #1 (Magnitude)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.angle(g[: , : , 0]))
            plt.title('Original Funtion #1 (Phase)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.angle(prediction[: , : , 0]))
            plt.title('Network Funtion #1 (Phase)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(loss[: , : , 0]))
            ax.set_zlim(0, maximum[0])
            plt.title('Loss #1')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(g[: , : , 1]))
            plt.title('Original Funtion #2 (Magnitude)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(prediction[: , : , 1]))
            plt.title('Network Funtion #2 (Magnitude)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.angle(g[: , : , 1]))
            plt.title('Original Funtion #2 (Phase)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.angle(prediction[: , : , 1]))
            plt.title('Network Funtion #2 (Phase)')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
            fig                 = plt.figure()
            ax                  = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, np.abs(loss[: , : , 1]))
            ax.set_zlim(0, maximum[1])
            plt.title('Loss #2')
            plt.xlabel('Input#2(s)')
            plt.ylabel('Input#1(s)')
            plt.show()
    """
    shape                       = np.shape(spectrum)
    feature_num                 = len(shape) - 1
    output_num                  = np.shape(spectrum)[-1]
    spec                        = [spectrum[..., i] for i in range(output_num)]
    unit                        = [fs[i] / shape[i] for i in range(feature_num)]
    bias_spec                   = []
    # Each layer of the network blocks of the spectrum (in reverse).
    for b in blocks:
        # Obtain the relative coordinates of the spectral blocks
        coor_block              = [[]]
        for length in b:
            tem                 = []
            for c in coor_block:
                for i in range(-(length // 2), length - length // 2):
                    tem.append(c + [i])
            coor_block          = tem
        bias_spec.append([])
        tem                     = []
        for i, s in enumerate(spec):
            # Spectrum partitioning
            s_tem, b_tem, _     = divide(unit, b, s, solveEnable)
            for c in coor_block:
                tem.append(s_tem[tuple(c)])
                if i == 0:
                    bias_spec[-1].append(b_tem[tuple(c)])
        spec                    = tem
    bias_spec.reverse()
    bias_spec                   = [np.array(b) for b in bias_spec]
    # Calculate the direction weights and the amplification weights
    amplifier                   = []
    fs                          = [([unit[i] * np.shape(s)[i] for i in range(feature_num)], s) for s in spec]
    with Pool(processes = processes) as pool:
        results                 = pool.map(singleLayer_process, paraList(fs, selfIter(solveEnable, len(fs))))
    for weights in results:
        amplifier.append(weights[1])
    amplifier                   = np.concatenate(tuple(amplifier), axis = 1)
    if complexEnable:
        model                   = FSNModel_Complex(blocks, bias_spec, weights[0].T, amplifier, classifyEnable)
    else:
        model                   = FSNModel(blocks, bias_spec, weights[0].T, amplifier, classifyEnable)
    model.to(device)
    for shift in model.shift:
        shift.to(device)
    return model

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# Conversion from real number output to complex number output
########################################################################################################################
def real2complex(tensor):
    chunks                      = torch.chunk(tensor, chunk = 2, dim = -1)
    result                      = chunks[0] + chunks[1] * 1j
    return result

# Custom dataset class
########################################################################################################################
class customDataset(Dataset):
    def __init__(self, data, device):
        self.data               = {'input': [], 'output': []}
        self.length             = len(data['input'])
        for index in range(self.length):
            self.data['input'].append(torch.from_numpy(data['input'][index]))
            self.data['output'].append(torch.from_numpy(data['output'][index]))
            self.data['input'][-1].to(device)
            self.data['output'][-1].to(device)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.data['input'][index], self.data['output'][index]

# Data set structure transformation
########################################################################################################################
def dataTransform(data, batch_size, shuffle = True, device = torch.device('cpu')):
    """
    dataTransform(data, batch_size, shuffle = True, device = torch.device('cpu')):
    Function Description:
        Convert the data structure to dataloader.
    Parameter Description:
        data                    Data set                         —— {'input': [], 'output': []}
        batch_size              Batch size                       —— int
        shuffle                 Shuffling enable                 —— True/False
        device                  Model location                   —— CPU (Default) / GPU
    Return Description:
        dataset                 Data loader                      —— torch.utils.data.DataLoader
    """
    customData                  = customDataset(data, device)
    return torch.utils.data.DataLoader(customData, batch_size = batch_size, shuffle = shuffle)

# Prediction from dataset
########################################################################################################################
def predict(data, model, device):
    """
    predict(data, model, device):
    Function Description:
        Predict outputs from the inputs of dataset.
    Parameter Description:
        data                    Data set                         —— {'input': [], 'output': []}
        model                   FSN network class                —— FSNModel()
        device                  device                           —— cpu/gpu
    Return Description:
        data                    Data set                         —— {'input': [], 'output': []}
    """
    dataloader                  = dataTransform(data, 1, shuffle = False)
    data['output']              = []
    for inputs, outputs in dataloader:
        inputs                  = inputs.to(device)
        data['output'].append(model(inputs)[0].to(torch.device('cpu')).numpy())
    data['output']              = np.array(data['output'], dtype = np.float32)
    return data

# FSN training module
########################################################################################################################
def train(model, dataloader, device, lossFunction, optimizer):
    """
    train(model, dataloader, device, lossFunction, optimizer):
    Function Description:
        Train the network.
    Parameter Description:
        model                   FSN network class               —— FSNModel()
        dataloader              Data loader                     —— torch.utils.data.DataLoader
        device                  device                          —— cpu/gpu
        lossFunction            Loss function                   —— torch.nn.Module()
        optimizer               Optimizer                       —— torch.optim.Optimizer()
    Return Description:
        model                   FSN network class               —— FSNModel()
    """
    model.train()
    for batch, (inputs, outputs) in enumerate(dataloader):
        inputs, outputs         = inputs.to(device), outputs.to(device)
        prediction              = model(inputs)
        loss                    = lossFunction(prediction, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return

# FSN test module
########################################################################################################################
def test(model, dataloader, device, lossFunction = None, classifyEnable = True, complexEnable = False, printEnable = True):
    """
    test(model, dataloader, device, lossFunction = None, classifyEnable = True, complexEnable = False, printEnable = True):
    Function Description:
        Evaluate the network.
    Parameter Description:
        model                   FSN network class               —— FSNModel()
        dataloader              Data loader                     —— torch.utils.data.DataLoader
        device                  device                          —— cpu/gpu
        lossFunction            Loss function                   —— torch.nn.Module()
        classifyEnable          Classification enable           —— True(Default)/False, Used to determine whether to output the classification result
        complexEnable           Real/plural network             —— False(Default): Real network/True: Plural network
        printEnable             Print enable for performance    —— True(Default)/False
    Return Description:
        result                  Loss/Accuracy rate              —— float
    """
    result                      = 0
    for inputs, outputs in dataloader:
        output_num              = outputs.shape[-1]
        break
    mix                         = np.zeros((output_num, output_num))
    with torch.no_grad():
        for inputs, outputs in dataloader:
            inputs, outputs     = inputs.to(device), outputs.to(device)
            prediction          = model(inputs)
            if classifyEnable:
                pre             = prediction.argmax(1)
                out             = outputs.argmax(1)
                result         += (pre == out).type(torch.float).sum().item()
                shape           = prediction.shape
                for i in range(shape[0]):
                    mix[int(pre[i].item()), int(out[i].item())]+= 1
            else:
                # prediction      = torch.unsqueeze((prediction[..., 0] ** 2 + prediction[..., 1] ** 2) ** 0.5, 1)
                result         += lossFunction(prediction, outputs).item()
        if classifyEnable:
            result             /= len(dataloader.dataset)
            if printEnable:
                print(f"Accuracy: {(100 * result): >0.1f}%")
        else:
            result             /= len(dataloader)
            if printEnable:
                print(f"Avg loss: {result: >8f}")
    return result, mix

# Model storage
########################################################################################################################
def save(path, filename, model):
    """
    save(path, filename, model):
    Function Description:
        Store the parameters of the FSN model.
    Parameter Description:
        path                    Save path                       —— str
        filename                File name                       —— str
        model                   FSN network class               —— FSNModel()
    """
    torch.save(model.state_dict(), path + '/' + filename + '.pth')
    return

# Model loading
########################################################################################################################
def load(path, filename, model, device):
    """
    load(path, filename, model, device):
    Function Description:
        Read the parameters of the FSN model.
    Parameter Description:
        path                    Save path                       —— str
        filename                File name                       —— str
        model                   FSN network class               —— FSNModel()
        device                  device                          —— cpu/gpu
    """
    model.load_state_dict(torch.load(path + '/' + filename + '.pth', map_location = device))
    return

