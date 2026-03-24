import sys
import numpy as np
#### my modules
sys.path.insert(0, "./modules/")
from Config import Config

class Grids:
    def __init__(self, config=None):
        if config is None:
            config = Config()
        self.weights_empty = True
        self.layers_empty = True
        self.X = np.array([])
        self.T = np.array([])
        self.W = []
        self.A = []
        self.Z = []
        self.DF = [] # elements of self.DF[self.layers-1] remain zero
        self.Delta = []
        self.Gradient_W = []
        self.set_weight_dimensions(config)
        self.__update_info()
        self.counter_reshape = 0

    # check only shapes of grids (values stored are not checked)
    def __str__(self):
        var_keys = self.__dict__.keys()
        var_dict = self.__dict__
        self.__update_info()
        shape = self.shape
        s  = ['#### bool attributes ####']
        s += [f'#\t{key} = {var_dict[key]}' for key in var_keys if not isinstance(var_dict[key], (list, np.ndarray, dict))]
        s += ['#### uninitialized lists ####']
        s += [f'#\t{key} = {np.shape(var_dict[key])}' for key in var_keys if isinstance(var_dict[key], (list, np.ndarray)) and len(var_dict[key]) == 0]
        s += ['####   initialized lists ####']
        s2 = '#       '
        for key in var_keys:
            if isinstance(var_dict[key], (list, np.ndarray)) and len(var_dict[key]) != 0:
                s2 += '    '.join([f'{key}[{i}] = {shape[key][i]}' for i in range(len(shape[key]))])
                s2 +='\n#       '
        s += [s2]
        return '\n'.join(s)

    def has_dimensions_of(self, other):
        matching_object = isinstance(other, type(self))
        grid_sizes_are_equal = True
        if matching_object:
            keys = (key for key in self.__dict__.keys() if isinstance(self.__dict__[key], (list, np.ndarray)))
            for attr in keys:
                shape_self = np.shape(self.__dict__[attr])
                shape_other = np.shape(other.__dict__[attr])
                grid_sizes_are_equal = grid_sizes_are_equal and np.array_equal(shape_self, shape_other)
        return (matching_object and grid_sizes_are_equal)

    def __get_initialized(self):
        initialized = {entry: self.__dict__[entry] for entry in self.__dict__.keys() if isinstance(self.__dict__[entry], (list, np.ndarray))}
        initialized = {entry: len(initialized[entry]) != 0 for entry in initialized.keys()}
        return initialized

    def __get_shape(self):
        shape_initialized = {key: [np.shape(array) for array in self.__dict__[key]] for key in self.initialized.keys() if self.initialized[key]}
        shape_not_initialized = {key: [] for key in self.initialized.keys() if not self.initialized[key] }
        shape = shape_initialized | shape_not_initialized
        shape['X'] = np.shape(self.X)
        shape['T'] = np.shape(self.T)
        return shape

    def __get_size(self):
        size = {key: len(self.shape[key]) for key in self.shape.keys()}
        size['X'] = np.shape(self.X)[-1]
        size['T'] = np.shape(self.T)[-1]
        return size

    def __update_info(self):
        self.initialized = self.__get_initialized()
        self.shape = self.__get_shape()
        self.size = self.__get_size()
        self.weights_empty = not bool(self.W)
        self.layers_empty = not bool(self.A)

    def __mknodes(self, config):
        """
        input is used to update weight and gradient dimensions
        returns a list with nodes matching input config
        """
        nodes = np.ones(len(config.hidden_nodes) + 2, dtype=int)
        nodes[1:-1] = config.hidden_nodes
        nodes[0] = config.input_dimension
        nodes[-1] = config.output_dimension
        return nodes

    def set_weight_dimensions(self, config=None, force=False):  # previously __config_weights
        if force:
            self.weights_empty = True
        if self.weights_empty:
            nodes = self.__mknodes(config)
            self.W = [np.zeros((nodes[i], nodes[i-1] + 1)) for i in range(1,nodes.size)]
            self.Gradient_W = [np.zeros((nodes[i], nodes[i-1] + 1)) for i in range(1,nodes.size)]
            self.__update_info()
            # for i in range(len(self.W)):
            #     print(f'----------------- >>>>>>>> setting weight dimension layer {i}: {np.shape(self.W[i])}')
    # uses default values of config
    def __mkconfig(self):
        config = Config()
        self.__update_info()
        if self.initialized['W']:
            config.hidden_nodes = [ self.shape['W'][i][0] for i in range(len(self.shape['W']) - 1) ]
            config.input_dimension = self.shape['W'][0][1] - 1
            config.output_dimension = self.shape['W'][-1][0]
        else:
            print('weights are not initialized, returning default instance of Config')
            self.set_weight_dimensions(config)
        return config

    def __setup_layer_grids(self, pts=1):
        config = self.__mkconfig()
        nodes = self.__mknodes(config)
        self.A = [np.zeros((nodes[i] + 1, pts )) for i in range(1, nodes.size)]
        self.Z = [np.zeros((nodes[i], pts )) for i in range(1, nodes.size)]
        self.Delta = [np.zeros((nodes[i], pts )) for i in range(1,nodes.size)]

    def __setup_DF_grid(self, pts=1):
        config = self.__mkconfig()
        nodes = self.__mknodes(config)
        self.DF = [np.zeros((nodes[i], pts )) for i in range(1, nodes.size)]

    def set_layer_dimensions(self, pts=None):
        if pts is None:
            if np.ndim(self.X) == 2:
                pts = np.shape(self.X)[1]
            else:
                pts = 1
        self.__setup_layer_grids(pts)
        self.__setup_DF_grid(pts)

    def weights_have_dimensions_of(self, weights):
        same_dimensions = False
        if not isinstance(weights, list):
            print(f'# weights are not a list, ignoring')
        elif len(weights) == 0:
            print(f'# weights are an empty list')
        else:
            same_dimensions = [ np.shape(weights[i])[0] == np.shape(weights[i+1])[1]-1 for i in range(len(weights) - 1)]
            same_dimensions = bool(np.prod(same_dimensions))
        return same_dimensions

    def set_X(self, X, data=None):
        config = self.__mkconfig()
        shape = (0,)
        if np.shape(X)[0] - 1 != config.input_dimension:
            sys.exit(f'Error: input X has incompatible dimension, aborting: {np.shape(X)[0] - 1} vs {config.input_dimension}')
        elif (np.ndim(self.X) == 2) and (X.shape == np.shape(self.X)):
            self.X = X
        else:
            self.counter_reshape += 1
            shape = np.shape(X)
            self.__setup_layer_grids(shape[1])
            self.__setup_DF_grid(shape[1])
            self.X = X

    def set_T(self, T):
        config = self.__mkconfig()
        shape = (0,)
        if np.shape(T)[0] != config.output_dimension:
            sys.exit(f'Error: input T has incompatible dimension, aborting: {np.shape(T)[0]} vs {config.output_dimension}')
        self.T = T

    def output(self):
        if len(self.A) == 0:
            sys.exit(f'Error: requesting output but A arrays have not been initialized, aborting.')
        else:
            return self.A[-1][1:,:]

    def fill_weights(self, rng_or_int=np.random.default_rng()):
        if isinstance(rng_or_int, int):
            for i in range(len(self.W)):
                self.W[i][:,:]  = rng_or_int
        else:
            delta = 0.1
            for i in range(len(self.W)):
                self.W[i] = rng_or_int.uniform(-delta, delta, self.W[i].shape)

    def check(self):
        print(f'# checking array sizes')
        self.__update_info()
        if sum(list(self.initialized.values())) == 0:
            print(f'# all arrays are empty')
        else:
            if self.initialized['W']:
                if self.weights_have_dimensions_of(self.W):
                    print(f'#   W arrays match')
                else:
                    print(f'#   W arrays do not match, shapes:  {self.shape["W"]}')
                if np.array_equal(self.shape['W'], self.shape['Gradient_W']):
                    print(f'#   Gradient_W arrays match')
                else:
                    print(f'#   Gradient_W and W shapes do not match, \n#   Gradient_W shapes:')
                    for i in range(len(self.shape['Gradient_W'])):
                        print(f'#    Gradient_W[{i}]: {self.shape["Gradient_W"][i]}')
                    print(f'#   W shapes:')
                    for i in range(len(self.shape['W'])):
                        print(f'#     W[{i}]: {self.shape["W"][i]}')
            if self.initialized['X']:
                arrays_match = self.shape['X'][0] == self.shape['W'][0][1]
                if arrays_match:
                    print(f'#   W[0] and X match')
                else:
                    print(f'#   W[0] and X do not match, shapes:  {self.shape['W'][0]}  {self.shape["X"]}')
            if self.initialized['A']:
                arrays_match = [ self.shape["W"][i][1] == self.shape["A"][i-1][0] for i in range(1, len(self.shape['A']))]
                arrays_match = bool(np.prod(arrays_match))
                if arrays_match:
                    print(f'#   W[i] and A[i] match')
                else:
                    print(f'#   W[i] and A[i] do not match, shapes:')
                    for i in range(1, len(self.shape['A'])):
                        print(f'#    W[{i}]: {self.shape["W"][i]}   A[{i-1}]: {self.shape["A"][i-1]}')

