import subprocess
import sys
import pickle
import uuid
from   pathlib import Path
import numpy as np
#### my modules
sys.path.insert(0, "./modules/")
from Grids import Grids
from Config import Config
from functions import logit, relu, cross_entropy, activation_dict, cost_dict
from backpropagation import forward_step_last, forward_step, backward_last_layer, backward_step
import load

CACHE_DIR = '.cache/'

class nn:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.data = load.mnist()
        self.data.config_settings(self.config)
        self.grids_train = Grids(config)
        self.grids_validation = Grids(config)

        self.activation_hidden = activation_dict[config.activation_hidden]
        self.cost = cost_dict[config.cost]
        self.states      = ['','']
        self.cache_dir   = CACHE_DIR
        self.C_train = []                   #cost
        self.C_validation = []
        self.Acc_train = []                 #accuracy
        self.Acc_validation = []

        self._data_loaded = False
        self.initialize_weights = True
        self.end_training = False
        self.pause_training = False
        self.training_iterator = None

    def __sync_grids(self):
        if isinstance(self.grids_train, Grids) and isinstance(self.grids_validation, Grids):
            self.grids_validation.W = self.grids_train.W
            self.grids_train._Grids__update_info()
            self.grids_validation._Grids__update_info()
        else:
            print(f'could not sync grids:')
            print(f'isinstance(self.grids_train, Grids) = {isinstance(self.grids_train, Grids)}')
            print(f'isinstance(self.grids_validation, Grids) = {isinstance(self.grids_validation, Grids)}')
            sys.exit()

    def last_index(self):
        return len(self.config.hidden_nodes)

    def start_weights(self, rng_or_int=np.random.default_rng()):
        self.grids_train.fill_weights(rng_or_int=rng_or_int)
        self.__sync_grids()

    def forward_pass(self, grids):
        last = self.last_index()
        if last == 0:
            Z = grids.Z[last]
            A = grids.A[last][1:,:]
            W = grids.W[last]
            X = grids.X
            Z[:,:], A[:,:] = forward_step_last(W, X)
        else:
            Z = grids.Z[0]
            A = grids.A[0][1:,:]
            DF = grids.DF[0]
            W = grids.W[0]
            X = grids.X
            Z[:,:], A[:,:], DF[:,:] = forward_step(W, X, self.activation_hidden[0], self.activation_hidden[1])
            for l in range(1,last):
                Z = grids.Z[l]
                A = grids.A[l][1:,:]
                DF = grids.DF[l]
                W = grids.W[l]
                A_before = grids.A[l-1]
                Z[:,:], A[:,:], DF[:,:] = forward_step(W, A_before, self.activation_hidden[0], self.activation_hidden[1])
            Z = grids.Z[last]
            A = grids.A[last][1:,:]
            W = grids.W[last]
            A_before = grids.A[last-1]
            Z[:,:], A[:,:] = forward_step_last(W, A_before)

    def backward_pass(self):
        last = self.last_index()
        if last == 0:
            Delta = self.grids_train.Delta[last]
            Gradient_W = self.grids_train.Gradient_W[last]
            T = self.grids_train.T
            A = self.grids_train.A[last]
            X = self.grids_train.X
            Delta[:,:], Gradient_W[::] = backward_last_layer(T, A, X,  self.cost[1])
        else:
            Delta = self.grids_train.Delta[last]
            Gradient_W = self.grids_train.Gradient_W[last]
            T = self.grids_train.T
            A = self.grids_train.A[last]
            A_previous = self.grids_train.A[last-1]
            Delta[:,:], Gradient_W[:,:] = backward_last_layer(T, A, A_previous,  self.cost[1])
            for l in range(last-1,0,-1):
                Delta = self.grids_train.Delta[l]
                Gradient_W = self.grids_train.Gradient_W[l]
                DF = self.grids_train.DF[l]
                W_next = self.grids_train.W[l+1][:,1:]
                Delta_next = self.grids_train.Delta[l+1]
                A_previous = self.grids_train.A[l-1]
                Delta[:,:], Gradient_W[:,:] = backward_step(DF, W_next, Delta_next, A_previous)
            Delta = self.grids_train.Delta[0]
            Gradient_W = self.grids_train.Gradient_W[0]
            DF = self.grids_train.DF[0]
            W_next = self.grids_train.W[1][:,1:]
            Delta_next = self.grids_train.Delta[1]
            X = self.grids_train.X
            Delta[:,:], Gradient_W[:,:] = backward_step(DF, W_next, Delta_next, X)

    def update_weights(self):
        for l in range(len(self.grids_train.W)):
            self.grids_train.W[l] = self.grids_train.W[l] - self.config.learning_rate * self.grids_train.Gradient_W[l]
        self.__sync_grids()

    def forward_backward_pass(self, do_backward=True):
        self.forward_pass(self.grids_train)
        if do_backward:
            self.backward_pass()
            self.update_weights()
        return  self.cost[0](self.grids_train.output(),self.grids_train.T)

    def total_cost(self):
        output = 0
        must_update_grids = True
        for s in self.data.slice_train:
            self.grids_train.set_X(self.data.X[s])
            self.grids_train.set_T(self.data.T[s])
            output += self.forward_backward_pass(do_backward=False)
            if must_update_grids:
                self.grids_train._Grids__update_info()
                must_update_grids = False
        return output / self.data.n_of_batches

    def validation_pass(self):
        self.forward_pass(self.grids_validation)
        output = self.cost[0](self.grids_validation.output(), self.grids_validation.T)
        return output

    def prediction_pass(self):
        self.forward_pass(self.grids_train)

    def train_setup(self):
        self.grids_validation.set_X(self.data.X[self.data.slice_validation])
        self.grids_validation.set_T(self.data.T[self.data.slice_validation])
        epochs = self.config.epochs
        if self.initialize_weights:
            self.start_weights()
            #### BEGIN: set weights by subtracting initial values
            W_init = [ weights.copy() for weights in self.grids_train.W ]
            self.grids_train.set_X(self.data.X[self.data.slice_train[0]])
            self.grids_train.set_T(self.data.T[self.data.slice_train[0]])
            self.forward_backward_pass()
            self.grids_train.W = [self.grids_train.W[i] - W_init[i] for i in range(len(W_init))]
            #### END
            self.__sync_grids()
        self.C_train = np.full(epochs, None)
        self.C_validation = np.full(epochs, None)
        self.Acc_train = np.full(epochs, None)
        self.Acc_validation = np.full(epochs, None)
        self.pause_training = False
        self.end_training = False
        self.training_iterator = (i for i in range(epochs))

        self.grids_train.counter_temp = 0
        self.grids_train.counter_temp_2 = [0, 0]
        self.grids_validation.counter_temp = 0
        self.grids_validation.counter_temp_2 = [0, 0]

    def train_one_epoch(self, i):
        epochs = self.config.epochs
        print_steps = self.config.print_steps
        self.C_train[i] = self.total_cost()
        self.C_validation[i] = self.validation_pass()
        self.Acc_train[i] = self.success_rate()[0]
        self.Acc_validation[i] = self.success_rate(validation=True)[0]
        if( print_steps != 0 and (i % max(1, epochs//print_steps) == 0 or i == epochs - 1 ) ):
            if i == 0:
                print('\n')
                print(f'{"epoch":<8} {"training cost":<15} {"validation cost":<15}')
            print(f'{i:<8} {self.C_train[i]:<15.4f} {self.C_validation[i]:<15.4f}')
        counter_loop = 0
        for s in self.data.slice_train:
            self.grids_train.set_X(self.data.X[s], self.data)
            self.grids_train.set_T(self.data.T[s])
            self.forward_backward_pass()
            counter_loop += 1

    def train(self):
        self.train_setup()
        for i in self.training_iterator:
            self.train_one_epoch(i)
            while self.pause_training:
                if self.end_training:
                        break
            if self.end_training:
                break

    def pause_continue_training(self):
        self.pause_training = not self.pause_training

    def end(self):
        self.end_training = True

    def get_one_hot(self, validation = False):
        output = []
        true = []
        self.grids_train._Grids__update_info()
        s = None                                             # slice object to get data
        X = self.data.X
        T = self.data.T
        output_dimension = self.config.output_dimension

        if validation:
            s = self.data.slice_validation
            self.grids_validation.set_X(X[s])
            self.validation_pass()
            output = self.grids_validation.output()
            true = T[s]
        else:
            shape = np.shape(T[self.data.slice_train_flat])
            output = np.zeros(shape)
            true = np.zeros(shape)
            for s in self.data.slice_train:
                self.grids_train.set_X(X[s])
                self.forward_backward_pass(do_backward=False)
                output[s] = self.grids_train.output()
                true[s] = T[s]

        output_max = np.max(output, axis = 0)
        output_one_hot = np.where(output == output_max, 1, 0)

        return output_one_hot, true

    def success_rate(self, validation = False):
        output_one_hot, true = self.get_one_hot(validation)
        success_array = ( output_one_hot == true )
        success_array = np.prod(success_array, axis = 0)
        total = success_array.shape[0]
        numof_success = np.sum(success_array)
        return (numof_success / total), numof_success, total

    def predict(self, X):
        self.grids_train._Grids__update_info()
        self.grids_train.set_X(X)
        self.prediction_pass()
        output = self.grids_train.output()
        return np.argmax(output, axis=0)

    def save_state(self):
        path_to_save = self.cache_dir + self.config.runid + '/'
        Path(path_to_save).mkdir(parents = True, exist_ok = True)
        fname = f'{path_to_save}{uuid.uuid4()}-{type(self).__name__}.pkl'
        self.states[0] = self.states[1]
        self.states[1] = fname
        print(f'# saving current state to "{self.states[1]}"')
        states  = self.states
        config  = self.config
        weights = self.grids_train.W
        with open(self.states[-1],'xb') as f:
            pickle.dump(states , f)
            pickle.dump(config , f)
            pickle.dump(weights, f)
            pickle.dump(self.C_train, f)
            pickle.dump(self.C_validation, f)

    def load_state(self):
        path_to_load = self.cache_dir + self.config.runid + '/'
        cmd = ['ls','-t',path_to_load]
        process = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        Filenames, err = process.communicate()
        if(len(Filenames) == 0):
            self.initialize_weights = True
            print(f'#no available state to load in {path_to_load}')
        else:
            self.initialize_weights = False
            filename = Filenames.split()[0].decode('utf-8')
            filename = path_to_load + filename
            print(f'#loading {filename}')
            with open(filename, 'rb') as f:
                states  = pickle.load(f)
                config  = pickle.load(f)
                weights = pickle.load(f)
                self.C_train = pickle.load(f)
                self.C_validation = pickle.load(f)
                self.states = [np.array(states [i]) for i in range(len(states))]
                self.sync_to(config, force_weights=True)                                   #must call before setting weights
                self.grids_train.W = [np.array(weights[i]) for i in range(len(weights))]
                self.__sync_grids()

    # some attributes are not loaded for flexibility in main
    def sync_to(self, config, full=False, force_weights=False): # previously load_config
        if full:
            self.config = config
        else:
            self.config.hidden_nodes = config.hidden_nodes
            self.config.input_dimension = config.input_dimension
            self.config.output_dimension = config.output_dimension
            self.config.runid = config.runid
            self.config.activation_hidden = config.activation_hidden
            self.config.cost = config.cost
        self.activation_hidden = activation_dict[config.activation_hidden]
        self.grids_train.set_weight_dimensions(config=config, force=force_weights)                 # these two needed for call
        self.grids_validation.set_weight_dimensions(config=config, force=force_weights)            # to set_layer_dimensions
        self.grids_train.set_layer_dimensions()
        self.grids_validation.set_layer_dimensions()
        self.__sync_grids()




