import numpy as np
import pandas as pd
import sys
#### my modules
from core.Config import DIGITS, SAMPLE, SAMPLE_TEST, RATIO, BATCH_SIZE
dir_data = './data/'


class mnist:
    def __init__(self):
        self.digits = None
        self.size_max = None
        self.labels = None
        self.T = None
        self.X = None
        self.X_test = None
        self.sample = None
        self.sample_validation = None
        self.sample_test = None
        self.batch_size = None
        self.n_of_batches = None
        self.slice_train = None # handle batches with slices
        self.slice_train_flat = None # handle batches with slices
        self.slice_validation = None
        self.slice_test = None
        self.load_training_data()
        self.load_testing_data()
        self.settings()

    def __str__(self):
        not_initialized = [ k for k in self.__dict__.keys() if self.__dict__[k] is None]
        initialized = [k for k in self.__dict__.keys() if k not in not_initialized]
        out = []
        out.append(f' ------------ uninitialized attributes:')
        for k, val in not_initialized:
            out.append[f'{k:>20} = {self.__dict__[k]}']
        out.append(f' ------------- initialized attributes:')
        for k in initialized:
            if isinstance(self.__dict__[k], (list, slice, np.ndarray)) and k != 'digits':
                out.append(f'{k:>20} shape : {np.shape(self.__dict__[k])}')
            else:
                out.append(f'{k:>20} = {self.__dict__[k]}')
        ratio = self.get_ratio()
        if ratio is not None:
            out.append(f'{'non-attribute ratio':>20} = {ratio:.3f}')
        return '\n'.join(out)

    def get_ratio(self):
        ratio = None
        if self.sample is not None and self.sample_validation is not None:
            ratio = self.sample / (self.sample + self.sample_validation)
        return ratio

    def load_training_data(self):
        """
        load all training MNIST data and return 1D images and labels
        Return values:
        X: np matrix with 28*28 = 784 rows and 42000 columns. Each columns is one images.
        Matrix elements originally in [0, 256] are normalized to be in  [0, 1]
        T: np matrix with 10 rowns and 42000 columns. Each columns is a label in one-hot form (bool valued entries)
        """
        df = pd.read_csv(dir_data + 'train.csv')
        X_raw = np.array(df.loc[:,'pixel0':]).T
        X_raw = X_raw / (X_raw.max() - X_raw.min()  )
        X = np.ones((np.shape(X_raw)[0] + 1, np.shape(X_raw)[1]))
        X[1:,:] = X_raw[:]
        T_raw = np.array(df.loc[:,'label'])
        T = np.array([[t == digit for t in T_raw] for digit in range(10)])
        self.X = np.copy(X)
        self.T = T.copy()
        self.labels = T_raw.copy()

    def load_testing_data(self):
        """
        load all testing MNIST data and return 1D images
        Return values:
        X: np matrix with 28*28 = 784 rows and 42000 columns. Each columns is one images.
        """
        df = pd.read_csv(dir_data + 'test.csv')
        X_raw = np.array(df.loc[:,'pixel0':]).T
        X_raw = X_raw / (X_raw.max() - X_raw.min()  )
        X = np.ones((np.shape(X_raw)[0] + 1, np.shape(X_raw)[1]))
        X[1:,:] = X_raw[:]
        self.X_test = np.asfortranarray(X)

    def set_size_max(self):
        size_max = len([label for label in self.labels if label in self.digits])
        self.size_max = size_max

    def X_T_ok(self):
        ok = True
        if self.X is None:
            print('X array is None')
            ok = False
        elif self.T is None:
            print('T array is None')
            ok = False
        elif np.shape(self.X)[1] != np.shape(self.T)[1]:
            print(f'X and T array have different sizes: {np.shape(self.X)[1]} and {np.shape(self.T)[1]}')
            ok = False
        return ok

    def set_sample_sizes(self, sample=SAMPLE, ratio=RATIO):
        """
        sample: requested size of training data
        ratio: requested value of sample / total pts
        enforce:
            r in [0, 1]
            total <= data size
        returns sample sizes for training and validation sets:
            sample, sample_validation
        """
        sample_validation = None
        if self.digits is None:
            self.digits = DIGITS
        self.set_size_max()

        if not self.X_T_ok():
            print('setting sample = sample_validation = None')
            sample = None
        else:

            size_max = self.size_max

            if ratio < 0.75 and ratio > 0:
                print(f'warning: ratio < 0.75, but prefer ratio > 0.75')
            if ratio > 1 or ratio < 0:
                print(f'ratio = {ratio} is out of interval [0, 1], using ratio = {RATIO}')
                ratio = RATIO
            if int(sample / ratio) > size_max:
                print(f'not enough data to set sample = {sample} and ratio = {ratio}\n'
                      f'need {int(sample / ratio)} points but have {size_max}\n'
                      f'using sample = {int(ratio * size_max)}')
                sample = int(ratio * size_max)

            sample_validation = round((1 - ratio) / ratio * sample)
            if sample + sample_validation < size_max:
                print(f'warning: sample + sample_validation = {sample + sample_validation} does not span available data\n'
                      f'         leaving out {size_max - sample - sample_validation} pts')
            self.sample = sample
            self.sample_validation = sample_validation

    def set_batch_sizes(self, batch_size=BATCH_SIZE):
        """
        enforce all batches of equal size, print warning
        if points have been left out. If self.sample has not been set,
        if calls set_sample_sizes
        returns n_of_batches, batch_size
        """
        if not self.X_T_ok():
            print('setting n_of_batches = batch_size = None')
            self.n_of_batches = None
            self.batch_size = None
        else:
            if (self.sample is None) or (self.sample_validation is None):
                print(f'found:   sample is None = {self.sample is None}\n'
                      f'         sample_validation is None = {self.sample_validation is None}\n')
                self.set_sample_sizes()
                print(f'         setting defaults sample = {self.sample}  sample_validation = {self.sample_validation}')
            if self.sample + self.sample_validation > self.size_max:
                print(f'sample + sample_validation > size_max for digits = {self.digits}'
                      f'calling set_sample_sizes with arguments:\n'
                      f'    sample = {RATIO * self.size_max}   ratio = {RATIO}')
                self.set_sample_sizes(int(RATIO * self.size_max), RATIO)
            if batch_size > self.sample:
                print(f'batch_size is larger than self.sample, setting batch_size = sample = {self.sample}')
                self.batch_size = self.sample
            else:
                self.batch_size = batch_size
            self.n_of_batches = self.sample // self.batch_size
            self.batch_size = self.sample // self.n_of_batches

            effective_pts = self.batch_size * self.n_of_batches
            if effective_pts != self.sample:
                print(f'warning: effective = {effective_pts}   total = {self.sample}\n'
                      f'         leaving out {self.sample - effective_pts} pts')

    def set_sample_test_size(self, sample_test=SAMPLE_TEST):
        if self.X_test is None:
            print('X_test array is None, nothing to do')
        else:
            if sample_test > np.shape(self.X_test)[1]:
                sample_test = SAMPLE_TEST
                print(f'sample_test is larger than total data available, setting default sample_test = {SAMPLE_TEST}')
            self.sample_test = sample_test

    def shuffle(self):
        indices = np.arange(0, np.shape(self.X)[1])
        np.random.default_rng().shuffle(indices)
        self.X = self.X[:,indices]
        self.T = self.T[:,indices]
        self.labels = self.labels[indices]

    def set_slice_train(self):
        if self.batch_size is None or self.n_of_batches is None:
            print(f'found:   batch_size is None = {self.batch_size is None}\n'
                  f'         n_of_batches is None = {self.n_of_batches is None}\n')
            self.set_batch_sizes()
            print(f'         setting defaults batch_size = {self.batch_size}  n_of_batches = {self.n_of_batches}')
        indices = []
        self.shuffle()
        all_digits_present = np.prod([(i in self.digits) for i in range(10)]) == 1
        if all_digits_present:
            indices = [i for i in range(np.shape(self.X)[1])]
        else:
            indices_in_digits = [i for i in range(np.shape(self.X)[1]) if self.labels[i] in self.digits]
            indices_nin_digits = [i for i in range(np.shape(self.X)[1]) if i not in indices_in_digits]
            indices = np.append(indices_in_digits, indices_nin_digits).astype(int)
        self.X = np.asfortranarray(self.X[:,indices])
        self.T = np.asfortranarray(self.T[:,indices])
        self.labels = np.asfortranarray(self.labels[indices])
        self.slice_train = [(slice(None), slice(self.batch_size * i, self.batch_size * i + self.batch_size, 1)) for i in range(self.n_of_batches)]
        self.slice_train_flat = (slice(None), slice(0, self.batch_size * self.n_of_batches))

    #WARNING: needs to be called after set_sample_sizes and set_slice_train always
    def set_slice_validation(self):
        begin = self.sample
        end = self.sample + self.sample_validation
        self.slice_validation = (slice(None), slice(begin, end, 1))

    def set_slice_test(self):
        if self.sample_test is None:
            self.set_sample_test_size()
        self.slice_test = ((slice(None), slice(0, self.sample_test, 1)))

    def settings(self, digits=None, sample=SAMPLE, ratio=RATIO, batch_size=BATCH_SIZE, sample_test=SAMPLE_TEST):
        if digits is None:
            digits = DIGITS
        for k in self.__dict__.keys():
            if k not in ['X', 'T', 'labels', 'X_test']:
                self.__dict__[k] = None
        self.digits = digits
        self.set_size_max()
        self.set_sample_sizes(sample, ratio)
        self.set_batch_sizes(batch_size)
        self.set_slice_train()
        self.set_slice_validation()
        self.set_slice_test()

        attr_not_set = [k for k in self.__dict__.keys() if (self.__dict__[k] is None)]
        # if len(attr_not_set) == 0:
        #     print(f'-----------> settings: all attributes set')
        # else:
        #     print(f'-----------> settings: some attributes have not been set:')
        #     for k in attr_not_set:
        #         print(f'{k:<100}')

    def config_settings(self, config):
        self.settings(digits = config.digits, sample = config.sample, ratio = config.ratio,
                 batch_size = config.batch_size, sample_test = config.sample_test)

    def images(self, X):
        """
        take a set of 1D image and return a numpy array corresponding to a 2D version, 28 X 28 matrix
        """
        pixels, n_of_images = np.shape(X)
        side = int(pixels**0.5)
        Images = np.copy(X[1:,:])
        Images = Images.T.reshape((n_of_images, side, side))
        return Images






