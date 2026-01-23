import sys
import numpy as np


def cross_entropy_f(Y, T):
    C = -1 * T * np.log(Y)
    return np.sum(C)/C.shape[1]

def cross_entropy_df(Y, T):              #gradient matrix w.r.t Y
    gradient_Y = - (T / Y) / T.shape[1]
    return gradient_Y

cross_entropy = [cross_entropy_f, cross_entropy_df]

def logit_f(matrix):
    return  (np.exp(-matrix) + 1)**(-1)

def logit_df(matrix):
    F  = logit_f(matrix)
    return  F - np.power(F,2)

logit = [logit_f, logit_df]


def relu_f(matrix):
    return  np.maximum(0,matrix)


def relu_df(matrix):
    return np.where(matrix >= 0,1,0)

relu = [relu_f, relu_df]

def softmax_f(matrix):
    maximum = np.max(matrix, axis = 0)
    diff = matrix - maximum
    num = np.exp(diff)
    output = num / np.sum(num, axis = 0)
    return output

activation_dict = {
    "logit"         : logit,
    "ReLu"          : relu
}

cost_dict = {
    "cross_entropy" : cross_entropy,
}

