import sys
import numpy as np
#### my modules
from core.functions import softmax_f

def forward_step(W, A_before, activation_f, activation_df):
    Z     = W @ A_before
    Asub  = activation_f(Z)
    DF    = activation_df(Z)
    return Z, Asub, DF

# for softmax_f only
def forward_step_last(W_last, A_before):
    Z_last     = W_last @ A_before
    A_last_sub  = softmax_f(Z_last)
    return Z_last, A_last_sub

# for softmax_f only
def backward_last_layer(T, A_last, A_before, cost_df):
    Y = A_last[1:,:]
    gradient_Y = cost_df(Y, T)
    Delta = Y*gradient_Y - Y*np.sum((Y*gradient_Y), axis=0)
    Gradient_W = Delta @ A_before.T
    return Delta, Gradient_W

def backward_step(DF, W_after, Delta_after, A_before):
    W_after_transpose = W_after.T
    Delta      = DF * ( W_after_transpose @ Delta_after )
    Gradient_W =  Delta @ A_before.T
    return Delta, Gradient_W











