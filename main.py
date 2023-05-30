
import numpy as np
import tensorflow as tf

from tensorflow import keras
#from tensorflow.keras import layers
##########################################################################
def softmax(x):
    exp_x = np.exp(x)
    sm_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return sm_x

def softmax_norm(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def nl(input, target):
    return -np.mean(np.log(input[range(target.shape[0]), target]))

##########################################################################
def nll(input, target):
    return - np.mean(input[range(target.shape[0]), target])

def log_softmax(x):
    c = x.max()
    logsumexp = np.log(np.sum(np.exp(x - c), axis=-1, keepdims=True))
    return x - c - logsumexp
################################################################################

x = np.array([[ 0.9826,  1.0630, -0.4096],
        [-0.6213,  0.2511,  0.5659],
        [ 0.5662,  0.7360, -0.6783],
        [-0.4638, -1.4961, -1.0877],
        [ 1.8186, -0.2998,  0.1128]])

target = np.array([1, 0, 1, 1, 1])

################################################################################
# SoftMax + NL = CrossEntropy + Loss
pred = softmax(x)
loss = nl(pred, target)
# loss = (1.4904)
################################################################################
# LogSoftMax + NLL = CrossEntropy + Loss
pred2 = log_softmax(x)
loss2 = nll(pred2, target)
# loss = (1.4904)
################################################################################

# tfr.keras.losses.SoftmaxLoss
# Computes Softmax cross-entropy loss between y_TRUE and y_PRED.

pred_k = log_softmax(np.array([0.6, 0.8]))
pred_k = np.multiply([1., 0.], pred_k)
loss = -np.sum(pred_k)
# loss = 0.798138
###############################################################################
#pred_k = log_softmax(np.array([[0.0, 0.6, 0.8], [0.5, 0.8, 0.4]]))
#pred_k = np.multiply(np.array([[0.0, 1., 0.], [0., 1., 0.]]), pred_k)
#loss = -np.sum(pred_k, axis=-1)


def softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy from logits[batch, n_classes] and ids of correct answers"""

    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy
tf.nn.softmax_cross_entropy_with_logits

def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy gradient from logits[batch, n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]

