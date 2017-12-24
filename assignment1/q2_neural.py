import numpy as np 
import random

from q1_softmax import softmax 
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
	"""
	Forward and backward propagation for a two-layer sigmoidal network

	Compute the forward propagation and for the cross entropy cost,
	anc backward propagation for the gradients for all parameters

	Arguments:
	data -- M x Dx matrix, where each row is a training example
	labels -- M x Dy matrix, where each row is a one-hot vector
	params -- Model parameters, these are unpacked for you.
	dimensions -- A tuple of input dimension, number of hidden units
	"""
    ofs = 0
    Dx, H, Dy = (dimension[0], dimension[1], dimension[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # forward propagation
    h = sigmoid (np.dot(data, W1) + b1)
    yhat = softmax(np.dot(h, W2) + b2)

    # backward propagation
    cost = np.sum(-np.log(yhat[labels==1])) / data.shape[0]

    d3 = (yhat - labels) / data.shape[0]
    gradW2 = np.dot(h.T, d3)
    gradb2 = np.sum(d3, 0, keepdims=True)

    dh = np.dot(d3, W2.T)
    grad_h = sigmoid_grad(h) * dh

    gradW1 = np.dot(data.T, grad_h)
    gradb1 = np.sum(grad_h, 0)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad
    