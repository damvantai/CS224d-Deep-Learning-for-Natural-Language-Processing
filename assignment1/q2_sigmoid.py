import numpy as np 

def sigmoid(x):
	s = 1.0 / (1 + np.exp(-x))

	return s

def sigmoid_grad(s):
	ds = s * (1 - s)
	return ds