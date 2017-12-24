import numpy as np 

def softmax(x):
	"""
	Compute the softmax function for each row of the input x

	It is crucial that this function is optimized for speed because
	it will be used frequently in later code

	"""

	orig_shape = x.shape

	if len(x.shape) > 1:
		# Matrix
		exp_minmax = lambda x: np.exp(x - np.max(x))
		denom = lambda x: 1.0 / np.sum(x)
		x = np.apply_along_axis(exp_minmax, 1, x)
		denominator = np.apply_along_axis(denom, 1, x)

		if len(denominator.shape) == 1:
			denominator = denominator.reshape((denominator.shape[0], 1))

		x = x * denominator
	else:
		x_max = np.max(x)
		x = x - x_max
		numerator = np.exp(x)
		denominator = 1.0 / np.sum(numerator)
		x = numerator.dot(denominator)