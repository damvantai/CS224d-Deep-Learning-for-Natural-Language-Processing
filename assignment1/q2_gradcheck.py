import numpy as np 
import random

def gradcheck_naive(f, x):
	""" Gradient check for a function f

	Arguments:
	f -- a function the takes a single argument and outputs the cost and its gradients

	x -- the point (numpy array) to check the gradient at
	"""

	rndstate = random.getstate()
	random.setstate(rndstate)
	fx, grad = f(x)
	h = 1e-4

	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

	while not it.finished:
		ix = it.multi_index

		x[ix] += h

		random.setstate(rndstate)
		new_f1 = f(x)[0]

		x[ix] -= 2*h

		random.setstate(rndstate)
		new_f2 = f(x)[0]

		x[ix] += h

		numgrad = (new_f1 - new_f2) / (2 * h)

		reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))

		if reldiff > 1e-5:
			print("Gradient check failed.")
			print("First gradient error found at index %s" % str(ix))
			print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))

			return 

		it.iternext()
	print("Gradient check passed!")
	