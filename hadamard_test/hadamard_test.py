#! /usr/bin/python

import numpy as np
import scipy.stats
import scipy.linalg
from numpy import random as rnd
from numpy import linalg
from sys import stderr, exit
import argparse


class size_action(argparse.Action):
	"""
	Determine if the input argument for dimension of the unitary matrix is
	valid. 
	"""
	def __call__(self, parser, namespace, values, option_string=None):
		if values < 2:
			parser.error("Matrix must be at least 2x2")

		setattr(namespace, self.dest, values)

"""
Main function of our program
"""
def main(args):
	# get size of matrix @$n$@
	n = args.n
	
	normalize = lambda vect : vect / linalg.norm(vect)
	
	# get unitary matrix @$U \in \mathbb{C}^{n\times n}$@
	U = scipy.stats.unitary_group.rvs(n)
	
	# get a @$2 \times 2$@ Hadamard matrix @$H$@
	H = scipy.linalg.hadamard(2)/np.sqrt(2)
	
	# get @$n\times n$@ identity matrix @$I_n$@
	I_n = np.identity(n)
	# get @$2\times 2$@ identity matrix @$I_2$@
	I_2 = np.identity(2)
	
	
	# @$\ket{0}$@
	k_zero = np.array([[1],[0]])
	# @$\ket{1}$@
	k_one = np.array([[0],[1]])
	
	# decide on vector @$\ket{\phi}$@
	# if '--phi random' was given get a random vector for @$\ket{\phi}$@
	phi = normalize(rnd.rand(n, 1) + 1.0j * rnd.rand(n, 1)) if args.phi == 'random' \
			else np.zeros((n, 1))
	if args.phi == 'zero':
		# if '--phi zero' was given, then @$\ket{\phi}=\ket{0}^{\otimes n}$@
		phi[0][0] = 1
	elif args.phi == 'one':
		# if '--phi zero' was given, then @$\ket{\phi}=\ket{1}^{\otimes n}$@
		phi[n-1][0] = 1
		
	
	print(f"Using matrix\n{U}", file=stderr)
	print(f"Using vector\n{phi}", file=stderr)
	
	
	# compute @$\ket{0} \otimes \ket{\phi}$@
	input_vec = np.kron(k_zero, phi)
	
	# compute @$O_1 = H \otimes I_n$@
	O1 = np.kron(H, I_n)
	
	"""
	generate matrix @$O_2 = \ket{0}\bra{0}\otimes I_n + \ket{1}\bra{1}
	\otimes U =\left(\begin{array}{c|c} I_n & 0\\ \hline 0 & U \end{array}\right)$@
	"""
	O2 = np.kron(k_zero.dot(k_zero.T), I_n) + \
			np.kron(k_one.dot(k_one.T), U)
	
	
	# operation applied is @$O_1 \times O_2 \times O_1$@
	oper = O1.dot(O2.dot(O1))
	
	# compute operation @$\ket{\Psi} = O_1 O_2 O_1 (\ket{0} \otimes \ket{\phi})$@
	out = oper.dot(input_vec)
	
	# compute @$P = \ket{0} \bra{0} \otimes I_n$@
	prob_zero_base = np.kron(k_zero.dot(k_zero.T), I_n)
	
	# compute actual probability @$||P \ket{\Psi}||^2$@
	pr0 = linalg.norm(prob_zero_base.dot(out))**2
	
	# and print the result
	print(f"Pr(0) {pr0}")

	return 0	

if __name__ == "__main__":
	# set arguments for program
	parser = argparse.ArgumentParser(description='Hadamard test')
	parser.add_argument("-n", dest='n', type=int, default=2, \
		action=size_action, \
		help='size of the unitary matrix (default=2)')
	parser.add_argument("--phi", dest='phi', default='random', \
		choices=['random', 'one', 'zero'], \
		help='input vector selection (default=random)')
	
	# parse them
	args = parser.parse_args()

	# and call main()
	exit(main(args))

