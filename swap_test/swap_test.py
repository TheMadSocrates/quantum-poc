#! /usr/bin/python

import numpy as np
import scipy.stats
import scipy.linalg
from numpy import random as rnd
from numpy import linalg
from sys import stderr, exit
import argparse


"""
Main function of our program
"""
def main(args):
	normalize = lambda vect : vect / linalg.norm(vect)
	
	# get swap matrix @$S \in \mathbb{C}^{2\times 2}$@
	S = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
	
	# get a @$2 \times 2$@ Hadamard matrix @$H$@
	H = scipy.linalg.hadamard(2)/np.sqrt(2)
	
	# get @$4\times 4$@ identity matrix @$I_4$@
	I_4 = np.identity(4)
	# get @$2\times 2$@ identity matrix @$I_2$@
	I_2 = np.identity(2)
	
	
	# @$\ket{0}$@
	k_zero = np.array([[1],[0]])
	# @$\ket{1}$@
	k_one = np.array([[0],[1]])
	
	# decide on vector @$\ket{\phi}$@ based on argument
	if args.phi == 'random':
		phi = normalize(rnd.rand(4, 1) + 1.0j * rnd.rand(4, 1))
	else:
		# decide between @$\ket{00}, \ket{01}, \ket{10}, \ket{11}$@
		phi_1 = np.array([[1], [0]]) if args.phi[0] == '0' \
			else np.array([[0], [1]])
		phi_2 = np.array([[1], [0]]) if args.phi[1] == '0' \
			else np.array([[0], [1]])
		phi = np.kron(phi_1, phi_2)
	
	print(f"Using swap matrix \n{S}", file=stderr)
	print(f"Using vector\n{phi}", file=stderr)
	
	# compute @$\ket{0} \otimes \ket{\phi}$@
	input_vec = np.kron(k_zero, phi)
	
	# compute @$O_1 = H \otimes I_4$@
	O1 = np.kron(H, I_4)
	
	"""
	generate matrix @$O_2 = \ket{0}\bra{0}\otimes I_4 + \ket{1}\bra{1}
	\otimes S =\left(\begin{array}{c|c} I_4 & 0\\ \hline 0 & S \end{array}\right)$@
	"""
	O2 = np.kron(k_zero.dot(k_zero.T), I_4) + \
			np.kron(k_one.dot(k_one.T), S)
	
	
	# operation applied is @$O_1 \times O_2 \times O_1$@
	oper = O1.dot(O2.dot(O1))
	
	# compute operation @$\ket{\Psi} = O_1 O_2 O_1 (\ket{0} \otimes \ket{\phi})$@
	out = oper.dot(input_vec)
	
	# compute @$P = \ket{0} \bra{0} \otimes I_4$@
	prob_zero_base = np.kron(k_zero.dot(k_zero.T), I_4)
	
	# compute actual probability @$||P \ket{\Psi}||^2$@
	pr0 = linalg.norm(prob_zero_base.dot(out))**2
	
	# and print the result
	print(f"Pr(0) {pr0}")

	return 0	

if __name__ == "__main__":
	# set arguments for program
	parser = argparse.ArgumentParser(description='Swap test for a 2-qbit input')
	parser.add_argument("--phi", dest='phi', default='01', \
		choices=['00', '01', '10', '11', 'random'], \
		help='input vector selection (default=01)')
	
	# parse them
	args = parser.parse_args()

	# and call main()
	exit(main(args))

