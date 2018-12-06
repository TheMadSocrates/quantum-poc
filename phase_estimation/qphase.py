#! /usr/bin/python

import numpy as np
import scipy
from sys import exit
import argparse


class validate_phi(argparse.Action):
	"""
	Determine if the value given for phi is actually a valid floating point
	number or a fraction, and if it is within the acceptable range.
	"""
	def __call__(self, parser, namespace, values, option_string=None):
		frac = 0
		if self.is_valid(values, '.'):
			frac = float(values)

		elif self.is_valid(values, '/'):
			v = values.split('/')
			frac = int(v[0]) / int(v[1])
		else:
			parser.error("invalid parameter: "
			"expected floating point or fraction")
		
		if frac >= 1 or frac <= 0:
			parser.error("value is out of range, expected (0, 1)")

		setattr(namespace, self.dest, frac)
	
	def is_valid(self, val, sep='.'):
		v = val.split(sep)
		return len(v) == 2 and all(i.isdigit() for i in v)

class basis():
	# @$\ket{0}$@
	k_zero = np.array([[1], [0]])
	# @$\ket{1}$@
	k_one = np.array([[0], [1]])

def get_iqft(N=2):
	"""
	generate the inverse quantum fourier transform matrix of size @$N \times
	N$@ as defined by
	@$F^\dagger_N = \frac{1}{\sqrt{N}} \sum_{k,l\in [N]} \omega^{k\cdot l}
	\ket{k} \bra{l}$@ with @$\omega = e^{2\pi i/N}$@
	"""
	F = np.zeros((N, N), dtype=np.complex)
	for k in range(N):
		for l in range(N):
			F[k][l] += np.exp(-2*l * k * np.pi * 1.0j/N)
	
	return F / np.sqrt(N)

def get_phi_term(k, phi):
	"""
	compute @$\frac{1}{\sqrt{2}}(\ket{0} + \exp(2\pi i 2^k \varphi)\ket{1})$@
	"""
	return (basis.k_zero + np.exp(2.0j * np.pi * (2**k) * phi) * basis.k_one)/np.sqrt(2)


def main(args):
	n = args.n
	phi = args.phi

	N = 2**n

	"""
	construct @$\ket{\varphi} = \bigotimes_{k=n-1,\cdots,0} \frac{1}{\sqrt{2}}
	\ket{\varphi_k}$@
	"""
	vec_phi = get_phi_term(n-1, phi)
	for p in range(n-2, -1, -1):
		vec_phi = np.kron(vec_phi, get_phi_term(p, phi))
	
	# obtain a properly sized inverse quantum fourier transform matrix
	IFT = get_iqft(N)
	# compute the output
	out = IFT.dot(vec_phi)
	
	# and compute the probability distribution on the output for each possible
	# @$\ket{x}$@, thus returning the binary fraction

	prob_total = np.zeros((N, 1), dtype=np.float)
	for x in range(N - 1, -1, -1):
		frac = np.zeros((N, 1))
		frac[x][0] = 1
		prob_total[x][0] = np.linalg.norm(frac.T.dot(out))**2

	if args.latex:
		# output for pgfplots comb plot
		print("% requires \\usepackage{tikz,pgfplots} in preamble")
		print("\\begin{tikzpicture}\\begin{axis}")
		print(f"[ymin=0,ymax=1.1,xmin=0,xmax={N},"
				f"title={{$N=2^{{{n}}},\\varphi={phi}$}},"
				"xlabel={$x$}, ylabel={$Pr(x)$},"
				"ytick distance=0.2,"
				"width=\\columnwidth, height=2.5in,]")
		print("\\addplot+[ycomb,mark options={black}] plot coordinates\n{(0, 0) ")
		for x, val in enumerate(prob_total[:-1]):
			print(f"({x+1}, {val[0]:.6f}) ", end='')
		print(f"({N}, {prob_total[-1][0]:.6f})" "};")
		print("\\end{axis}\\end{tikzpicture}")
	
	if args.matplotlib:
		# generate a matplotlib plot
		import matplotlib.pyplot as plt
		x = [0]
		y = [0]
		for cx, cy in enumerate(prob_total):
			x.append(cx+1)
			y.append(cy[0])
		plt.stem(x, y)
		plt.title(fr"$N=2^{n}, \varphi={phi}$")
		plt.xlabel(r"$x$")
		plt.ylabel(r"$Pr(x)$")
		plt.ylim(0, 1.1)
		plt.xlim(0, N)
		plt.show()
	
	if not (args.matplotlib or args.latex):
		# raw output
		print(f"N=2^{n}, phi={phi}")
		print(prob_total)

	return 0

if __name__ == "__main__":
	# set arguments for program
	parser = argparse.ArgumentParser( \
		description='Quantum phase estimation test')
	parser.add_argument("-n", dest='n', type=int, default=2, \
		help='use an n-qubit vector')
	parser.add_argument("--phi", dest='phi', default=0.5, \
		action=validate_phi,
		help='input fraction to use')
	parser.add_argument("--latex", dest='latex', action="store_true", \
		help='give latex output for run'),
	parser.add_argument('--matplotlib', dest='matplotlib', \
		action="store_true", help='give matplotlib output')
	
	# parse them
	args = parser.parse_args()

	exit(main(args))
