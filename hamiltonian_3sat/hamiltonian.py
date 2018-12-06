#! /usr/bin/python

import numpy as np
import argparse
from sys import exit, stderr


'''
Because generating @$\sigma_x^{k}$@ and @$\sigma_z^{k}$@ follows a similar
procedure, we use this function as prototype.
'''
def get_matrix(m, k, n = 5):
	# grab a @$2\times 2$@ identity matrix
	I = np.identity(2)
	'''
	Initialize the return value to either the identity matrix or the other
	matrix. This way, if k is 1, then we have
	@$m\otimes I \otimes \cdots $@
	Otherwise, we obtain
	@$I\otimes \cdots \otimes m \otimes \cdots $@
	'''
	ret = m if k == 1 else I
	for i in range(2, n+1):
		ret = np.kron(ret, m if k == i else I)

	return ret

'''
Generate the matrix @$\sigma_x^{(k)}$@
'''
def get_sigma_x_matrix(k, n = 5):
	return get_matrix(np.array([[0, 1], [1, 0]]), k, n)

'''
Generate the Hamiltonian matrix which increases the energy state for every
vector except @$\ket{vect}$@
'''
def get_solution_matrix(vect, n = 5):
	ret = np.identity(2**n)
	ret[vect][vect] = 0
	return ret

'''
Generate a pgfplots graph with values on list points.
'''
def output_plot(points, ymin, ymax, title):
	# plot preamble data
	print("% requires \\usepackage{tikz,pgfplots} in preamble")
	print("\\begin{tikzpicture}\\begin{axis}")
	print(f"[ymin={ymin},ymax={ymax},xmin=0,xmax=1,"
			f"title={{Using unique solution ${title}$}},"
			"xlabel={$\\tau$}, ylabel={Overlap},"
			f"ytick distance={0.2*(ymax-ymin)},"
			"width=0.9\\columnwidth, height=2.in,"
			"legend entries={$\\braket{\\varphi(0)}{\\varphi(\\tau)}$,"
			"$\\braket{\\varphi(1)}{\\varphi(\\tau)}$},"
			f"legend style={{at={{(axis cs:0.05, {0.5*(ymax + ymin)})}},"
			"anchor=west,legend columns=1},]")

	# we assume multiple lists of points, we gather each one independently
	for l in points:
		# and create a point for each one of them
		print("\\addplot+[mark=none] plot coordinates\n{ ", end='')
		for x, val in l:
			print(f"({x:.6f}, {val:.6f}) ", end='')
		print("};")

	# close environments
	print("\\end{axis}\\end{tikzpicture}")


def get_eigenstates(matrix, states):
	# get the eigenvalues and eigenvectos
	eigvals, eigvects = np.linalg.eig(matrix)
	# get the index of the @$states$@ smallest eigenvalues
	idx = np.argpartition(eigvals, states)
	# and get the corresponding eigenvectors
	return eigvects[:,idx[:states][0]]


def main(args):
	'''
	Generate @$H_1 = \sum_{k=1}^n \sigma_x^{(k)}$@
	'''
	H_1 = get_sigma_x_matrix(1)
	for i in range(2, 6):
		H_1 += get_sigma_x_matrix(i)

	'''
	First obtain which vector is the solution
	'''
	vect = (int(args.entry) if args.entry != 'random' \
		else np.random.randint(0, 32)) 
	
	'''
	Generate @$H_2$@
	'''
	H_2 = get_solution_matrix(vect)

	'''
	Generate Hermitian matrix as a function of @$\tau$@
	'''
	H = lambda tau: (1 - tau) * H_1 + tau * H_2


	# obtain @$\bra{\varphi(0)}$@ and @$\bra{\varphi(1)}$@, though technically
	# we are not transposing the vectors
	phi = [get_eigenstates(H(0).conj(), 1), \
			get_eigenstates(H(1).conj(), 1)]

	# create a list to store the results
	points = [[], []]

	# step through values of @$\tau$@
	for tau in np.arange(0, 1 + args.step, args.step):
		# compute @$\bra{\varphi(\tau)}$@
		eigenstate = get_eigenstates(H(tau), 1)
		# compute @$\braket{\varphi(0)}{\varphi(\tau)}$@
		l = np.absolute(np.inner(phi[0], eigenstate))
		# compute @$\braket{\varphi(1)}{\varphi(\tau)}$@
		s = np.absolute(np.inner(phi[1], eigenstate))

		# add points to the list
		points[0].append((tau, l))
		points[1].append((tau, s))

	# generate the plot
	output_plot(points, 0, 1.25, f"\\ket{{{vect:05b}}}")

	# and return
	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--step", type=float, default=0.01)
	parser.add_argument("--entry", choices=[ \
			str(x) for x in range(32)] + ['random'], default='random')

	args = parser.parse_args()
	exit(main(args))
