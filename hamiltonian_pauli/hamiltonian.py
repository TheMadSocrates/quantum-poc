#! /usr/bin/python

import numpy as np
import argparse
from sys import exit


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
Generate the matrix @$\sigma_z^{(k)}$@
'''
def get_sigma_z_matrix(k, n = 5):
	return get_matrix(np.array([[1, 0], [0, -1]]), k, n)

'''
Generate a pgfplots graph with the eigenvalues.
'''
def output_plot(points, ymin, ymax):
	print("% requires \\usepackage{tikz,pgfplots} in preamble")
	print("\\begin{tikzpicture}\\begin{axis}")
	print(f"[ymin={ymin},ymax={ymax},xmin=0,xmax=1,"
			"title={$H(\\tau) = (1 - \\tau)H_1 + \\tau H_2$},"
			"xlabel={$\\tau$}, ylabel={Eigenvalues},"
			f"ytick distance={0.1*(ymax-ymin)},"
			"width=0.9\\columnwidth, height=2.5in,]")

	for l in points:
		print("\\addplot+[mark=none] plot coordinates\n{ ", end='')
		for x, val in l[:-1]:
			print(f"({x:.6f}, {val:.6f}) ", end='')
		print(f"({l[-1][0]}, {l[-1][1]:.6f})" "};")
	print("\\end{axis}\\end{tikzpicture}")


def main(args):
	'''
	Generate @$H_1 = \sum_{k=1}^n \sigma_x^{(k)}$@
	'''
	H_1 = get_sigma_x_matrix(1)
	for i in range(2, 6):
		H_1 += get_sigma_x_matrix(i)

	'''
	Generate @$H_2 = \sum_{k=1}^{n-1} \sigma_z^{(k)}\sigma_z^{(k+1)}$@
	'''
	H_2 = get_sigma_z_matrix(1).dot(get_sigma_z_matrix(2))
	for i in range(2, 5):
		H_2 += get_sigma_z_matrix(i).dot(get_sigma_z_matrix(i+1))

	points = [[], []]
	ymin = None
	ymax = None
	# step through values of @$\tau$@
	for tau in np.arange(0, 1 + args.step, args.step):
		# generate @$H(\tau) = (1 - \tau)H_1 + \tau H_2$@
		H = (1 - tau) * H_1 + tau * H_2
		# because of the way floating point works, numpy gives some ``complex
		# numbers'' as a result with imaginary part 0, take the real part only
		l, s = np.real(np.sort(np.linalg.eigvals(H))[0:2])
		# add points to the list
		points[0].append((tau, l))
		points[1].append((tau, s))
		# and compute maximum and minimum coordinates for the y-axis
		ymin = np.min([l, s]) if ymin == None else np.min([ymin, l, s])
		ymax = np.max([l, s]) if ymax == None else np.max([ymax, l, s])

	# give some margin to y
	delta = 0.1 * (ymax - ymin)
	# generate the plot
	output_plot(points, ymin - delta, ymax + delta)

	# and return
	return 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--step", type=float, default=0.01)

	args = parser.parse_args()
	exit(main(args))
