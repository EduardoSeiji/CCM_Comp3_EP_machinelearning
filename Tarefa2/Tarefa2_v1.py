#---------------------------------------------------------
# Task 2
# 
# goal: factorizate A (nxm) = W (nxp) x H (pxm)
#
#
#
#
#
#---------------------------------------------------------

import math
import numpy as np

# created just for code testing
def main():

	# matrix A
	A = np.array([3/10,3/5,0],[1/2,0,1],[4/10,4/5,0])

	n = len(A)
	m = len(A[0])
	p = 2

	# determine H and W
	H,W = function(A,n,m,p)

	print('Matrix H')
	print('-'*2*m)
	print(np.matrix(H))
	print('-'*2*m)
	print()

	print('Matrix W')
	print('-'*2*p)
	print(np.matrix(W))
	print('-'*2*p)

#=========================================================

def function(A,n,m,p):

	# generate a random (all entries positive) matrix w
	W = np.empty((n,p))

	# store a copy of A (not a reference to A!)
	Acopy = np.copy(A)

	# calculate the error E
	Einitial = calculateError(Acopy, W, H)
	Efinal = 2*calculateError(Acopy, W, H)

	itmax = 0

	while 1:

		itmax += 1

		# normalize W
		for j in range(p):

			s = 0

			for k in range(n):

				s += (W[i][j])**2

			for i in range(n):

				W[i][j] = w[i][j]/s

		# solve the MMQ problem W H = A, determining H
		# ver como usar a tarefa1 p isso
		# ah nao, cipa a tarefa 2 seja implementar o algoritmo p isso


		# redefine H
		for i in range(p):

			for j in range(m):

				H[i][j] = max(0,H[i][j])

		# compute At, transposed
		At = A.T

		# solve the MMQ problem Ht Wt = At, determining Wt
		# ver como usar a tarefa1 p isso
		# ah nao, cipa a tarefa 2 seja implementar o algoritmo p isso


		# compute W
		W = Wt.T

		# redefine W
		for i in range(n):

			for j in range(p):

				W[i][j] = max(0,W[i][j])


		# conditions to exit the while
		if(abs(Einitial-Efinal) < 1e-5): break
		if(itmax = 100): break

	return W,H


#=========================================================

# calculate the error (absolute) value
def calculateError(Acopy, W, H):

	E = 0

	# calculate WH
	WH = np.dot(W,H)

	for i in range(n):

		for j in range(m):

			E += (Acopy[i][j] - WH[i][j])**2

	return E

#=========================================================

#=========================================================

#=========================================================
main()


