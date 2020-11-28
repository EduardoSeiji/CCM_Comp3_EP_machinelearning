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

def main():

	# matrix A
	A = np.array([[3/10,3/5,0],[1/2,0,1],[4/10,4/5,0]])

	n = len(A)
	m = len(A[0])
	p = 2

	# determine H and W
	W,H = factorizate(A,n,m,p)
	
	# print W
	print('-'*10)
	print('Matrix W')	
	print('-'*10)
	print(np.matrix(W))
	
	# print H
	print('-'*10)
	print('Matrix H')	
	print('-'*10)
	print(np.matrix(H))
	print()

	# get the matrix A by multiplying W and H, just to make the checking easier
	A = np.dot(W,H)

	# print A
	print('-'*15)
	print('Matrix A = WH')
	print('-'*15)
	print(np.matrix(A))

#=========================================================

def factorizate(A,n,m,p):

	# generate a random matrix W, with all entries in [0,1[
	W = np.random.rand(n,p)

	# store a copy of A (not a reference to A!)
	Acopy = np.copy(A)

	itmax = 0
	Einitial = 1
	Efinal = 0

	while(abs(Einitial-Efinal) > 1e-5 or itmax < 100):

		itmax += 1

		# normalize W
		for j in range(p):

			s = 0

			for i in range(n):


				s += (W[i][j])**2

			for i in range(n):

				W[i][j] = W[i][j]/math.sqrt(s)

		# solve the MMQ problem W H = A, determining H
		H = QRfactorizationSimultaneous(W,A)
	
		H = H.clip(min = 1e-10)

		# compute At, transposed
		A = np.copy(Acopy)
		At = A.T
		A = np.copy(Acopy)

		# solve the MMQ problem Ht Wt = At, determining Wt
		# ver como usar a tarefa1 p isso
		Hcopy = np.copy(H)
		Ht = H.T
		Wt = QRfactorizationSimultaneous(Ht,At)
		# ah nao, cipa a tarefa 2 seja implementar o algoritmo p isso


		# compute W
		W = Wt.T

		W = W.clip(min = 1e-10)

		# calculate the error
		Einitial = Efinal
		Efinal = calculateError(Acopy,W,Hcopy)

		# conditions to exit the while
		#if(abs(Einitial-Efinal) < 1e-5): break
		#if(itmax == 100): break

	return W,Hcopy

#=========================================================

# calculate the error (absolute) value
def calculateError(Acopy, W, H):

	E = 0

	# calculate WH
	WH = np.dot(W,H)

	n = len(Acopy)
	m = len(Acopy[0])

	for i in range(n):

		for j in range(m):

			E += (Acopy[i][j] - WH[i][j])**2

	return E

#=========================================================
#=========================================================

# compute cos(teta) and sin(teta)
# more numerically stable
def computeCosSin1(w, i, j, k):

	# compute sin(teta) and cos(teta)
	if (abs(w[i][k]) > abs(w[j][k])):

		# compute tau
		T = -float(w[j][k])/w[i][k]

		# compute cos(teta)
		cos = 1/( ( 1+T**2 ) ** (1/2) )

		# compute sin(teta)
		sin = cos*T

	else:

		# compute tau
		T = -float(w[i][k])/w[j][k]

		# compute sin(teta)
		sin = 1/( ( 1+T**2 ) ** (1/2) )

		# compute cos(teta)
		cos = sin*T

	return cos, sin

#=========================================================

# compute cos(teta) and sin(teta)
# less numerically stable
def computeCosSin2(w, i, j, k):

	# compute cos(teta)
	cos = w[i][k] / (w[i][k]**2 + w[j][k]**2)**(1/2)

	# compute sin(teta)
	sin = -w[j][k] / (w[i][k]**2 + w[j][k]**2)**(1/2)

	return cos, sin

#=========================================================

# apply 1 Givens' Rotation
# use vectorization instead of a for loop (like RotGivens2), much faster
def RotGivens(w, i, j, k, cos, sin, m):

	w[i,0:m] , w[j,0:m] = cos * w[i,0:m] - sin * w[j,0:m] , sin *w[i,0:m] + cos * w[j,0:m]

#=========================================================

# apply 1 Givens' Rotation
# use a for loop, slower
def RotGivens2(w, i, j, k, cos, sin, m):

	for r in range(0,m):

		aux = cos*w[i][r] - sin*w[j][r]
		w[j][r] = sin*w[i][r] + cos*w[j][r]
		w[i][r] = aux

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
def QRfactorization(w,b):

	n = len(w)
	m = len(w[0])

	# apply sucessives Givens Rotations, until W becomes R (a triangular superior matrix)
	for k in range (0, m):

		for j in range (n-1, k, -1):

			i = j - 1

			if (w[j][k] != 0): 

				# compute cos(teta) and sin(teta) to be used in RotGivens
				cos, sin = computeCosSin1(w, i, j, k)

				# apply a Givens' Rotation to matrix W and vector b
				RotGivens(w, i, j, k, cos, sin, m)
				RotGivens(b, i, j, k, cos, sin, 1)

	# With this, W was transformed in R (a triangular superior matrix)

	# generates the vector x, which will hold the solution
	x = np.zeros((m))

	# solves the system to find each coordinate of vector x
	for k in range(m-1,-1,-1):

		summation = 0

		for j in range(k+1,m):

			summation += w[k][j]*x[j]

		x[k] = (b[k] - summation) / w[k][k]

	return x

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
def QRfactorizationSimultaneous(w,A):

	n = len(A)
	p = len(w[0])
	m = len(A[0])

	# apply sucessives Givens Rotations, until W becomes R (a triangular superior matrix)
	for k in range (0, p):

		for j in range (n-1, k, -1):

			i = j - 1

			if (w[j][k] != 0): 

				# compute cos(teta) and sin(teta) to be used in RotGivens
				cos, sin = computeCosSin1(w, i, j, k)

				# apply a Givens' Rotation to matrix W and A
				RotGivens(w, i, j, k, cos, sin, p)
				RotGivens(A, i, j, k, cos, sin, m)

	# With this, W was transformed in R (a triangular superior matrix)

	# generates the vector x, which will hold the solution
	h = np.zeros((p,m))

	# solves the system to find each entry of matrix A
	for k in range(p-1,-1,-1):

		for j in range(0,m):

			summation = 0

			for i in range(k+1,p):

				summation += w[k][i]*h[i][j]

			if(w[k][k] == 0):

				h[k][j] = (A[k][j] - summation) / 1e-10	
			
			else:			

				h[k][j] = (A[k][j] - summation) / w[k][k]

	return h

#=========================================================
main()


