#---------------------------------------------------------
# Task 1
# 
# Given W (nxm) and b (n), make an algorithm such that:
#
# 	transform (W -> R) and (b -> b~), by sucessives Givens' Rotations
#
#	solve the system Rx = b~
#---------------------------------------------------------

# import the libraries math and numpy
import math
import numpy as np

def main():

	test = input("Digite o item que deseja testar (a, b, c, d):")
	print()

	#---------------------------------------------------

	# test a
	if(test == 'a'):

		n = 64
		m = 64

		# create the w matrix
		# create a null matrix
		w = np.zeros((n,m), dtype = float)
		
		# fill it properly
		for i in range(n):
		
			for j in range(m):
		
				if(i == j): w[i][j] = 2

				if(abs(i-j) == 1): w[i][j] = 1

				if(abs(i-j) > 1): w[i][j] = 0

		# create b vector
		b = np.ones((n,1))

		# runs the QR factorization
		x = QRfactorization(w,b)

		# print x
		print('-'*20)
		print('O vetor resposta é:')
		print('-'*20)
		print(np.matrix(x))

	#---------------------------------------------------

	# test b
	if(test == 'b'):

		n = 20
		m = 17

		# create the w matrix
		# create a null matrix
		w = np.zeros((n,m))
		
		# fill it properly
		for i in range(n):
		
			for j in range(m):

				if(abs((i+1)-(j+1)) <= 4): w[i][j] = 1/(i+j+1)

				elif(abs((i+1)-(j+1)) > 4): w[i][j] = 0

		# create b vector
		b = np.zeros((n,1))

		# fill it properly
		for i in range(n):

			b[i] = i

		# runs the QR factorization
		x = QRfactorization(w,b)

		# print x
		print('-'*20)
		print('O vetor resposta é:')
		print('-'*20)
		print(np.matrix(x))

	#---------------------------------------------------
	
	# test c
	if(test == 'c'):

		n = 64
		p = 64
		m = 3

		# create the w matrix
		# create a null matrix
		w = np.zeros((n,p))
		
		# fill it properlyD	
		for i in range(n):
		
			for j in range(p):

				if(i == j): w[i][j] = 2

				elif(abs(i-j) == 1): w[i][j] = 1

				elif(abs(i-j) > 1): w[i][j] = 0

		# create matrix A
		A = np.ones((n,3))

		# fill it properly
		for i in range(n):

			A[i][0] = 1

			A[i][1] = i+1

			A[i][2] = 2*(i+1) - 1

		# runs QR factorization in in multiple systems simultaneously
		h = QRfactorizationSimultaneous(w,A)

		# print h
		print('-'*20)
		print('A matriz resposta é:')
		print('-'*20)
		print(np.matrix(h))

	#---------------------------------------------------

	# test d
	if(test == 'd'):

		n = 20
		p = 17
		m = 3

		# create the w matrix
		# create a null matrix
		w = np.zeros((n,p))
		
		# fill it properly
		for i in range(n):
		
			for j in range(p):

				if(abs(i-j) <= 4): w[i][j] = 1/(i+j+1)

				elif(abs(i-j) > 4): w[i][j] = 0

		# create matrix A
		A = np.ones((n,3))

		# fill it properly
		for i in range(n):

			A[i][0] = 1

			A[i][1] = i+1

			A[i][2] = 2*(i+1) - 1

		# runs QR factorization in in multiple systems simultaneously
		h = QRfactorizationSimultaneous(w,A)

		# print h
		print('-'*20)
		print('A matriz resposta é:')
		print('-'*20)
		print(np.matrix(h))

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