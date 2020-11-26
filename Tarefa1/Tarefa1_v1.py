#---------------------------------------------------------
# Task 1
# 
# Given W (nxm) and b (n), make an algorithm such that:
#
# 	transform (W -> R) and (b -> bTilde), by sucessives Givens' Rotations
#
#	solve the system Rx = bTilde
#
#
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

	teste = input("Digite o item que deseja testar: ")
	print()

	#---------------------------------------------------

	# test a
	if(teste == 'a'):

		n = 64
		m = 64

		# create the w matrix
		# create a null matrix
		w = np.zeros((n,m), dtype = float)
		
		# fill it properly
		for i in range(n):
		
			for j in range(m):
		
				if(i == j): 
					w[i][j] = 2

				if(abs(i-j) == 1): 
					w[i][j] = 1

				if(abs(i-j) > 1): 
					w[i][j] = 0
			print()

		# print matrix just for checking
		#printMatrix(w)

		# create b vector
		b = np.ones((n,1))

		# print vector just for checking
		#printVector(w)

		QRFatoration(w,b)

	#---------------------------------------------------

	# test b
	if(teste == 'b'):

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

		# print matrix just for checking
		#printMatrix(w)

		# create b vector
		b = np.zeros((n,1))

		# fill it properly
		for i in range(n):

			b[i] = i

		# print vector just for checking
		#printVector(b)

		QRFatoration(w,b)

	#---------------------------------------------------
	
	# test c
	if(teste == 'c'):

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

		# print matrix just for checking
		#printMatrix(w)

		# create matrix A
		A = np.ones((n,3))

		# fill it properly

		for i in range(n):

			A[i][0] = 1

			A[i][1] = i+1

			A[i][2] = 2*(i+1) - 1




		QRFatorationSimultaneous(w,A)

	#---------------------------------------------------

	# test d
	if(teste == 'd'):

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

		# print matrix just for checking
		#printMatrix(w)

		# create matrix A
		A = np.ones((n,3))

		# fill it properly

		for i in range(n):

			A[i][0] = 1

			A[i][1] = i+1

			A[i][2] = 2*(i+1) - 1


		QRFatorationSimultaneous(w,A)

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
# the 3rd given algorithm
def RotGivens(w, i, j, k, cos, sin, m): # m = len(w[0])

	# FAZER VETORIZAÇÃO (MIKI FEZ ASSIM TB)
	# SAULO DISSE Q ACELERA FORTE
	# SUBSTITUI O FOR: 
	# FAZER MAIS ROTGIVENS TESTANDO (ESSE DE CIMA É O MAIS RAPIDO)
	'''
	for r in range(0,m = len(w[0])): # OU PASSAR M COMO ENTRADA TB

		aux = cos*w[i][r] - sin*w[j][r]

		w[j][r] = sin*w[i][r] + cos*w[j][r]

		w[i][r] = aux

	'''
	w[i,0:m] , w[j,0:m] = cos * w[i,0:m] - sin * w[j,0:m] , sin *w[i,0:m] + cos * w[j,0:m]


#=========================================================

# apply sucessives Givens' Rotations in a convenient order
# the 1st given algorithm
def QRFatoration(w,b):

	# TB DA P OBTER M ASSIM: m = w.shape[1]
	n = len(w)
	m = len(w[0])

	for k in range (0, m):

		for j in range (n-1, k, -1):

			i = j - 1

			if (w[j][k] != 0): 

				# apply a Givens' Rotation to matrix W

				cos, sin = computeCosSin1(w, i, j, k)

				RotGivens(w, i, j, k, cos, sin, m) # passar cos e sin como argumento
				RotGivens(b, i, j, k, cos, sin, 1)

	# AGORA W JÁ É R
	# LETSS

	# CRIAR VETOR X
	x = np.zeros((m))

	for k in range(m-1,-1,-1):

		summation = 0

		for j in range(k+1,m):

			summation += w[k][j]*x[j]

		x[k] = (b[k] - summation) / w[k][k]

	#printVector(x)
	print(np.matrix(x))

	# AE CARAI TEMO O X!!!

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
# the 1st given algorithm
def QRFatorationSimultaneous(w,A):

	# TB DA P OBTER M ASSIM: m = w.shape[1]
	n = len(w)
	p = len(w[0])
	m = len(A[0])

	for k in range (0, p):

		for j in range (n-1, k, -1):

			i = j - 1

			if (w[j][k] != 0): 

				# apply a Givens' Rotation to matrix W

				cos, sin = computeCosSin1(w, i, j, k)

				RotGivens(w, i, j, k, cos, sin, p) # passar cos e sin como argumento
				RotGivens(A, i, j, k, cos, sin, m)

	# AGORA W JÁ É R
	# LETSS

	# CRIAR VETOR X
	h = np.zeros((p,m))

	for k in range(p-1,-1,-1):

		for j in range(0,m):

			summation = 0

			for i in range(k+1,p):

				summation += w[k][i]*h[i][j]

			if(w[k][k] == 0):

				h[k][j] = (A[k][j] - summation) / 1e-10	
			
			else:			

				h[k][j] = (A[k][j] - summation) / w[k][k]

	#printVector(x)
	print(np.matrix(h))

	# AE CARAI TEMO O X!!!

#=========================================================
'''
# solve the system of equations
# the 2nd given algorithm
def solveSystem(w, b, x):

	# TIRAR ESSA POHA

	for k in range(1,m):

		for j in range(n, k,-1):

			i = j-1

			if(w[j][k] != 0):

				# apply a Givens' Rotation to matrix W
				RotGivens1(w, i, j, k)
	# TIRAR ^

	# solve the resulting triangular system
	# since w is always overwritted, it'll have the R values at the end
	for k in range(m-1,-1,-1):

		summation = 0

		for j in range(k+1,m):

			summation += w[k][j]*x[j]

		x[k] = (b[k] - summation) / w[k][k]

	# APAGAR ESSA FUNÇÃO TODA
'''
#=========================================================

# print the given matrix
# just for testing
def printMatrix(matrix):

	print()
	print('MATRIX:')
	print('-'*2*len(matrix))

	for i in range(len(matrix)):

		for j in range(len(matrix[i])):

			print(matrix[i][j], end = ' ')

		print()

	print('-'*2*len(matrix))

#=========================================================

# print the given vector
# just for testing
def printVector(vector):

	for i in range(0,len(vector)):

		print(vector[i], end = ' ')

	print()

#=========================================================
main()