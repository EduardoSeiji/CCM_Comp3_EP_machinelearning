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
	A = np.array([[3/10,3/5,0],[1/2,0,1],[4/10,4/5,0]])

	n = len(A)
	m = len(A[0])
	p = 2

	# determine H and W
	W,H = function(A,n,m,p)

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
	#W = np.empty((n,p))
	W = np.random.rand(n,p) # [0,1[


	# store a copy of A (not a reference to A!)
	Acopy = np.copy(A)

	# calculate the error E

	itmax = 0

	Einitial = 0
	Efinal = 0

	while 1:

		itmax += 1

		# normalize W
		for j in range(p):

			s = 0

			for i in range(n):

				s += (W[i][j])**2

			for i in range(n):

				W[i][j] = W[i][j]/math.sqrt(s)

		# solve the MMQ problem W H = A, determining H
		# ver como usar a tarefa1 p isso
		H = QRFatorationSimultaneous(W,A)
		# ah nao, cipa a tarefa 2 seja implementar o algoritmo p isso


		# redefine H
		'''
		for i in range(p):

			for j in range(m):

				H[i][j] = max(0,H[i][j])
		'''
		H = H.clip(min = 1e-10)

		# compute At, transposed
		A = np.copy(Acopy)
		At = A.T
		A = np.copy(Acopy)

		# solve the MMQ problem Ht Wt = At, determining Wt
		# ver como usar a tarefa1 p isso
		Hcopy = np.copy(H)
		Ht = H.T
		Wt = QRFatorationSimultaneous(Ht,At)
		# ah nao, cipa a tarefa 2 seja implementar o algoritmo p isso


		# compute W
		W = Wt.T

		# redefine W
		'''
		for i in range(n):

			for j in range(p):

				W[i][j] = max(0,W[i][j])
		'''
		W = W.clip(min = 1e-10)

		# calcula o erro
		Einitial = Efinal
		Efinal = calculateError(Acopy,W,Hcopy)



		# conditions to exit the while
		if(abs(Einitial-Efinal) < 1e-5): break
		if(itmax == 100): break

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
	n = len(A)
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

	return h

	#printVector(x)
	#print(np.matrix(h))

	# AE CARAI TEMO O X!!!

#=========================================================

#=========================================================
main()


