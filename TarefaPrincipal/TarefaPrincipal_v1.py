#---------------------------------------------------------
# Main task
# 
# Goal: teach a machine how to read manuscripted digits
#
# The non-negative matrix factoration will be used
#---------------------------------------------------------

import math
import numpy as np
import time

def main():

	n = 784

	print('-'*18)
	print('Tarefa Principal')
	print('-'*18)

	Wdtotal = []

	a = input('Deseja treinar(a) ou testar(b)? ')

	# go to the training
	if(a == 'a'):

		# number of columms of W and lines of H (degrees of freedom)
		p = int(input('> Digite o valor de p (5, 10, 15): '))

		# number of images used in the training of each digit
		ndig_treino = int(input('> Digite o valor de ndig_treino (100, 1k, 4k): '))

		#executes the training
		Wdtotal = training(n,p,ndig_treino)

	# import data to go to straight the testing
	if(a == 'b'):

		print('Carregando classificadores...')

		for i in range(10):

			Wdtotal.append(np.genfromtxt("treinado/treinado_" + str(i) + ".txt"))

	# number of images used in testing
	n_test = int(input('> Digite o valor de n_test (1 a 10k): '))

	# executes the testing
	DigitandError = testing(n_test, Wdtotal)

	# check the result
	checking(DigitandError)

#=========================================================

# use the train_digX.txt archives (x in [0,9])
# each txt is 784x5300 (each columm is a digit image)
def training(n,p,ndig_treino):

	print('-'*13)
	print('Treinando:')
	print('-'*13)

	# matrix which will hold all the Wds (classificators)
	Wdtotal = []

	startTotal = time.time()

	# train the classificators generating a Wd matrix for each digit
	for i in range(10):

		print('Treinando o dígito ', i, '...', end = '  ')

		# calculate the matrix Wd (n x p), by factorizating
		# A = Wd.H
		A = np.genfromtxt("dados/dados_mnist/train_dig" + str(i) + ".txt")[0:784,0:ndig_treino]/255.0

		start = time.time()

		Wd,H = factorizate(A,n,p)

		# append Wd to Wdtotal
		Wdtotal.append(Wd)

		end = time.time()

		print(int(end - start), 's')

	endTotal = time.time()
	print('Treinamento completo! (', int(endTotal - startTotal), 's)')
 
	return Wdtotal

#=========================================================

# use the test_images.txt archive, 784xn_test (each columm is a testing image)
def testing(n_test, Wdtotal):

	print('-'*13)
	print('Testando:')
	print('-'*13)

	startTotal = time.time()

	# generate a matrix A (784 x n_test)
	# each columm is an image to be tested
	A = np.genfromtxt("dados/dados_mnist/test_images.txt")[:,:n_test]

	# vector that holds the most probable digit and its error
	DigitandError = (np.ones((n_test,2)))*1e10

	# checks which digit is the most probable for this image
	for j in range(10):

		startTesting = time.time()

		print('Testando dígito ', j, '...', end = ' ')

		Wd = Wdtotal[j].copy()
		Acopy = A.copy()

		# solve the system Wd.H = A
		H = QRFactorizationSimultaneousWithRotgivens(Wd,Acopy)
		#H = QRfactorizationSimultaneousWithHouseholder(Wd,Acopy)

		Erro = (A - np.dot(Wdtotal[j],H))**2

		aux = np.zeros((784))

		for i in range(n_test):

			aux[0:784] = Erro[0:784,i]

			E = math.sqrt(sum(aux))

			if(E < DigitandError[i,1]):

				DigitandError[i,0] = j
				DigitandError[i,1] = E

		endTesting = time.time()
		print(int(endTesting - startTesting), 's')

	endTotal = time.time()
	print('Teste completo! (', int(endTotal - startTotal), 's)')
		
	return DigitandError

#=========================================================

# use the test_index.txt to check the anwers given by the algorithm
def checking(DigitandError):

	print('-'*13)
	print('Checando:')
	print('-'*13)

	answers = np.genfromtxt("dados/dados_mnist/test_index.txt")

	n = DigitandError.shape[0]

	rightAnswers = 0

	errors = np.zeros((n,2))

	errorPerDigit = np.zeros((10))
	amountOfDigit = np.zeros((10))

	for i in range(n):

		if(DigitandError[i,0] == answers[i]): 

			rightAnswers += 1

			amountOfDigit[int(answers[i])] += 1

		else: 

			amountOfDigit[int(answers[i])] += 1

			errorPerDigit[int(answers[i])] += 1

	for i in range(10):

		print('O índice de acertos para', i,'é ', round((100-errorPerDigit[i]*100/amountOfDigit[i]),2), '%')

	print('O índice de acertos total é  ', (float(rightAnswers))*100/n, '%')

#=========================================================
#=========================================================

def factorizate(A,n,p):

	# generate a random matrix W, with all entries in [0,1[
	W = np.random.rand(n,p)

	# store a copy of A (not a reference to A!)
	Acopy = np.copy(A)

	itmax = 0
	Einitial = 1
	Efinal = 0

	At = A.T

	Atcopy = np.copy(At)

	A = np.copy(Acopy)

	while(abs(Einitial-Efinal) > 1e-5 and itmax < 100):

		itmax += 1

		# normalize W
		for j in range(p):

			s = np.sum(W[0:n,j]**2)

			W[0:n,j] = W[0:n,j]/math.sqrt(s)

		# solve the MMQ problem W H = A, determining H
		H = QRFactorizationSimultaneousWithRotgivens(W,A)
		#H = QRfactorizationSimultaneousWithHouseholder(W,A)
	
		H = H.clip(min = 1e-10)

		A = np.copy(Acopy)

		Hcopy = np.copy(H)
		Ht = H.T
		Wt = QRFactorizationSimultaneousWithRotgivens(Ht,At)
		#Wt = QRfactorizationSimultaneousWithHouseholder(Ht,At)

		At = np.copy(Atcopy)

		# compute W
		W = Wt.T

		W = W.clip(min = 1e-10)

		# calculate the error
		Einitial = Efinal

		WH = np.dot(W,Hcopy)

		Efinal = np.sum((Acopy - WH)**2)

	return W,Hcopy

#=========================================================
#=========================================================

# compute cos(teta) and sin(teta)
# more numerically stable
def computeCosSin1(w, i, j, k):

	# compute sin(teta) and cos(teta)
	if (abs(w[i,k]) > abs(w[j,k])):

		# compute tau
		T = -w[j,k]/w[i,k]

		# compute cos(teta)
		cos = 1/( math.sqrt( 1+(T)**2 ) )

		# compute sin(teta)
		sin = cos*T

	else:

		# compute tau
		T = -w[i,k]/w[j,k]

		# compute sin(teta)
		sin = 1/( math.sqrt( 1+(T)**2 ) )

		# compute cos(teta)
		cos = sin*T

	return cos, sin

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
def QRFactorizationSimultaneousWithRotgivens(w,A):

	n = A.shape[0]
	p = w.shape[1]
	m = A.shape[1]

	# apply sucessives Givens Rotations, until W becomes R (a triangular superior matrix)
	for k in range (0, p):

		for j in range (n-1, k, -1):

			i = j - 1

			if (w[j,k] != 0): 

				# compute cos(teta) and sin(teta) to be used in RotGivens
				cos, sin = computeCosSin1(w, i, j, k)

				# apply a Givens' Rotation to matrix W and A
				w[i,k:p] , w[j,k:p] = cos * w[i,k:p] - sin * w[j,k:p] , sin *w[i,k:p] + cos * w[j,k:p]
				A[i,0:m] , A[j,0:m] = cos * A[i,0:m] - sin * A[j,0:m] , sin *A[i,0:m] + cos * A[j,0:m]

	# With this, W was transformed in R (a triangular superior matrix)

	# generates the vector x, which will hold the solution
	h = np.zeros((p,m))

	# solves the system to find each entry of matrix A
	for j in range(0,m):

		for k in range(p-1,-1,-1):

			summation = np.sum(w[k,k+1:p]*h[k+1:p,j])

			if(w[k,k] == 0):

				h[k,j] = (A[k,j] - summation) / 1e-10	
			
			else:			

				h[k,j] = (A[k,j] - summation) / w[k,k]

	return h

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
def QRfactorizationSimultaneousWithHouseholder(w,A):

	n = A.shape[0]
	p = w.shape[1]
	m = A.shape[1]

	for k in range(p):

		x = w[k:n,k]

		norm_x = math.sqrt((x[0:] ** 2).sum())

		rho = -np.sign(x[0])

		uk = x[0] - rho * norm_x

		u = x / uk

		u[0] = 1

		beta = -rho * uk / norm_x

		w[k:n, k:p] = w[k:n, k:p] - beta * np.outer(u, u).dot(w[k:n, k:p])

		A[k:n, 0:m] = A[k:n, 0:m] - beta * np.outer(u, u).dot(A[k:n, 0:m])

	# With this, W was transformed in R (a triangular superior matrix)

	# generates the vector x, which will hold the solution
	h = np.zeros((p,m))

	# solves the system to find each entry of matrix A
	for j in range(0,m):

		for k in range(p-1,-1,-1):

			summation = np.sum(w[k,k+1:p]*h[k+1:p,j])

			if(w[k,k] == 0):

				h[k,j] = (A[k,j] - summation) / 1e-10	
			
			else:			

				h[k,j] = (A[k,j] - summation) / w[k,k]

	return h

#=========================================================
main()