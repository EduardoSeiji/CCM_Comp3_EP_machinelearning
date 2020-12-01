#---------------------------------------------------------
# Main task
# 
# Goal: teach a machine how to read manuscripted digits
#
# The non-negative matrix factoration will be used
#
#
#
#
#
#---------------------------------------------------------

# TAREFA 2 É O HEART

'''
TAREFA PRINCIPAL

INPUTS:

INSERIR P. QUANTO MAIOR, MAIS GL, MAIS PRECISO. 
é o numero de colunas de W e linhas de H (acho). 
número de gl da fatoração. 
Aumenta bastante o tempo computacional. 
tempo e p sao lineares

SE NDIG TREINO ALTO, MAIS PRECISO. 
é usado p ler o txt por digito (train dig n). 
Determina quantos digitos pegar por txt (max10000 acho). 
tempo e ndig tem uma relação estranha

NUMERO DE TESTES: NUMERO DE IMAGENS DO BANCO DE DADOS (10000).
usa na etapa de classificar. nao demora mto, leva uns 3s/digito


COMEÇAR O CRONOMETRO: FROM TIME IMPORT TIME


MATRIZ WDTOT VAI INICIAR VAZIA
VETOR CJ TB (ERROS DAS NORMAS EUCLIDIANAS, P DECIDIR QUAL O DIGITO)

FASE DE APRENDIZAGEM (PARTE Q DEMORA, VER ONDE DA P OTIMIZAR: INTEGRAR FUNÇÃO, VETORIZAR O Q PUDER (SOMA, ERRO)...)
FOR I IN RANGE 10(10 DIGITOS)
	FUNÇÃO TREINA DIGITO
		LER
		USAR A TAREFA2 (FATORAR Ad = WdH)
		VOU QUERER O Wd, DE CADA DIGITO
		DAR APPEND NA WDTOT

TREINAMENTO ACABOU

AGORA COMEÇA A CLASSIFICAÇÃO:

NP.LOADTEXT(DADOSMNIST...)

RODAR UM FOR OU VETORIZAÇÃO

TEMPO:
	TREINAMENTO: 
		10-15s/digito (no caso mais facil, 100, 10mil, 5)
		média 300s/digito (no caso mais dificil, 4000, 10000, 15)
	colocar o cronometro dentro do laço for, p ver o tempo de cada digito

passo 3: calcular o % de erro
	
	pegar o gabarito (text index txt, c 10000 linhas, c o digito n. matrix), copiar, fazer função q conta os acertos
	na matriz contadora (2x10)
	printar o acerto em cada digito e o geral

	piupiu: acuracia smp acima de 90% (tem uns digitos q nao kk) 
	se aumentar o p, ele nivela os piores, melhora os q tava ruim


relatório do Piupiu: 32 páginas
dica: colocar varias fotos e fazer uma conclusao baseada na tarefa final

cipa fazer o metodo de househholder p testar (Maitah e Miki falaram q é lerdao)
'''

import math
import numpy as np
import time

# created just for code testing
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
		#p = 5
		p = int(input('> Digite o valor de p (5, 10, 15): '))

		# number of images used in the training of each digit
		#ndig_treino = 100
		ndig_treino = int(input('> Digite o valor de ndig_treino (100, 1k, 4k): '))

		#executes the training
		Wdtotal = training(n,p,ndig_treino)

	# import data to go to straight the testing
	if(a == 'b'):

		print('Carregando classificadores...')

		for i in range(10):

			Wdtotal.append(np.genfromtxt("treinado/treinado_" + str(i) + ".txt"))

	# number of images used in testing
	#n_test = 10000
	n_test = int(input('> Digite o valor de n_test (1 a 10k): '))

	# executes the testing
	DigitandError = testing(n_test, Wdtotal)

	# check the result
	# 88% piupiu
	# 94% p caso mais osso
	checking(DigitandError)

#=========================================================

# use the train_digX.txt archives (x in [0,9])
# each txt is 784x5300 (each columm is a digit image)
def training(n,p,ndig_treino):

	# matrix which will hold all the Wds (classificators)
	Wdtotal = []

	print('-'*13)
	print('Treinando:')
	print('-'*13)

	startTotal = time.time()

	# train the classificators generating a Wd matrix for each digit
	for i in range(10):

		print('Treinando o dígito ', i, '...', end = '  ')

		# calculate the matrix Wd (n x p), by factorizating
		# A = Wd.H
		A = np.genfromtxt("dados/dados_mnist/train_dig" + str(i) + ".txt")[0:784,0:ndig_treino]/255.0

		print('Dados lidos')

		start = time.time()

		print('Bora p o factorizate')
		Wd,H = factorizate(A,n,p)
		print('Factorizate feito')

		# append Wd to Wdtotal
		Wdtotal.append(Wd)

		end = time.time()

		print(int(end - start), 's')

	endTotal = time.time()
	print('Treinamento completo! (', int(endTotal - startTotal), 's)')

	startSaving = time.time()

	print('Salvando matriz de treino...', end = ' ')
	for i in range(10):

		np.savetxt("treinado/treinado_" + str(i) + ".txt",Wdtotal[i],fmt='%f')

	endSaving = time.time()
	print(int(endSaving - startSaving), 's')

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

		Wd = Wdtotal[j].copy() # tanto faz essa linha
		Acopy = A.copy()

		# solve the system Wd.H = A
		H = QRfactorizationSimultaneous(Wd,Acopy)

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

def checking(DigitandError):

	answers = np.genfromtxt("dados/dados_mnist/test_index.txt")

	n = DigitandError.shape[0]

	rightAnswers = 0

	print('-'*13)
	print('Checando:')
	print('-'*13)


	errors = np.zeros((n,2))

	counter = 0

	for i in range(n):

		if(DigitandError[i,0] == answers[i]): rightAnswers += 1

	print('O índice de acertos é de ', (float(rightAnswers))*100/n, '%')

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
		H = QRfactorizationSimultaneous(W,A)
	
		H = H.clip(min = 1e-10)

		A = np.copy(Acopy)

		Hcopy = np.copy(H)
		Ht = H.T
		Wt = QRfactorizationSimultaneous(Ht,At)

		At = np.copy(Atcopy)

		# compute W
		W = Wt.T

		W = W.clip(min = 1e-10)

		# calculate the error
		Einitial = Efinal

		WH = np.dot(W,Hcopy)

		Efinal = np.sum((Acopy - WH)**2)
		#Efinal = np.square(Acopy-np.matmul(W,H)).sum()

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

# apply 1 Givens' Rotation
# use vectorization instead of a for loop (like RotGivens2), much faster
def RotGivens(w, i, j, k, cos, sin, m):

	w[i,k:m] , w[j,k:m] = cos * w[i,k:m] - sin * w[j,k:m] , sin *w[i,k:m] + cos * w[j,k:m]

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
def QRfactorizationSimultaneous(w,A):

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
				#RotGivens(w, i, j, k, cos, sin, p)
				#RotGivens(A, i, j, 0, cos, sin, m)	
				w[i,k:p] , w[j,k:p] = cos * w[i,k:p] - sin * w[j,k:p] , sin *w[i,k:p] + cos * w[j,k:p]
				A[i,0:m] , A[j,0:m] = cos * A[i,0:m] - sin * A[j,0:m] , sin *A[i,0:m] + cos * A[j,0:m]

	# With this, W was transformed in R (a triangular superior matrix)

	# generates the vector x, which will hold the solution
	h = np.zeros((p,m))

	# solves the system to find each entry of matrix A
	for j in range(0,m):

		for k in range(p-1,-1,-1):

			'''
			summation = 0
			
			for i in range(k+1,p):

				summation += w[k,i]*h[i,j]
			'''
			summation = np.sum(w[k,k+1:p]*h[k+1:p,j])

			if(w[k,k] == 0):

				h[k,j] = (A[k,j] - summation) / 1e-10	
			
			else:			

				h[k,j] = (A[k,j] - summation) / w[k,k]

	return h

#=========================================================
main()