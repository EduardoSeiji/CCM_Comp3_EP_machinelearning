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

	# number of columms of W and lines of H (degrees of freedom)
	p = int(input('Digite o valor de p (5, 10, 15): '))

	# number of images used in the training of each digit
	ndig_treino = int(input('Digite o valor de ndig_treino (100, 1000, 4000): '))

	# number of images used in testing


	# executes the training
	training(n,p,ndig_treino)



#=========================================================

# use the train_digX.txt archives (x in [0,9])
# each txt is 784x5300 (each columm is a digit image)
def training(n,p,ndig_treino):

	# matrix which will hold all the Wds (classificators)
	Wdtotal = []

	print('-'*13)
	print('Treinando...')
	print('-'*13)

	startTotal = time.time()

	# train the classificators generating a Wd matrix for each digit
	for i in range(10):

		start = time.time()

		print('Treinando o dígito ', i, '...')

		# read the train_digX.txt archive, x in [0,9]
		# generating a matrix A (n=784 x m=ndig_treino)
		A = np.genfromtxt("dados/dados_mnist/train_dig" + str(i) + ".txt")[:,:ndig_treino]

		# calculate the matrix Wd (n x p), by factorizating
		# A = Wd.H
		Wd,H = factorizate(A,n,ndig_treino,p)

		# append Wd to Wdtotal
		Wdtotal.append(Wd)


		end = time.time()

		print(int(end - start), 's')

	print('Treinamento completo!')
	endTotal = time.time()
	print(int(endTotal - startTotal), 's')

#=========================================================

# use the test_images.txt archive
# each txt is 784x10000 (each columm is a testing image)
# use the test_index.txt archive
# it gives the answers
#def testing():



#=========================================================
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