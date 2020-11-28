#---------------------------------------------------------
# Main task
# 
# Goal: teach a machine how to read manuscripted digits
#
# the non-negative matrix factoration will be used
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


# TAREFA 2 É O HEART

'''
TAREFA PRINCIPAL

INPUTS
INSERIR P. QUANTO MAIOR, MAIS GL, MAIS PRECISO. é o numero de colunas de W e linhas de H (acho). número de gl da fatoração. Aumenta bastante o tempo computacional. tempo e p sao lineares
SE NDIG TREINO ALTO, MAIS PRECISO. é usado p ler o txt por digito (train dig n). Determina quantos digitos pegar por txt (max10000 acho). tempo e ndig tem uma relação estranha
NUMERO DE TESTES: NUMERO DE IMAGENS DO BANCO DE DADOS (10000). usa na etapa de classificar. nao demora mto, leva uns 3s/digito

COMEÇAR O CRONOMETRO: FROM TIME IMPORT TIME

MATRIZ WDTOT VAI INICIAR VAZIA
VETOR CJ (ERROS DAS NORMAS EUCLIDIANAS, P DECIDIR QUAL O DIGITO) TB

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


#=========================================================

def oneDigitTraining():



#=========================================================

# each image is a nx x ny matrix, which value [0,255]
# processing transform the images in one matrix A
def processImage():

	# read m images (nx x ny) from the database
	pegarocodigo do ep passado

	# resize each image to a columm vector (n = nxny lines)
	# it can be done only stacking the columms of the image
	fazer

	# merge all columm vectors to form a matrix A (nxm)
	# each columm of A is an image
	fazer

	# normalize the entries of A to [0,1]
	dividir cada elemento da matriz por 255

	# obs: later, the matrix A will be factorized in WH

#=========================================================

def learning():



#=========================================================

#=========================================================
main()