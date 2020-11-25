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

# created just for code testing
def main():

	teste = input("Digite o item que deseja testar: ")

	#---------------------------------------------------

	# test a
	if(teste == 'a'):

		n = 64
		m = 64

		# create the w matrix
		# create a null matrix
		w = matrix = n*[m*[0]]
		
		# fill it properly
		for i in range(n):
		
			for j in range(m):
		
				if(i == j): w[i][j] = 2

				elif(abs(i-j) == 1): w[i][j] = 1

				elif(abs(i-j) > 1): w[i][j] = 0

		# print matrix just for checking
		printMatrix(w)

		# create b vector
		b = n*[1]

		# print vector just for checking
		printVector(w)

	#---------------------------------------------------

	# test b
	if(teste == 'b'):

		n = 20
		m = 17

		# create the w matrix
		# create a null matrix
		w = matrix = n*[m*[0]]
		
		# fill it properly
		for i in range(n):
		
			for j in range(m):

				if(abs(i-j) <= 4): w[i][j] = 1/(i+j-1)

				elif(abs(i-j) > 4): w[i][j] = 0

		# print matrix just for checking
		printMatrix(w)

		# create b vector
		b = n*[0]

		# fill it properly
		for i in range(n):

			b[i] = i

		# print vector just for checking
		printVector(w)

	#---------------------------------------------------
	
	# test c
	if(teste == 'b'):

		n = 64
		p = 64

		# create the w matrix
		# create a null matrix
		w = matrix = n*[p*[0]]
		
		# fill it properly
		for i in range(n):
		
			for j in range(m):

				if(i == j): w[i][j] = 2

				elif(abs(i-j) == 1): w[i][j] = 1

				elif(abs(i-j) > 1): w[i][j] = 0

		# print matrix just for checking
		printMatrix(w)

		# define m = 3, solving 3 simultaneous systems
		#(???)

	# test d
	if(teste == 'b'):

		n = 20
		m = 17

		# create the w matrix
		# create a null matrix
		w = matrix = n*[m*[0]]
		
		# fill it properly
		for i in range(n):
		
			for j in range(m):

				if(abs(i-j) <= 4): w[i][j] = 1/(i+j-1)

				elif(abs(i-j) > 4): w[i][j] = 0

		# print matrix just for checking
		printMatrix(w)

		# define m = 3, solving 3 simultaneous systems
		#(???)

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
def RotGivens(w, i, j, k):

	cos, sin = computeCosSin1(w, i, j, k)

	for r in range(1,m+1):

		aux = cos*w[i][r] - sin*w[j][r]

		w[j][r] = sin*w[i][r] + cos*w[j][r]

		w[i][r] = aux

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
# the 1st given algorithm
def QRFatoration(w):

	for k in range (1, m+1):

		for j in range (n, k, -1):

			i = j - 1

			if (w[j][k] != 0): 

				# apply a Givens' Rotation to matrix W
				RotGivens(w, i, j, k)

#=========================================================

# solve the system of equations
# the 2nd given algorithm
def function2(w, b, x):

	for k in range(1,m):

		for j in range(n, k,-1):

			i = j-1

			if(w[j][k] != 0):

				# apply a Givens' Rotation to matrix W
				RotGivens1(w, i, j, k)

	# solve the resulting triangular system
	# since w is always overwritted, it'll have the R values at the end
	for k in range(m,0,-1):

		summation = 0

		for j in range(k+1,m):

			summation += w[k][j]*x[j]

		x[k] = (b[k] - summation) / w[k][k]



#=========================================================

# print the given matrix
# just for testing
def printMatrix(matrix):

	for i in range(0,len(matrix)):

		for j in range(len(matrix[i])):

			print(matrix[i][j])

		print()

	print()

#=========================================================

# print the given vector
# just for testing
def printVector(vector):

	for i in range(0,len(vector)):

		print(vector[i], end = ' ')

	print()

#=========================================================
main()