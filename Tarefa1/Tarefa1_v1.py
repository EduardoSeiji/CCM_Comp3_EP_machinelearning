#---------------------------------------------------------
# Task 1
# 
#
#
#---------------------------------------------------------

# created just for code testing
def main():

main()

#=========================================================

# apply 1 Givens' Rotation
# more numerically stable
def RotGivens1( wik, wjk, w ):

	# compute sin(teta) and cos(teta)

	if (abs(wik) > abs(wjk)):

		# compute tau
		T = -float(wjk)/wik

		# compute cos(teta)
		cos = 1/( ( 1+T**2 ) ** (1/2) )

		# compute sin(teta)
		sin = cos*T

	else:

		# compute tau
		T = -float(wik)/wjk

		# compute sin(teta)
		sin = 1/( ( 1+T**2 ) ** (1/2) )

		# compute cos(teta)
		cos = sin*T


	# apply a Givens' Rotation to matrix W

#=========================================================

# apply 1 Givens' Rotation
# less numerically stable
def RotGivens2( wik, wjk, w ):

#=========================================================

# apply 1 Givens' Rotation
# the 3rd given algorithm
def RotGivens3( wik, wjk, w ):

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
# the 1st given algorithm
def QRFatoration(w):

	for k in range (1, m+1):

		for j in range (n, k, -1):

			i = j - 1

			if (w[j,k] != 0): 

				# apply a Givens' Rotation to matrix W
				RotGivens1(w[i,k], w[j,k], w)


#=========================================================

# the 2nd given algorithm
def function2():

	for k in range(1,m):

		for j in range(n, k,-1):

			i = j-1

			if()

	for k in range(m,0,-1):



#=========================================================

# Rot-givens
# the 3rd given algorithm
def function3():

	for r in range(1,m+1):

		aux = c * 

#=========================================================

# print the given matrix, just for testing

def printMatrix(matrix):

	for i in range(0,len(matrix)):

		for j in range(len(matrix[i])):

			print(matrix[i,j])

		print()

	print()

#=========================================================

