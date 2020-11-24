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

main()

#=========================================================

# compute cos(teta) and sin(teta)
# more numerically stable
def computeCosSin1(w, i, j, k):

	# compute sin(teta) and cos(teta)

	if (abs(w[i,k]) > abs(w[j,k])):

		# compute tau
		T = -float(w[j,k])/w[i,k]

		# compute cos(teta)
		cos = 1/( ( 1+T**2 ) ** (1/2) )

		# compute sin(teta)
		sin = cos*T

	else:

		# compute tau
		T = -float(w[i,k])/w[j,k]

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
	cos = w[i,k] / (w[i,k]**2 + w[j,k]**2)**(1/2)

	# compute sin(teta)
	sin = -w[j,k] / (w[i,k]**2 + w[j,k]**2)**(1/2)

	return cos, sin

#=========================================================

# apply 1 Givens' Rotation
# the 3rd given algorithm
def RotGivens(w, i, j, k):

	cos, sin = computeCosSin1(w, i, j, k)

	for r in range(1,m+1):

		aux = cos*w[i,r] - sin*w[j,r]

		w[j,r] = sin*w[i,r] + cos*w[j,r]

		w[i,r] = aux

#=========================================================

# apply sucessives Givens' Rotations in a convenient order
# the 1st given algorithm
def QRFatoration(w):

	for k in range (1, m+1):

		for j in range (n, k, -1):

			i = j - 1

			if (w[j,k] != 0): 

				# apply a Givens' Rotation to matrix W
				RotGivens(w, i, j, k)

#=========================================================

# solve the system of equations
# the 2nd given algorithm
def function2(w, b, x):

	for k in range(1,m):

		for j in range(n, k,-1):

			i = j-1

			if(w[j,k] != 0):

				# apply a Givens' Rotation to matrix W
				RotGivens1(w, i, j, k)

	# solve the resulting triangular system
	# since w is always overwritted, it'll have the R values at the end
	for k in range(m,0,-1):

		summation = 0

		for j in range(k+1,m):

			summation += w[k,j]*x[j]

		x[k] = (b[k] - summation) / w[k,k]



#=========================================================

# print the given matrix
# just for testing
def printMatrix(matrix):

	for i in range(0,len(matrix)):

		for j in range(len(matrix[i])):

			print(matrix[i,j])

		print()

	print()

#=========================================================

