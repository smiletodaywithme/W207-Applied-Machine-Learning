## Using Singular Value Decomposition (SVD) for manually performing a pseudoinverse on a non-square matrix
# by Alexander I. Iliev - ailiev@berkeley.edu - Nov.19.2018
# This will not be the actual inverse matrix, but the "best approximation"

from numpy import random, matrix, linalg, diag, allclose, dot

# Create Matrix A with size (3,5) containing random numbers:
A = random.random(15)
A = A.reshape(3,5)
A = matrix(A)

# 1-3. Using the SVD function will return:
U,s,V = linalg.svd(A, full_matrices=False)

# Constrcut a giagonal matrix 'S', from the giagonal 's':
S = diag(s)

# 2-3. Invert the square diagonal matrix by inverting each diagonal element:
S[0,0], S[1,1], S[2,2] = 1/diag(S)

# 3-3. Now we use the SVD elements to obtain the pseudo-inverse of matrix A:
X = dot(U, dot(S, V))
X = X.T # Final step: we must transpose

# Check each matrix:
print('A has the shape:',A.shape,', U has the shape:',U.shape,',\
 S has the shape:',S.shape,', V has the shape:',V.shape)

# The inverse of A is:
print('The inverse of A is:\n',A.I)

# Comparison test 1:
A.I-X

# Comparison test 2:
print(allclose(A.I, X))

## Compression using SVD:


## Task to students in class in break-room:
# Find the best approximate of A.I after compressing by eliminating the most insignificant singular value S
# and its corresponding Us and Vs.
# - Compare the results for larger matrices.
# - comment on your results