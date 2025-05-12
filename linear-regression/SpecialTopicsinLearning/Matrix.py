#Def to calculate the transposed Matrix
def transpM(m):
    return map(list,zip(*m))

#Def to calculate Minor of the Matrix
def gMMin(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

#Def to calculate Determinant
def gMatrixDeterminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*gMatrixDeterminant(gMMin(m,0,c))
    return determinant

#Def to calculate inverse
def getMatrixInverse(m):
    determinant = gMatrixDeterminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #finding matrix of cofactor
    cofac = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = gMMin(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * gMatrixDeterminant(minor))
        cofac.append(cofactorRow)
    cofac = list(transpM(cofac))
    for r in range(len(cofac)):
        for c in range(len(cofac)):
            cofac[r][c] = cofac[r][c]/determinant
    return cofac
