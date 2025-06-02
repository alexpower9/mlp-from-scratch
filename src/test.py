import numpy as np 

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    
    if isinstance(a[0], np.ndarray): #change this for the actual solution
        cols = len(a[0])
    else:
        cols = 1
    result = [[0 for _ in range(len(a))] for _ in range(cols)]


    for i in range(len(a)):
        for j in range(cols):
            element = a[i][j]
            result[j][i] = element

    return result

def make_diagonal(x):
    size = len(x)

    result = [[0 for _ in range(len(x))] for _ in range(len(x))]

    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                result[i][j] = x[i]

    return result





arr1 = np.random.randint(low=0, high=10, size=(4))
res = np.array(make_diagonal(arr1))

print(arr1)
print(res)
