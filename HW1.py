import math
import numpy as np



def newton_method(f, df, x1, tol, max_iter):
    x = x1
    arr1 = np.array([])
    for i in range(max_iter):
        fx = f(x)
        arr1 = np.append(arr1, x)
        if abs(fx) < tol:
            x = x - fx / df(x)
            arr1 = np.append(arr1, x)
            return x, arr1, i+1
        if df == 0:
            return None, arr1, i+1
        x = x - fx / df(x)
    return None, arr1, max_iter

def bisection_method(f, a, b, tol, max_iter):
    arr2 = np.array([])
    for i in range(max_iter):
        c = (a+b)/2
        fc = f(c)
        arr2 = np.append(arr2, c)
        if abs(fc) < tol:
            return c, arr2, i+1
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return None, arr2, i+1

def f(x):
    return x * math.sin(3*x) - math.exp(x)

def df(x):
    return 3*x*math.cos(3*x) - math.exp(x) + math.sin(3*x)

a = -0.7
b = -0.4
x1 = -1.6
tol = 1e-6
max_iter = 100
result1 = newton_method(f, df, x1, tol, max_iter)
result2 = bisection_method(f, a, b, tol, max_iter)

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3],[0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])


A1 = result1[1]
A2 = result2[1]
A3 = np.array([result1[2],result2[2]])

A4 = A + B
A5 = 3*x - 4*y
A6 = np.dot(A, x)
A7 = np.dot(B, (x-y))
A8 = np.dot(D, x)
A9 = np.dot(D, y) + z
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)


np.save('A1.npy', A1)
np.save('A2.npy', A2)
np.save('A3.npy', A3)
np.save('A4.npy',A4)
np.save('A5.npy',A5)
np.save('A6.npy',A6)
np.save('A7.npy',A7)
np.save('A8.npy',A8)
np.save('A9.npy',A9)
np.save('A10.npy',A10)
np.save('A11.npy',A11)
np.save('A12.npy',A12)

'''
print(np.load('A5.npy'))
print(np.load('A6.npy'))
print(np.load('A7.npy'))
print(np.load('A8.npy'))
print(np.load('A9.npy'))
print(np.load('A10.npy'))
'''
