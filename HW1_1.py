import math
import numpy as np



def newton_method(f, df, x1, tol, max_iter):
    x = x1
    iter1 = 0
    arr1 = np.array([x1])
    for i in range(0,max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return arr1, iter1
        x = x - fx / df(x)
        arr1 = np.append(arr1, x)
        iter1 += 1
    return None

def bisection_method(f, a, b, tol, max_iter):
    iter2 = 0
    x = (a+b)/2
    fx = f(x)
    fa = f(a)
    fb = f(b)
    arr2 = np.array([])
    while abs(fx) > tol and iter2 < max_iter:
        c = (a+b)/2
        fc = f(c)
        if fa*fc < 0:
            b = c
            fb = fc    
        elif fb*fc < 0:
            a = c
            fa = fc
        else:
            arr2 = np.append(arr2, c)
            return arr2, iter2
        arr2 = np.append(arr2, (a+b)/2)
        iter2 += 1
    return arr2, iter2

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

x_newton = result1[0]
np.save('A1.npy',x_newton)
x_mid = result2[0]
np.save('A2.npy',x_mid)
num_iter = np.array([result1[1],result2[1]])
np.save('A3.npy',num_iter)

'''
print("newton:", result1[0])
print("bisection:", result2[0])
'''