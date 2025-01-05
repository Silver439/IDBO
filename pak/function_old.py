import numpy as np
from bbobold import bbobbenchmarks as bn
from numpy.random import standard_normal as _randn

def fGauss(ftrue, beta):
    """Returns Gaussian model noisy value."""
    # expects ftrue to be a np.array
    popsi = np.shape(ftrue)
    fval = ftrue * np.exp(beta * _randn(popsi)) # with gauss noise
    tol = 1e-8
    fval = fval + 1.01 * tol
    idx = ftrue < tol
    '''
    try:
        fval[idx] = ftrue[idx]
    except IndexError: # fval is a scalar
    '''
    if idx:
        fval = ftrue
    return fval

def Fn1S(x):
    f = bn.F1(1)
    return fGauss(f(x),0.01)
# 79.48
def Fn8S(x):
    f = bn.F8(3)
    return fGauss(f(x),0.01)
# 98.62
def Fn7S(x):
    f = bn.F7(1)
    return fGauss(f(x),0.01)
# 92.94
def Fn10S(x):
    f = bn.F10(2)
    return fGauss(f(x),0.01)
# 59.13
def Fn14S(x):
    f = bn.F14(3)
    return fGauss(f(x),0.01)
# 77.31
def Fn18S(x):
    f = bn.F18(7)
    return fGauss(f(x),0.01)
# 119.54
def Fn19S(x):
    f = bn.F19(2)
    return fGauss(f(x),0.01)
# 71.69
def Fn22S(x):
    f = bn.F22(5)
    return fGauss(f(x),0.01)
# 51.57

def Fn1M(x):
    f = bn.F1(1)
    return fGauss(f(x),0.1)
# 79.48
def Fn2M(x):
    f = bn.F2(10)
    return fGauss(f(x),0.1)
# 66.95
def Fn8M(x):
    f = bn.F8(3)
    return fGauss(f(x),0.1)
# 98.62
def Fn7M(x):
    f = bn.F7(1)
    return fGauss(f(x),0.1)
# 92.94
def Fn10M(x):
    f = bn.F10(2)
    return fGauss(f(x),0.1)
# 59.13
def Fn13M(x):
    f = bn.F13(18)
    return fGauss(f(x),0.1)
# 68.42
def Fn14M(x):
    f = bn.F14(3)
    return fGauss(f(x),0.1)
# 77.31
def Fn16M(x):
    f = bn.F16(1)
    return fGauss(f(x),0.1)
# 71.35
def Fn18M(x):
    f = bn.F18(7)
    return fGauss(f(x),0.1)
# 119.54
def Fn19M(x):
    f = bn.F19(2)
    return fGauss(f(x),0.1)
# 71.69
def Fn22M(x):
    f = bn.F22(5)
    return fGauss(f(x),0.1)
# 51.57

def Fn1L(x):
    f = bn.F1(1)
    return fGauss(f(x),1)
# 79.48
def Fn8L(x):
    f = bn.F8(3)
    return fGauss(f(x),1)
# 98.62
def Fn7L(x):
    f = bn.F7(1)
    return fGauss(f(x),1)
# 92.94
def Fn10L(x):
    f = bn.F10(2)
    return fGauss(f(x),1)
# 59.13
def Fn14L(x):
    f = bn.F14(3)
    return fGauss(f(x),1)
# 77.31
def Fn18L(x):
    f = bn.F18(7)
    return fGauss(f(x),1)
# 119.54
def Fn19L(x):
    f = bn.F19(2)
    return fGauss(f(x),1)
# 71.69
def Fn22L(x):
    f = bn.F22(5)
    return fGauss(f(x),1)
# 51.57

def F1(x):
    return bn.F1(1)(x) # 79.48
def F2(x):
    return bn.F2(10)(x) # 66.95
def F3(x):
    return bn.F3(2)(x) # 77.66
def F4(x):
    return bn.F4(2)(x) # 77.66
def F5(x):
    return bn.F5(3)(x) # 66.71
def F6(x):
    return bn.F6(16)(x) # 65.87
def F7(x):
    return bn.F7(1)(x) # 92.94
def F8(x):
    return bn.F8(3)(x) # 98.62
def F9(x):
    return bn.F9(5)(x) # 65.61
def F10(x):
    return bn.F10(2)(x) # 59.13
def F11(x):
    return bn.F11(1)(x) # 76.27
def F12(x):
    return bn.F12(3)(x) # 56.61
def F13(x):
    return bn.F13(18)(x) # 68.42
def F14(x):
    return bn.F14(3)(x) # 77.31
def F15(x):
    return bn.F15(2)(x) # 70.03
def F16(x):
    return bn.F16(1)(x) # 71.35
def F17(x):
    return bn.F17(8)(x) # 69.83
def F18(x):
    return bn.F18(7)(x) # 119.54
def F19(x):
    return bn.F19(2)(x) # 71.69
def F20(x):
    return bn.F20(7)(x) # 71.29
def F21(x):
    return bn.F21(7)(x) # 124.08
def F22(x):
    return bn.F22(5)(x) # 51.57
def F23(x):
    return bn.F23(6)(x) # 85.39
def F24(x):
    return bn.F24(2)(x) # 93.30