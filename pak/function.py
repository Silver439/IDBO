import numpy as np
from bbobold import bbobbenchmarks as bn
from numpy.random import standard_normal as _randn

sigma = [12.57,10044455.67,418.57,171.86,29.02,237396.11,431.55,46480.79,21715.05,9634378.45,21241083.28,9607260658.85,
         449.99,41.04,521.74,79.07,19.00,308.17,74.72,45818.02,12.21,24.05,20.12,18.18]

def fGauss(ftrue,index,beta):
    """Returns Gaussian model noisy value."""
    # expects ftrue to be a np.array
    popsi = np.shape(ftrue)
    fval = ftrue + sigma[index-1] * beta * _randn(popsi) # with gauss noise
    return fval

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

def Fn1S(x):
    f = bn.F1(1)
    return fGauss(f(x),1,0.05)

def Fn2S(x):
    f = bn.F2(10)
    return fGauss(f(x),2,0.05)

def Fn7S(x):
    f = bn.F7(1)
    return fGauss(f(x),7,0.05)

def Fn8S(x):
    f = bn.F8(3)
    return fGauss(f(x),8,0.05)

def Fn10S(x):
    f = bn.F10(2)
    return fGauss(f(x),10,0.05)

def Fn13S(x):
    f = bn.F13(18)
    return fGauss(f(x),13,0.05)

def Fn14S(x):
    f = bn.F14(3)
    return fGauss(f(x),14,0.05)

def Fn15S(x):
    f = bn.F15(2)
    return fGauss(f(x),15,0.05)

def Fn18S(x):
    f = bn.F18(7)
    return fGauss(f(x),18,0.05)

def Fn19S(x):
    f = bn.F19(2)
    return fGauss(f(x),19,0.05)

def Fn22S(x):
    f = bn.F22(5)
    return fGauss(f(x),22,0.05)

def Fn23S(x):
    f = bn.F23(6)
    return fGauss(f(x),23,0.05)


def Fn1M(x):
    f = bn.F1(1)
    return fGauss(f(x),1,0.2)

def Fn2M(x):
    f = bn.F2(10)
    return fGauss(f(x),2,0.2)

def Fn7M(x):
    f = bn.F7(1)
    return fGauss(f(x),7,0.2)

def Fn8M(x):
    f = bn.F8(3)
    return fGauss(f(x),8,0.2)

def Fn10M(x):
    f = bn.F10(2)
    return fGauss(f(x),10,0.2)

def Fn13M(x):
    f = bn.F13(18)
    return fGauss(f(x),13,0.2)

def Fn14M(x):
    f = bn.F14(3)
    return fGauss(f(x),14,0.2)

def Fn16M(x):
    f = bn.F16(1)
    return fGauss(f(x),16,0.2)

def Fn18M(x):
    f = bn.F18(7)
    return fGauss(f(x),18,0.2)

def Fn19M(x):
    f = bn.F19(2)
    return fGauss(f(x),19,0.2)

def Fn22M(x):
    f = bn.F22(5)
    return fGauss(f(x),22,0.2)

def Fn23M(x):
    f = bn.F23(6)
    return fGauss(f(x),23,0.2)


def Fn1L(x):
    f = bn.F1(1)
    return fGauss(f(x),1,0.5)

def Fn2L(x):
    f = bn.F2(10)
    return fGauss(f(x),2,0.5)

def Fn7L(x):
    f = bn.F7(1)
    return fGauss(f(x),7,0.5)

def Fn8L(x):
    f = bn.F8(3)
    return fGauss(f(x),8,0.5)

def Fn10L(x):
    f = bn.F10(2)
    return fGauss(f(x),10,0.5)

def Fn13L(x):
    f = bn.F13(18)
    return fGauss(f(x),13,0.5)

def Fn14L(x):
    f = bn.F14(3)
    return fGauss(f(x),14,0.5)

def Fn18L(x):
    f = bn.F18(7)
    return fGauss(f(x),18,0.5)

def Fn19L(x):
    f = bn.F19(2)
    return fGauss(f(x),19,0.5)

def Fn22L(x):
    f = bn.F22(5)
    return fGauss(f(x),22,0.5)

