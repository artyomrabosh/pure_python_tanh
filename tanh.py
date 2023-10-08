import typing
from math import exp
from typing import List, Union, Callable
from typing import SupportsFloat as Numeric

Matrix2d = List[List[Numeric]]

def matrix_elementwise_unary(x: Matrix2d, fun: Callable) -> Matrix2d:
    return [[fun(x[i][j]) for i in range(len(x))] for j in range(len(x[0]))]

def matrix_elementwise_binary(x: Matrix2d, y: Matrix2d, fun: Callable) -> Matrix2d:
    return [[fun(x[i][j], y[i][j]) for i in range(len(x))] for j in range(len(x[0]))]

def matrix_multiplication(x: Matrix2d, y: Matrix2d, fun: Callable) -> Matrix2d:
    return [[sum(i * j for i, j in zip(r, c)) for c in zip(*y)] for r in x]



class Tanh():

    @staticmethod
    def scalar_tanh(x: Numeric) -> Numeric:
        return (exp(2 * x) - 1) / (exp(2 * x) + 1)

    @staticmethod
    def scalar_dtanh(x: Numeric) -> Numeric:
        return 1 - Tanh.scalar_tanh(x) ** 2

    @staticmethod
    def __call__(x: Union[Numeric, Matrix2d]) -> Union[Numeric, Matrix2d]:
        if isinstance(x, (int, float)):
            return Tanh.scalar_tanh(x)
        else:
            return matrix_elementwise_unary(x, Tanh.scalar_tanh)

    @staticmethod
    def derive(x : Union[Numeric, Matrix2d]) -> Union[Numeric, Matrix2d]:
        if isinstance(x, (int, float)):
            return Tanh.scalar_dtanh(x)
        else:
            return matrix_elementwise_unary(x, Tanh.scalar_dtanh)
