import unittest
import random
from itertools import product

import numpy as np
from numpy import tanh

from tanh import Tanh
from num_diff import numdiff_central

my_tanh = Tanh()


class TestTanh(unittest.TestCase):
    STEP = 1e-4
    START = -2
    STOP = 2

    def test_scalar_forward(self):
        inputs = list(np.arange(start=self.START, stop=self.STOP, step=self.STEP))
        outputs_exp = [tanh(inp) for inp in inputs]
        outputs_act = [my_tanh(inp) for inp in inputs]
        for exp, act in zip(outputs_exp, outputs_act):
            self.assertAlmostEqual(exp, act, 15)

    def test_matrix_forward(self):
        size = 100
        n = 10
        inputs = [[[random.random()] * n for _ in range(n)] for _ in range(size)]
        outputs_exp = [tanh(np.array(inp).T) for inp in inputs]
        outputs_act = [my_tanh(inp) for inp in inputs]

        for exp, act in zip(outputs_exp, outputs_act):
            for i, j in product(range(n), range(n)):
                self.assertAlmostEqual(exp[i][j], act[i][j], 15)

    def test_scalar_derive(self):
        inputs = list(np.arange(start=self.START, stop=self.STOP, step=self.STEP))
        outputs_exp = numdiff_central(inputs, self.STEP, tanh)
        outputs_true = [my_tanh.derive(inp) for inp in inputs]
        for exp, true in zip(outputs_exp, outputs_true):
            self.assertAlmostEqual(exp, true, 7)

    def test_matrix_derive(self):
        n = 10
        linspace = np.arange(start=self.START, stop=self.STOP, step=self.STEP)
        matrix_inputs = [[[x for _ in range(n)] for _ in range(n)] for x in linspace]

        outputs_exp = [[numdiff_central(linspace, self.STEP, tanh) for _ in range(n)] for _ in range(n)]
        outputs_true = [my_tanh.derive(inp) for inp in matrix_inputs]
        outputs_exp = np.transpose(np.array(outputs_exp), (2, 0, 1)).tolist()
        for exp, act in zip(outputs_exp, outputs_true):
            for i, j in product(range(n), range(n)):
                self.assertAlmostEqual(exp[i][j], act[i][j], 7)


if __name__ == '__main__':
    unittest.main()
