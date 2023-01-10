# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import perceval.algorithm.differentialEquationSolver as eq
import pytest
import sympy as sp
import numpy as np

x = sp.Symbol("x")
u_0 = sp.Function("u_0")("x")
u_1 = sp.Function("u_1")("x")
u_2 = sp.Function("u_2")("x")
s_0 = sp.Symbol("s_0")
s_2 = sp.Symbol("s_2")

test_array = np.linspace(0, 1, 50)


def test_basic():
    expression = "u_0(x) ** 2 + x + s_0"
    equation = eq.ResultPostProcess(expression)

    assert pytest.approx(equation(test_array, test_array, [1])) == test_array ** 2 + test_array + 1


def test_integral():
    expression = sp.Integral(u_0, x)
    equation = eq.ResultPostProcess(expression)

    assert pytest.approx(equation(test_array, test_array, [])) == test_array ** 2 / 2
    equation_1_lim = eq.ResultPostProcess(expression.subs(x, 1))
    assert pytest.approx(equation_1_lim(test_array, test_array, [])) == 1/2 * 1 ** 2

    equation_2_lims = eq.ResultPostProcess(sp.Integral(u_0, (x, 1/2, 1)))
    assert pytest.approx(equation_2_lims(test_array, test_array, [])) == 1/2 * (1 ** 2 - (1/2) ** 2)


def test_derivative():
    expression = sp.Derivative(u_0, x)
    equation = eq.ResultPostProcess(expression)

    assert pytest.approx(equation(test_array, test_array, [])) == np.ones_like(test_array)

    expression = sp.Derivative(u_0, (x, 2))
    equation = eq.ResultPostProcess(expression)
    assert pytest.approx(equation(test_array, test_array, [])) == np.zeros_like(test_array)


def test_multiple_eq():
    expression = [sp.Integral(u_0, x), sp.Derivative(u_0, x)]
    equation = eq.ResultPostProcess(expression)

    res = equation(test_array, test_array, [])

    assert pytest.approx(res[0]) == test_array ** 2 / 2
    assert pytest.approx(res[1]) == np.ones_like(test_array)


def test_multiple_fn_scalars():
    expression = u_0 ** 2 + x + s_0 + u_1 - s_2 ** 2
    equation = eq.ResultPostProcess(expression)

    second_array = np.linspace(2, 3.5, 50)
    final_array = np.zeros((50, 2))
    final_array[:, 0] = test_array
    final_array[:, 1] = second_array
    scalars = [3, 2, 1]

    assert pytest.approx(equation(final_array, test_array, scalars))\
           == test_array ** 2 + test_array + scalars[0] + second_array - scalars[2] ** 2


def test_equation_loss():
    expression = u_0
    equation = eq.Equation(expression, weight=2)

    test_array = np.linspace(0, 1, 500)

    assert pytest.approx(equation(test_array, test_array, []), rel=1e-2) == 2 / 3  # weight * Integral(x ** 2, (x, 0, 1))


def test_bc_value():
    bc_val = eq.BCValue(0, [1, None, -3])

    assert bc_val.expression == [u_0.subs(x, 0) - 1, u_2.subs(x, 0) + 3]

    bc_val = eq.BCValue(1, 2)
    assert bc_val.expression == [u_0.subs(x, 1) - 2]

    bc_val = eq.BCValue(1, 2, act_on_derivative=True)
    assert bc_val.expression == [sp.Derivative(u_0).subs(x, 1) - 2]


def test_linear_equation():
    A = np.array([[1, 2 * s_2],
                  [s_0, -1]])

    lin_eq = eq.LinearEquation(A)
    assert lin_eq.expression == [- sp.Derivative(u_0) + u_0 + 2 * s_2 * u_1,
                                 - sp.Derivative(u_1) + s_0 * u_0 - u_1]

    lin_eq = eq.LinearEquation(A, n_input=[1, 2], n_eq=[0, 2])
    assert lin_eq.expression == [- sp.Derivative(u_0) + u_1 + 2 * s_2 * u_2,
                                 - sp.Derivative(u_2) + s_0 * u_1 - u_2]

    lin_eq = eq.LinearEquation(1, n_eq=2)
    assert lin_eq.expression == [- sp.Derivative(u_0) + 1. * u_0,
                                 - sp.Derivative(u_1) + 1. * u_1]  # 1. is not simplified automatically


def test_collection():
    expression = u_0 + s_0
    equation = eq.Equation(expression, weight=2)

    test_array = np.linspace(0, 1, 500)

    dec = eq.DECollection(equation).add(equation)

    assert dec.n_out == 1
    assert dec.n_scalar == 1

    assert pytest.approx(dec(test_array, test_array, [0]),
                         rel=1e-2) == 4 / 3  # weight * Integral(x ** 2, (x, 0, 1))