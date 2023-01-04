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
from typing import Union

import sympy as sp

import numpy as np
from multipledispatch import dispatch

from .sympy_parser import lambdify_diff_eq, CallableArray, expr_to_np
from .expression import Expression


class DifferentialEquation(Expression):

    handle_equation_list = True

    def __init__(self, expression: Union[sp.Expr, str, list], weight: Union[float, int] = 1):
        r"""
        :param expression: A Sympy expression or its str representation, or a list of the above.
        :param weight: The multiplier that will be given to this equation.
         Higher values mean this equation will be more taken into account.
        """
        super().__init__(expression)
        self.weight = weight

    def create_func(self, n_out: int, n_scalars: int):
        self._func = lambdify_diff_eq(self.expression, n_out, n_scalars)

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        assert weight > 0, "weight must be a positive number"
        self._weight = weight

    @property
    def sub_properties(self):
        return self._weight

    def __call__(self, y: np.ndarray, x: np.ndarray, scalars: list, with_weight=True):
        r"""
        :param y: The array of functions.
        :param x: The grid on which y and y_prime are defined
        :param scalars: a list of all the scalars. Can be empty if no scalar is needed in the equation.
        :param with_weight: If True, returns the value of the evaluation of the boundary function times the weight.
        :return: The value of the boundary function at the boundary point(s).
         of f and f_prime at these points
        """
        if self._func is None:
            try:
                self.create_func(y.shape[1], len(scalars))
            except IndexError:  # y_prime has only 1 dimension
                self.create_func(1, len(scalars))

        y = y.view(CallableArray)
        y.X = x
        weight = (self.weight if with_weight else 1)
        val = self._func(y, x, scalars)
        if isinstance(self.expression, list):
            res = 0
            for i in range(len(self.expression)):
                res += np.sum(val[i] ** 2) / \
                       (1 if not isinstance(val[i], np.ndarray) or val[i].shape == tuple() else val[i].shape[0])
        else:
            res = np.sum(val ** 2) / (1 if not isinstance(val, np.ndarray) or val.shape == tuple() else val.shape[0])
        return weight * res


class BCValue(DifferentialEquation):

    def __init__(self, point: Union[float, int], values: Union[list, float, int],
                 weight: Union[float, int] = 1, act_on_derivative=False):
        used = self._used_fn_generator(act_on_derivative)

        expr = []
        for i, value in enumerate(values):
            if value is not None:
                expr.append(used(i).subs("x", point) - value)

        super().__init__(expr, weight=weight)

    @staticmethod
    def _used_fn_generator(act_on_derivative):
        def used_fn(i):
            u = sp.Function(f"u_{i}")("x")
            return sp.Derivative(u, "x") if act_on_derivative else u

        return used_fn


class LinearEquation(DifferentialEquation):
    r"""
    Given a (symbolic) array A, linear equation of the form u' = A u.
    """

    def __init__(self, A, n_eq: Union[int, list] = None, n_input: Union[int, list] = None, weight: Union[float, int] = 1):
        r"""
        :param A: a scalar or a matrix. Can be partly composed of sympy Symbols.
         If it is a scalar, it is converted to A * np.eye(n_eq | len(n_eq))
        :param n_eq: The number of equations.
         Can be a list to specify on which u_prime the matrix act (can have repeated / unordered indexes).
         default: A.shape[0]
        :param n_input: The number of inputs to use. Can also be a list to specify which inputs the matrix takes.
         default: A.shape[1]
        :param weight: The multiplier that will be given to this equation.
         Higher values mean this equation will be more taken into account.
        """
        if isinstance(A, (int, float, sp.Expr)):
            assert n_eq is not None, "The size of the system must be defined when using scalar"

            if isinstance(n_eq, int):
                n_eq = list(range(n_eq))
            A = A * np.eye(len(n_eq))

        if isinstance(n_eq, int):
            n_eq = list(range(n_eq))
        if isinstance(n_input, int):
            n_eq = list(range(n_input))

        if n_eq is None:
            n_eq = list(range(A.shape[0]))
        assert len(n_eq) == A.shape[0], f"got {len(n_eq)} equations to set, but matrix has {A.shape[0]} equations"

        if n_input is None:
            n_input = list(range(A.shape[1]))
        assert len(n_input) == A.shape[1], f"got {len(n_input)} inputs, but matrix has {A.shape[1]} inputs"

        eq = []
        for i in range(len(n_eq)):
            expr = - sp.Derivative(sp.Function(f"u_{n_eq[i]}")("x"), "x")
            for j in range(len(n_input)):
                expr += A[i, j] * sp.Function(f"u_{n_input[j]}")("x")
            eq.append(expr)
        super().__init__(eq, weight=weight)


class DECollection:
    r"""
    Holds one or several differential equations.
    """

    def __init__(self, diff_eq=None):
        self._des = []
        if diff_eq is not None:
            self._des.append(diff_eq)

    @property
    def des(self):
        return self._des

    @dispatch(DifferentialEquation)
    def add(self, de):
        self._des.append(de)
        return self

    @dispatch(object)  # Can not dispatch directly DECollection as doesn't exist for now
    def add(self, dec):
        assert isinstance(dec, DECollection), "Only DifferentialEquation and DECollection can be added"
        self._des += dec.des
        return self

    def __call__(self, y: np.ndarray, x: np.ndarray, scalars: list):
        return np.sum([equation(y, x, scalars) for equation in self.des])


class ProcessX(Expression):
    r"""
    :param expression: A Sympy expression or its str representation
    """

    def create_func(self):
        self._func = expr_to_np(self.expression, ["x"])

    def __call__(self, x: float):
        if self._func is None:
            self.create_func()
        return self._func(x)


class AnalyticalSolution(Expression):

    handle_equation_list = True

    def create_func(self):
        self._func = expr_to_np(self.expression, ["x"])

    def __call__(self, x: np.ndarray):
        if self._func is None:
            self.create_func()
        res = self._func(x)
        if isinstance(res, list):
            return np.array(res).swapaxes(0, 1)
        return np.array(res).reshape((len(res), 1))
