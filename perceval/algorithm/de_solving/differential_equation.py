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
from abc import ABC, abstractmethod
from typing import Union

import sympy as sp

import numpy as np
from multipledispatch import dispatch

from .sympy_parser import lambdify_diff_eq, CallableArray, expr_to_np


class Expression(ABC):

    handle_equation_list = False

    def __init__(self, expression: Union[sp.Expr, str], **kwargs):
        r"""
        :param expression: A Sympy expression or its str representation,
         or a list of the above if the current class can handle them.
        """
        self._func = None
        self.expression = expression

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, expr):
        if isinstance(expr, str):
            expr = sp.parse_expr(expr)
        if isinstance(expr, list):
            if not self.handle_equation_list:
                raise RuntimeError(f"{type(self).__name__} class does not accept lists of equations")
            for i, ss_expr in enumerate(expr):
                if isinstance(ss_expr, str):
                    expr[i] = sp.parse_expr(ss_expr)
                assert isinstance(expr[i], sp.Expr), "expression must be a sympy expression"
        else:
            assert isinstance(expr, sp.Expr), "expression must be a sympy expression"

        self._expression = expr

    @property
    def sub_properties(self):
        # Used to serialize important info
        return None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        # Must transform self.expression into a callable if it doesn't exist, using .sympy_parser.expr_to_np for example
        # then use the arguments to compute it
        pass


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

    def __call__(self, y_prime: np.ndarray, y: np.ndarray, x: np.ndarray, scalars: list,
                 with_weight=True):
        r"""
        :param f: A callable representing Y.
        :param f_prime: A callable representing Y_prime.
        :param return_abs: If True, returns the absolute value of the evaluation of the boundary function.
        :param with_weight: If True, returns the value of the evaluation of the boundary function times the weight.
        :return: The value of the boundary function at the boundary point(s).
         of f and f_prime at these points
        """
        if self._func is None:
            try:
                self.create_func(y_prime.shape[1], len(scalars))
            except IndexError:  # y_prime has only 1 dimension
                self.create_func(1, len(scalars))

        y_prime = y_prime.view(CallableArray)
        y_prime.X = x
        y = y.view(CallableArray)
        y.X = x
        weight = (self.weight if with_weight else 1)
        val = self._func(y_prime, y, x, scalars)
        # TODO: Remove this computation part from the solver
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
        used = "u_prime" if act_on_derivative else "u"
        if not isinstance(values, list):
            # Single value, suppose n_out = 1
            expr = f"{used}({point}) - {values}"
        else:
            expr = "["
            expr += ",".join((f"{used}_{i}({point}) - {value}" for i, value in enumerate(values) if value is not None))
            expr += "]"

        super().__init__(expr, weight=weight)


class LinearEquation(DifferentialEquation):
    r"""
    Given a (symbolic) array A, linear equation of the form u' = A u.
    """

    def __init__(self, A, n_eq: Union[int, list] = None, n_input: Union[int, list] = None, weight: Union[float, int] = 1):
        r"""
        :param A: a scalar or a matrix. Can be partly composed of sympy Symbols.
         If it is a scalar, it is converted to A * np.eye(n_eq | len(n_eq))
        :param n_eq: The number of equations.
         Can be a list to specify on which u_prime the matrix act (can have repeated / unordered indexes)
         If A is a scalar and n_eq == 1, it supposes there is exactly one equation in total
          (use [0] if there is more equations). default: A.shape[0]
        :param n_input: The number of inputs to use. Can also be a list to specify which inputs the matrix takes.
         default: A.shape[1]
        :param weight: The multiplier that will be given to this equation.
         Higher values mean this equation will be more taken into account.
        """
        if isinstance(A, (int, float, sp.Expr)):
            assert n_eq is not None, "The size of the system must be defined when using scalar"
            if n_eq == 1:
                super().__init__(f"- u_prime + {A} * u", weight=weight)
                return

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
            expr = f"- u_prime_{n_eq[i]}"
            for j in range(len(n_input)):
                expr += f" + {A[i, j]} * u_{n_input[j]}"
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

    def __call__(self, y_prime: np.ndarray, y: np.ndarray, x: np.ndarray, scalars: list):
        return np.sum([equation(y_prime, y, x, scalars) for equation in self.des])


class ProcessX(Expression):
    r"""
    :param expression: A Sympy expression or its str representation
    """

    def create_func(self):
        self._func = expr_to_np(self.expression, ["x"])

    def __call__(self, x):
        if self._func is None:
            self.create_func()
        return self._func(x)
