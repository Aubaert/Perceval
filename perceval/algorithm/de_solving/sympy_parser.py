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
from typing import List, Union

import sympy as sp
from scipy import interpolate
from scipy.integrate import trapezoid
import numpy as np


def _integral_as_trapezoid(expr, lims):
    """
    Function to be used by the parser to allow the evaluation of the Integral sympy expression.

    :param expr: expected array resulting of the internal expression of the Integral
    :param lims: Integral can be defined as Integral(expr, x) in such case lims is the array x,
     or as Integral(expr, (x, lim_inf, lim_sup)) in such case lims is a tuple of size 3
    """

    if isinstance(lims, tuple):
        var, a, b = lims
    else:
        var, a, b = lims, min(lims), max(lims)
    try:
        select = np.tile(var < a, expr.shape[1])
    except IndexError:  # expr is a one dimensional array
        select = var < a
    expr[select] = 0
    try:
        select = np.tile(var > b, expr.shape[1])
    except IndexError:
        select = var > b
    expr[select] = 0
    return trapezoid(expr, var, axis=0)


def expr_to_np(expr: sp.Expr, inputs: List[str]):
    # Equivelent to sp.lambdify, but integrate more functions
    return sp.lambdify(inputs, expr, modules=["numpy", {"Integral": _integral_as_trapezoid}])


def lambdify_diff_eq(expr: Union[sp.Expr, List[sp.Expr]], n_out: int, n_scalars: int):
    r"""
    Generate a lambda function from a DifferentialEquation expression.
    """
    # Generate the input names according to the number of elements. Finally, we have 2 * n_out + 1 + n_scalars inputs
    inputs = ["u_prime"] if n_out == 1 else [f"u_prime_{i}" for i in range(n_out)]
    inputs += ["u"] if n_out == 1 else [f"u_{i}" for i in range(n_out)]
    inputs += ["x"]
    inputs += ["scalar"] if n_scalars == 1 else [f"scalar_{i}" for i in range(n_scalars)]

    # allow for evaluating several expressions
    if isinstance(expr, list):
        funcs = [expr_to_np(ss_expr, inputs) for ss_expr in expr]

        def expr_list(y_prime, y, x, scalars):
            # prepare the inputs so that there is the good number of inputs
            y_prime = to_list(y_prime, n_out)
            y = to_list(y, n_out)

            res = [funcs[i](*y_prime, *y, x, *scalars) for i in range(len(funcs))]
            return res

        return expr_list

    f = expr_to_np(expr, inputs)

    def expr_fn(y_prime, y, x, scalars):
        y_prime = to_list(y_prime, n_out)
        y = to_list(y, n_out)

        return f(*y_prime, *y, x, *scalars)

    return expr_fn


def to_list(y, n_out):
    return [y] if n_out == 1 else [y[:, i] for i in range(n_out)]


class CallableArray(np.ndarray):
    r"""
    Class to be used to allow evaluation in expressions such as u(1) starting from an array U and its point positions X.
    Usage:
        U = U.view(CallableArray)

        U.X = X

        U(1) --> Interpolation of U given X at point 1
    """
    X = None
    f = None

    def __call__(self, x):
        if self.f is None:
            self.f = interpolate.interp1d(self.X, self, axis=0, kind="cubic", assume_sorted=True)
        return self.f(x)

    def __getitem__(self, item):
        sliced = super().__getitem__(item)
        sliced.X = self.X
        return sliced
