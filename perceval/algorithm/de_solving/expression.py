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


class Expression(ABC):

    handle_equation_list = False

    def __init__(self, expression: Union[sp.Expr, str, list]):
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
