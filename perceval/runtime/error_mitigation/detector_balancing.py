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
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from copy import deepcopy, copy
import math
import warnings

from .abstract_mitigation import AbstractMitigation
from ..computation import Computation

from perceval.utils import NoiseModel
from perceval.utils.constants import KEY_RESULTS

class DetectorBalancing(AbstractMitigation):

    APPLY_MIN_PHOTONS = True
    APPLY_LOGICAL_SELECTION = True

    def __init__(self):
        """
        A mitigation process that adjusts the probabilities of each output state based on the output
        loss and number of photons in each mode.
        """

    @staticmethod
    def _validate_ratios(ratios: list[float] | None, m: int) -> list[float]:
        if ratios is None:
            warnings.warn(
                "DetectorBalancing was created without specifying loss ratios: "
                "defaulting to 1.",
                RuntimeWarning,
            )
            ratios = [1] * m

        if len(ratios) != m:
            raise ValueError(
                "Size of detector loss ratios must match raw processor mode count "
                f"(expected {m}, got {len(ratios)})."
            )

        def valid(v):
            res = math.isfinite(v) and v > 0.
            if not res and not valid.warned:
                warnings.warn(
                    "Calibrated detector loss ratios contain non-positive or non-finite "
                    "values. Replacing invalid entries with 1.0.",
                    RuntimeWarning,
                )
                valid.warned = True
            return res
        valid.warned = False

        return [v if valid(v) else 1. for v in ratios]

    def extend_computation(self, computation: Computation, noise: NoiseModel) -> list[Computation]:
        comp = deepcopy(computation)
        comp.command = "probs"
        return [comp]

    def _parse_results(self, computation: Computation, results: list[dict], noise: NoiseModel) -> dict:
        ratios = self._validate_ratios(noise.loss_ratios, computation.experiment.m)

        def _balance(k, v):
            return math.prod([(1 / ratios[k.photon2mode(i)]) for i in range(k.n)], start = v)

        res = copy(results[0])  # We are going to modify this to keep custom fields as much as we can
        assert len(ratios) == computation.experiment.m, "Loss ratios do not match the distribution lengths."

        for k, v in res[KEY_RESULTS].items():
            res[KEY_RESULTS][k] = _balance(k, v)
        return res
