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

from perceval.components import Experiment
from perceval.utils.logging import get_logger, channel

from .abstract_computer import AbstractComputer
from .computation import Computation
from .computation_iterator import ComputationIterator
from .execution import Execution


class ExecutionFactory:

    def __init__(self, computer: AbstractComputer, experiment: Experiment):
        self.computer = computer
        self.experiment = experiment
        self._iterator = []

    def add_iteration(self, **kwargs):
        """
        Add a single iteration to future jobs.

        :param kwargs: List of accepted keywords:

           - circuit_params: dict containing pairs (parameter_name: str - value : number)
           - input_state: BasicState
           - min_detected_photons: int
           - max_samples: int
           - max_shots: int
           - noise: NoiseModel
           - postselect: PostSelect
        """
        get_logger().info("Add 1 iteration to ExecutionFactory", channel.general)
        self._iterator.append(kwargs)

    def add_iteration_list(self, iterations: list[dict]):
        """
        Add multiple iterations to future jobs.
        """
        get_logger().info(f"Add {len(iterations)} iterations to ExecutionFactory", channel.general)
        for iter_params in iterations:
            self._iterator.append(iter_params)

    def clear_iterations(self):
        """
        Clear all prepared iterations.
        """
        get_logger().info(f"Clear all iterations in ExecutionFactory", channel.general)
        self._iterator.clear()

    @property
    def n_iterations(self):
        return len(self._iterator)

    def build_computation(self, name: str) -> Computation | ComputationIterator:
        # Note: this can also be used to create a computation that will last for more than
        comp = Computation(self.computer.get_command(name), self.experiment)
        if self.n_iterations > 0:
            comp = ComputationIterator(comp)
            for it in self._iterator:
                comp.add_iteration(**it)

        return comp

    def build_execution(self, computation: Computation | ComputationIterator) -> Execution:
        return Execution(computation, self.computer)

    @property
    def probs(self):
        return self.build_execution(self.build_computation('probs'))

    @property
    def samples(self):
        return self.build_execution(self.build_computation('samples'))

    @property
    def sample_count(self):
        return self.build_execution(self.build_computation('sample_count'))

    def __getattr__(self, item):  # Allows writing things like factory.custom_command
        return self.build_execution(self.build_computation(item))
