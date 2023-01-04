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
from typing import Union, List, Tuple, Callable

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from perceval.runtime import RemoteProcessor
from perceval.runtime.remote_job import RemoteJob
from .de_solving.differential_equation import DECollection, DifferentialEquation
from perceval.components.abstract_processor import ProcessorType
from perceval.algorithm.abstract_algorithm import AAlgorithm

solving_fn_name = "DESolver:solve"


def convert_to_numpy(result: dict):
    for key in result:
        if isinstance(result[key], dict):
            result[key] = convert_to_numpy(result[key])
        elif isinstance(result[key], list):
            result[key] = np.array(result[key])
    return result


class DESolver(AAlgorithm):
    r"""
    Holds the methods to solve a given differential equation represented by a DECollection,
     as well as post-optimisation methods and display methods.
    """

    def __init__(self, X, de_collection: Union[DECollection, DifferentialEquation], processor: RemoteProcessor,
                 alpha_noise: Union[float, List[float]] = 0,
                 nb_out: int = 1, nb_scalar: int = 0,
                 bounds: Union[List[Tuple[float, float]], Tuple[float, float]] = (-10, 10),
                 scalar_bound: Union[List[Tuple[float, float]], Tuple[float, float]] = (-10, 10),
                 force_scalar_bounds=False):
        r"""
        :param X: the grid on which the solving will take place (e.g. np.linspace(0, 1, 50))
        :param de_collection: The collection with all boundary conditions and domain differential equations.
        :param processor: A RemoteProcessor to perform computation on.
        :param alpha_noise: The penalty coefficient in loss for probability weights.
        :param nb_out: The number of dimensions of the output.
        :param nb_scalar: The number of unknown scalars.
        :param bounds: The approximate boundaries that the solution can cover. Can be specified for each dimension.
         Only concerns the starting point, not the results.
        :param scalar_bound: The approximate boundaries that the scalars can cover. Can be specified for each scalar.
         Only concerns the starting point, not the results, unless force_scalar_bound is True.
        """
        self.initiated = False
        self._job_id = None
        self._scalars = []
        self.nb_scalar = nb_scalar
        self.scalar_bounds = scalar_bound
        self.force_scalar_bounds = force_scalar_bounds
        self.scalar_legend = list(range(nb_scalar))
        self._max_y = None
        self._min_y = None
        self._Y = None
        self._sigma_Y = None
        self._lines = None
        self._sigma_lines = []
        self.plot_opacity = 0.2
        self._ax = None
        self._fig = None
        self.display_curve = None
        self.post_optimisation_result = None
        self.results: List[dict] = []
        self.pbar = None
        super().__init__(processor)
        self.alpha_noise = alpha_noise
        if isinstance(de_collection, DifferentialEquation):
            de_collection = DECollection(de_collection)
        self._de_collection = de_collection
        self.nb_out = nb_out
        self.legend = list(range(nb_out))
        self.X = X
        self.analytical_solution = None  # Can be used for display
        self.bounds = bounds
        self.parameters = self.default_parameters
        self.default_job_name = None

    @property
    def de_collection(self):
        return self._de_collection

    @de_collection.setter
    def de_collection(self, de_collection):
        assert isinstance(de_collection, DECollection), "de_collection must be a DECollection"
        assert len(de_collection.des), "There must be at least one equation to solve"
        self._de_collection = de_collection

    @property
    def processor(self):
        return self._processor

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._nb_grid = len(X)
        X.sort()
        if self.initiated:
            assert X[0] == self.x_min and X[-1] == self.x_max, \
                "range of the grid cannot be changed after initialisation"
        self._range_min = X[0]
        self._range_max = X[-1]
        self._X = X

    @property
    def nb_grid(self):
        return self._nb_grid

    @property
    def x_min(self):
        return self._range_min

    @property
    def x_max(self):
        return self._range_max

    @property
    def nb_out(self):
        return self._nb_out

    @nb_out.setter
    def nb_out(self, val):
        assert isinstance(val, int), f"wrong type for nb_out, got {type(val)}"
        self._nb_out = val

    @property
    def alpha_noise(self):
        return self._alpha_noise

    @alpha_noise.setter
    def alpha_noise(self, val: Union[list, tuple, int, float]):
        r"""
        A value or a list of values to penalise the coefficients, using an alpha * norm2(coefficients) error.
         If a list is given, the value will be used for the corresponding output dimension.
         Useful when using sampling.
        """
        if isinstance(val, (list, tuple)):
            for val2 in val:
                assert val2 >= 0, "penalisation of coefficients must be a positive number"
        else:
            assert val >= 0, "penalisation of coefficients must be a positive number"
        self._alpha_noise = val

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, val: Union[List[Tuple[float]], Tuple[float]]):
        r"""
        The approximate value of the minimum and maximum of the targeted solution.
         If given a list of tuples, the bound will be applied for each dimension.
        """
        assert len(val), "no value given"
        for val2 in val:
            self.single_bound = False
            if isinstance(val2, (list, tuple)):
                assert len(val2) == 2, "Two values must be given for each dimension"
                for val_num in val2:
                    assert isinstance(val_num, (float, int)), f"Given value must be a real number, got {type(val_num)}"
            else:
                self.single_bound = True
                break
        if self.single_bound:
            assert len(val) == 2, "Two values must be given for each dimension"
            for val_num in val:
                assert isinstance(val_num, (float, int)), f"Given value must be a real number, got {type(val_num)}"
        self._bounds = val

    @property
    def scalar_bounds(self):
        return self._scalar_bounds

    @scalar_bounds.setter
    def scalar_bounds(self, val: Union[List[Tuple[float]], Tuple[float]]):
        r"""
        The approximate value of the minimum and maximum of the targeted solution.
         If given a list of tuples, the bound will be applied for each dimension.
        """
        assert len(val), "no value given"
        for val2 in val:
            self.scalar_single_bound = False
            if isinstance(val2, (list, tuple)):
                assert len(val2) == 2, "Two values must be given for each dimension"
                for val_num in val2:
                    assert isinstance(val_num, (float, int)), f"Given value must be a real number, got {type(val_num)}"
            else:
                self.scalar_single_bound = True
                break
        if self.scalar_single_bound:
            assert len(val) == 2, "Two values must be given for each dimension"
            for val_num in val:
                assert isinstance(val_num, (float, int)), f"Given value must be a real number, got {type(val_num)}"
        self._scalar_bounds = val

    @property
    def default_parameters(self):
        return {
            "nf_order": 2,
            "nf_frequency": 0.15,
            "resampling_it_number": 0,
            "gtol": 1e-5,
        }

    def compute_curve(self, unitary_parameters, lambda_random):
        assert self.processor.type == ProcessorType.SIMULATOR, "Impossible to recompute when using a QPU"
        job_context = {"result_mapping": ['perceval.utils', "dict_list_to_numpy"]}

        data = self.processor.prepare_job_payload("DESolver:compute_curve")
        self.update_payload(data, unitary_parameters=unitary_parameters, coefficients=lambda_random)
        job_name = self.default_job_name if self.default_job_name is not None else "DESolver:compute_curve"

        results = (RemoteJob(data,
                             self.processor.get_rpc_handler(),
                             job_name,
                             job_context=job_context)
                   .execute_sync()["results"])

        self._sigma_Y = results["sigma_Y"]
        return results["function"]

    def update_payload(self, payload, **kwargs):
        payload["payload"].update({
            "grid": self.X.tolist() if isinstance(self.X, np.ndarray) else self.X,
            "equations": self.de_collection,
            "equation_parameters": {
                "nb_out": self.nb_out,
                "bounds": self.bounds,
                "nb_scalar": self.nb_scalar,
                "scalar_bound": self.scalar_bounds,
                "force_scalar_bounds": self.force_scalar_bounds,
                "alpha_noise": self.alpha_noise,
            },
            "solver_parameters": self.parameters,
            **kwargs
        })

        return payload

    @property
    def solve(self) -> RemoteJob:
        """
        Performs an optimisation to solve the differential equations.
        It is highly advised to append the result in self.results to use it in post-optimisation.
        """
        self._job_id = None  # Used to reset plot
        self.initiated = True

        job_context = {"result_mapping": ['perceval.utils', "dict_list_to_numpy"]}

        data = self.processor.prepare_job_payload(solving_fn_name)
        self.update_payload(data)
        job_name = self.default_job_name if self.default_job_name is not None else solving_fn_name

        job = RemoteJob(data, self.processor.get_rpc_handler(), job_name, job_context=job_context)
        return job

    def display_job(self, job: RemoteJob, display_curves=False):
        """
        Call self.pbar(intermediate_result), and update curves if display_curves is True.
        Does nothing if pbar function has not been set or display_curves is False.
        Does not alter the async behaviour of the script.
        :param job: A job that has been created through the run function.
        :param display_curves: If True, it will update or create curves of the intermediate results
        """
        assert job._request_data['payload']["command"] == solving_fn_name, "given job is not a solve job"

        try:
            res = job.get_results()  # TODO: add possibility to retrieve intermediate results in Jobs
        except RuntimeError:
            return  # Maybe something else would be clever

        if self.pbar is not None:
            self.pbar(res)

        res = res["results"]

        if display_curves:
            if self._job_id != job.id:
                self._job_id = job.id
                self._fig = plt.figure()
                self._ax = self._fig.add_subplot(111)
                if self.analytical_solution is not None and self._min_y is None:
                    y = self.analytical_solution(self.X)
                    self._min_y = np.min(y)
                    self._max_y = np.max(y)

            Y = res["function"]
            sigma_Y = res["sigma_Y"]
            if self._lines is None:
                self._sigma_lines = []
                self._lines = self._ax.plot(self.X, Y, label=f"solution at iteration {res['tot_it_count']}")
                for i, line in enumerate(self._lines):
                    self._sigma_lines.append(self._ax.fill_between(self.X, Y[:, i] - sigma_Y[:, i],
                                                                   Y[:, i] + sigma_Y[:, i],
                                                                   color=line.get_color(),
                                                                   alpha=self.plot_opacity))
                self.plot(best_solution=False, solution_numbers=[])  # display analytical solutions
            else:
                for i, line in enumerate(self._lines):
                    line.set_ydata(Y[:, i])
                    line.set_label(f"solution {self.legend[i]} at iteration {res['tot_it_count']}")
                    # Invisible, just to compute things
                    dummy = self._ax.fill_between(self.X, Y[:, i] - sigma_Y[:, i],
                                                  Y[:, i] + sigma_Y[:, i], alpha=0)
                    dp = dummy.get_paths()[0]
                    dummy.remove()
                    self._sigma_lines[i].set_paths([dp.vertices])
            min_y = np.min(Y)
            if self._min_y is not None:
                min_y = min(min_y, self._min_y)
            max_y = np.max(Y)
            if self._max_y is not None:
                max_y = max(max_y, self._max_y)
            self._ax.set_ylim(min_y - 0.2 * abs(min_y), max_y + 0.2 * abs(max_y))
            plt.legend()
            if self.nb_scalar:
                title = ""
                for i in range(self.nb_scalar):
                    title += f"{self.scalar_legend[i]} = {res['scalars'][i]}, "
                self._ax.set_title(title[:-2])
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def loss_from_curve(self, Y):
        # Assumes X, Y are np.arrays
        loss = self.de_collection(Y, self.X, self._scalars)

        return loss

    # Post optimisation
    def retrieve_solution(self, i: int = -1, recompute=False):
        """
        :param i: The number of the solution. Default to the last computed solution.
        :param recompute: If True, the curve will be computed again. May make the result vary a bit with parameters
         involving random probabilities such as samples.
        Return the solution array.
        """
        assert len(self.results), "missing results"

        if np.any(self.X != self.results[i]["X"]) or recompute:
            lambda_random = self.results[i]["weights"]
            unitary_parameters = self.results[i]["unitary_parameters"]

            Y = self.compute_curve(unitary_parameters, lambda_random)
        else:
            Y = self.results[i]["function"]
            self._sigma_Y = self.results[i]["results"]["sigma_Y"]

        return Y

    def _linear_combination_optimisation(self, coeffs, base_functions, scalars):
        Y = np.array(coeffs) @ np.swapaxes(np.array(base_functions), 0, 1)
        if self.nb_scalar:
            self._scalars = coeffs @ scalars
        return self.loss_from_curve(Y)

    def post_optimisation(self, loss_max: float = None, optimize=True, recompute=False, solution_numbers: list = None,
                          post_selection_fn: Callable = None):
        r"""
        :param loss_max: The maximum loss to consider a solution into account. None to use all solutions.
        :param optimize: If False, no optimisation will be performed, but only the initial guess will be given.
        :param recompute: True to recompute all solution curves before optimising.
        :param solution_numbers: A list of indexes to select some solutions.
        :param post_selection_fn: A callable that, given a solution from self.results, returns True
         if the solution must be kept.
        Performs an optimisation finding the best linear combination of the solutions.
         Store the result in self.post_optimisation_result.
        """
        tot_loss = 0
        real_loss_max = 0

        kept_losses = []
        kept_functions = []
        kept_sigma = []
        kept_scalars = []
        kept_index = []

        if solution_numbers is None:
            solution_numbers = range(len(self.results))
        for i in solution_numbers:
            solution = self.results[i]
            cur_loss = solution["final_loss"]
            if (loss_max is None or cur_loss < loss_max) and (post_selection_fn is None or post_selection_fn(solution)):
                tot_loss += cur_loss
                kept_losses.append(cur_loss)
                kept_functions.append(self.retrieve_solution(i, recompute))
                kept_sigma.append(self._sigma_Y.copy())
                if self.nb_scalar:
                    kept_scalars.append(solution["scalars"])
                kept_index.append(i)
                if loss_max is None and cur_loss > real_loss_max:
                    real_loss_max = cur_loss

        real_loss_max = loss_max or real_loss_max

        # Functions with the lesser loss are more represented (may converge faster)
        coefficients = np.array([(real_loss_max - loss) / (len(kept_losses) * real_loss_max - tot_loss)
                                 for loss in kept_losses]) if len(kept_losses) > 1 else [1]
        # coefficients = [1 / len(kept_losses)] * len(kept_losses)

        self.post_optimisation_result = {
            "X": self.X.copy(),
            "indexes": kept_index
        }

        if optimize:
            res = minimize(self._linear_combination_optimisation, np.array(coefficients),
                           args=(kept_functions, kept_scalars), method='BFGS', jac="3-point",
                           options={'gtol': 1E-5, 'finite_diff_rel_step': 1e-2})

            self.post_optimisation_result["coefficients"] = res.x
            self.post_optimisation_result["function"] = res.x @ np.swapaxes(np.array(kept_functions), 0, 1)
            self.post_optimisation_result["sigma_Y"] = np.sqrt(res.x ** 2 @ np.swapaxes(np.array(kept_sigma), 0, 1) ** 2)
            if self.nb_scalar:
                self.post_optimisation_result["scalars"] = res.x @ kept_scalars
            self.post_optimisation_result["final_loss"] = res.fun

        else:
            Y = coefficients @ np.swapaxes(np.array(kept_functions), 0, 1),
            self.post_optimisation_result["coefficients"] = coefficients
            self.post_optimisation_result["final_loss"] = self.loss_from_curve(Y)
            if self.nb_scalar:
                self.post_optimisation_result["scalars"] = coefficients @ kept_scalars
            self.post_optimisation_result["function"] = Y
            self.post_optimisation_result["sigma_Y"] = np.sqrt(coefficients ** 2 @
                                                               np.swapaxes(np.array(kept_sigma), 0, 1) ** 2)

        return self.post_optimisation_result["function"]

    def compute_post_optimised_solution(self):
        """
        :returns: the curve of the already computed optimised solution.
        """
        assert self.post_optimisation_result is not None, "No optimisation has been computed"
        assert np.all(self.X == self.post_optimisation_result["X"])
        kept_functions = []
        for i in self.post_optimisation_result["indexes"]:
            kept_functions.append(np.array(self.retrieve_solution(i)))

        return self.post_optimisation_result["coefficients"] @ np.swapaxes(np.array(kept_functions), 0, 1)

    def plot(self, best_solution=True, post_optimised=False, plot_solutions=False, loss_max=None, recompute=False,
             solution_numbers: list = None, curve_indexes: list = None, post_selection_fn: Callable = None,
             with_analytical=True, plot_error=True, **kwargs):
        r"""
        :param best_solution: If True, the best solution found will be plotted.
         Will ignore loss_max but not post_selection_fn.
        :param post_optimised: if True, will plot the post-optimised solution, and will compute it if none exists.
        :param plot_solutions: if True, will plot all the solutions obtained through th run method.
        :param loss_max: Only the solutions having a loss inferior to loss_max will be displayed
         and used to compute the post-optimised solution if none exists.
        :param recompute: if True, solutions will be recomputed before displaying.
        :param solution_numbers: A list of indexes of the solutions that will possibly be plotted
         (given they match the other conditions).
        :param curve_indexes: A list of indexes of the output that will be plotted. default to all.
        :param post_selection_fn: A callable that, given a solution from self.results returns True
         if the solution must be kept.
        :param with_analytical: If True and an analytical solution has been set, display it.
        :param plot_error: If True, error bar will be displayed around the solutions.

        Plot the results and the analytical solution if one is provided. Uses the legend to name the curves.
        If there are scalars, their value for the post-optimised solution will be set as title.
        """
        if solution_numbers is None:
            solution_numbers = range(len(self.results))
        if curve_indexes is None:
            curve_indexes = range(self.nb_out)
        if best_solution and not plot_solutions:
            min_sol = min([(i, sol) for i, sol in enumerate(self.results)
                           if post_selection_fn is None or post_selection_fn(sol)],
                          key=lambda sol: sol[1]["final_loss"])
            self.plot(best_solution=False, plot_solutions=True, loss_max=loss_max, recompute=recompute,
                      solution_numbers=[min_sol[0]], with_analytical=False, curve_indexes=curve_indexes)
            if self.nb_scalar:
                title = ""
                for i in range(self.nb_scalar):
                    title += f"{self.scalar_legend[i]} = {min_sol[1]['scalars'][i]}, "
                plt.title(title[:-2])

        if plot_solutions:
            for i in solution_numbers:
                solution = self.results[i]
                if (loss_max is None or solution["final_loss"] < loss_max) \
                        and (post_selection_fn is None or post_selection_fn(solution)):
                    Y = self.retrieve_solution(i)
                    for j in curve_indexes:
                        line, = plt.plot(self.X, Y[:, j], label=f"Solution {i} {self.legend[j]}", **kwargs)
                        if plot_error:
                            sigma_Y = self._sigma_Y
                            plt.fill_between(self.X, Y[:, j] - sigma_Y[:, j], Y[:, j] + sigma_Y[:, j],
                                             color=line.get_color(), alpha=self.plot_opacity)

        if post_optimised:
            if self.post_optimisation_result is None or \
                    np.any(self.X != self.post_optimisation_result["X"]) or recompute:
                self.post_optimisation(loss_max, recompute=recompute, solution_numbers=solution_numbers,
                                       post_selection_fn=post_selection_fn)
            Y = self.post_optimisation_result["function"]
            for j in curve_indexes:
                line, = plt.plot(self.X, Y[:, j], ".", label=f"Post optimised solution {self.legend[j]}", **kwargs)
                if plot_error:
                    sigma_Y = self.post_optimisation_result["sigma_Y"]
                    plt.fill_between(self.X, Y[:, j] - sigma_Y[:, j], Y[:, j] + sigma_Y[:, j],
                                     color=line.get_color(), alpha=self.plot_opacity)
            if self.nb_scalar:
                title = ""
                for i in range(self.nb_scalar):
                    title += f"{self.scalar_legend[i]} = {self.post_optimisation_result['scalars'][i]}, "
                plt.title(title[:-2])

        if self.analytical_solution is not None and with_analytical:
            Y = self.analytical_solution(self.X)
            for j in curve_indexes:
                plt.plot(self.X, Y[:, j], '--', label=f'Analytical solution {self.legend[j]}')
