import numpy as np
from sobol_seq import i4_sobol_generate
from emukit.core.initial_designs.base import ModelFreeDesignBase
from emukit.core import ParameterSpace

class SobolDesign(ModelFreeDesignBase):
    """
    Sobol experiment design.
    Based on sobol_seq implementation. For further reference see https://github.com/naught101/sobol_seq
    """

    def __init__(self, parameter_space: ParameterSpace) -> None:
        """
        param parameter_space: The parameter space to generate design for.
        """
        super(SobolDesign, self).__init__(parameter_space)

    def get_samples(self, point_count: int) -> np.ndarray:
        """
        Generates requested amount of points.
        :param point_count: Number of points required.
        :return: A numpy array of generated samples, shape (point_count x space_dim)
        """
        bounds = self.parameter_space.get_bounds()
        lower_bound = np.asarray(bounds)[:, 0].reshape(1, len(bounds))
        upper_bound = np.asarray(bounds)[:, 1].reshape(1, len(bounds))
        diff = upper_bound - lower_bound

        X_design = np.dot(i4_sobol_generate(len(bounds), point_count), np.diag(diff[0, :])) + lower_bound

        samples = self.parameter_space.round(X_design)

        return samples
    
    
    
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def plot(x0, x1, x_plot, val, title):
    res = 200
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(x0, x1, c=list(range(0, x0.shape[0])), linewidths=6, cmap='Reds', zorder=100)
    plt.contourf(x_plot[:, 0].reshape(res, res), x_plot[:, 1].reshape(res, res), val.reshape((res, res)), levels=500)
    plt.colorbar()
    plt.xlabel(r"$x0$")
    plt.ylabel(r"$x1$")
    fig.suptitle(title)
    plt.show()

def plot_f(x0, x1, x_plot, mu_plot, var_plot):
    plot(x0, x1, x_plot, mu_plot, 'mean')
    plot(x0, x1, x_plot, var_plot, 'var')
    
def plot_acquisition(x0, x1, x_plot, ei_plot, pof_plot, composite_plot):
    ei_plot = (ei_plot - np.min(ei_plot)) / (np.max(ei_plot) - np.min(ei_plot))
    pof_plot = (pof_plot - np.min(pof_plot)) / (np.max(pof_plot) - np.min(pof_plot))
    composite_plot = (composite_plot - np.min(composite_plot)) / (np.max(composite_plot) - np.min(composite_plot))
    plot(x0, x1, x_plot, ei_plot, 'EI')
    plot(x0, x1, x_plot, pof_plot, 'PoF')
    plot(x0, x1, x_plot, composite_plot, 'Composite')
    
def plot_3d(x0, x1, x2):
    plt.style.use('dark_background')
    fig = pyplot.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter(x0, x1, x2, c=list(range(0, x0.shape[0])), linewidths=6, cmap='Reds', zorder=100)
    ax.set_xlabel(r"$x0$")
    ax.set_ylabel(r"$x1$")
    ax.set_zlabel(r"$x2$")
    pyplot.show()
    
# def plot_f_3d(x0, x1, x2, x_plot0, x_plot1, x_plot2, mu_plot):
#     fig = pyplot.figure(figsize=(12, 12))
#     ax = Axes3D(fig)
#     ax.scatter(x0, x1, x2, c=list(range(0, x0.shape[0])), linewidths=6, cmap='Reds', zorder=100)
#     ax.scatter(x_plot0, x_plot1, x_plot2, c=mu_plot, marker='.', zorder=100)
#     ax.set_xlabel(r"$x0$")
#     ax.set_ylabel(r"$x1$")
#     ax.set_zlabel(r"$x2$")
#     pyplot.show()
    
def plot_pof_3d(x0, x1, x2, x_plot_3d, pof_plot_3d):
    plt.style.use('dark_background')
    x_plot_3d_filtered = []
    pof_plot_3d_filtered = []
    for idx, pof in enumerate(pof_plot_3d):
        if pof > 0.9:
            x_plot_3d_filtered.append(x_plot_3d[idx])
            pof_plot_3d_filtered.append(pof)
    x_plot_3d_filtered = np.array(x_plot_3d_filtered)
    pof_plot_3d_filtered = np.array(pof_plot_3d_filtered)
    
    fig = pyplot.figure(figsize=(12, 12))
    ax = Axes3D(fig)
    ax.scatter(x0, x1, x2, c=list(range(0, x0.shape[0])), linewidths=6, cmap='Reds', zorder=100)
    print (x_plot_3d_filtered.shape)
    ax.scatter(x_plot_3d_filtered[:,0].squeeze(), x_plot_3d_filtered[:,1].squeeze(), x_plot_3d_filtered[:,2].squeeze(), 
               c=pof_plot_3d_filtered, marker='.', linewidths=2, zorder=100)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")
    pyplot.show()
    
    
    
from typing import Tuple, Union
import scipy.stats
import numpy as np
import itertools
from emukit.core.interfaces import IModel, IDifferentiable, IJointlyDifferentiable
from emukit.core.acquisition import Acquisition
from emukit.bayesian_optimization.acquisitions.expected_improvement import get_standard_normal_pdf_cdf

class MyExpectedImprovement(Acquisition):
    def __init__(self, model: Union[IModel, IDifferentiable], 
                 constraint_model: Union[IModel, IDifferentiable], jitter: float=0.0)-> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        the feasible region.
        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.con_model = constraint_model
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        mean += self.jitter

        y_minimum = get_constraint_min(self.model.Y.copy(), self.con_model.Y.copy())
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)
        improvement = standard_deviation * (u * cdf + pdf)

        return improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.
        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = get_constraint_min(self.model.Y.copy(), self.con_model.Y.copy())

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u, pdf, cdf = get_standard_normal_pdf_cdf(y_minimum, mean, standard_deviation)

        improvement = standard_deviation * (u * cdf + pdf)
        dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx

        return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)


def get_constraint_min(Y, Yc):
    """Get minimum in the feasible region"""
    y_min = float('inf')
    for idx, yc in enumerate(Yc):
        if yc <= 0 and Y[idx] < y_min:
            y_min = Y[idx]
    # Make sure y_min is updated
    assert y_min != float('inf')
    return y_min







from typing import Union, Callable, Tuple

import numpy as np

from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel, IPriorHyperparameters


class MyIntegratedHyperParameterAcquisition(Acquisition):
    """
    This acquisition class provides functionality for integrating any acquisition function over model hyper-parameters
    """
    def __init__(self, model: Union[IModel, IPriorHyperparameters], acquisition_generator: Callable, constraint_model: Union[IModel,IPriorHyperparameters]=None, n_samples: int=10, n_burnin: int=100, subsample_interval: int=10, step_size: float=1e-1, leapfrog_steps: int=20):
        """
        :param model: An emukit model that implements IPriorHyperparameters
        :param acquisition_generator: Function that returns acquisition object when given the model as the only argument
        :param n_samples: Number of hyper-parameter samples
        :param n_burnin: Number of initial samples not used.
        :param subsample_interval: Interval of subsampling from HMC samples.
        :param step_size: Size of the gradient steps in the HMC sampler.
        :param leapfrog_steps: Number of gradient steps before each Metropolis Hasting step.
        """
        self.model = model
        self.constraint_model = constraint_model
        self.acquisition_generator = acquisition_generator
        self.n_samples = n_samples
        self.samples = self.model.generate_hyperparameters_samples(n_samples, n_burnin,
                                                                   subsample_interval, step_size, leapfrog_steps)
        if constraint_model:
            acquisition = self.acquisition_generator(model, constraint_model)
        else:
            acquisition = self.acquisition_generator(model)
        self._has_gradients = acquisition.has_gradients

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate acquisition by integrating over the hyper-parameters of the model
        :param x: locations where the evaluation is done.
        :return: Array with integrated acquisition value at all input locations
        """
        acquisition_value = 0
        for sample in self.samples:
            self.model.fix_model_hyperparameters(sample)
            if self.constraint_model:
                acquisition = self.acquisition_generator(self.model, self.constraint_model)
            else:
                acquisition = self.acquisition_generator(self.model)
            acquisition_value += acquisition.evaluate(x)

        return acquisition_value / self.n_samples

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the acquisition value and its derivative integrating over the hyper-parameters of the model
        :param x: locations where the evaluation with gradients is done.
        :return: tuple containing the integrated expected improvement at the points x and its gradient.
        """

        if x.ndim == 1:
            x = x[None, :]

        acquisition_value = 0
        d_acquisition_dx = 0

        for sample in self.samples:
            self.model.fix_model_hyperparameters(sample)
            if self.constraint_model:
                acquisition = self.acquisition_generator(self.model, self.constraint_model)
            else:
                acquisition = self.acquisition_generator(self.model)
            improvement_sample, d_improvement_dx_sample = acquisition.evaluate_with_gradients(x)
            acquisition_value += improvement_sample
            d_acquisition_dx += d_improvement_dx_sample

        return acquisition_value / self.n_samples, d_acquisition_dx / self.n_samples

    def update_parameters(self):
        self.samples = self.model.generate_hyperparameters_samples(self.n_samples)

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return self._has_gradients