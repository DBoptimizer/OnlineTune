import numpy as np
from onlinetune.safe.domain import ContinuousDomain
from onlinetune.safe.model import ModelMixin
from onlinetune.knobs import logger


class Algorithm():
    """
    Base class for algorithms.
    """

    def __init__(self, **experiment_info):
        self.experiment_info = experiment_info
        self._dtype_fields = []
        self.name = type(self).__name__

    def initialize(self, **kwargs):
        """
        Called to initialize the algorithm. Resets the algorithm, discards previous data.

        """
        self.domain = kwargs.get("domain")
        self.domain_all = kwargs.get("domain_all")
        self.x0 = kwargs.get("x0", None)
        self.initial_data = kwargs.get("initial_data", [])
        self._exit = False
        self.t = 0

        self.lower_bound_objective_value = kwargs.get("lower_bound_objective_value", None)
        self.num_constraints = kwargs.get("num_constraints", None)
        self.noise_obs_mode = kwargs.get("noise_obs_mode", None)

        self.__best_x = None
        self.__best_y = -10e10
        self.knobL = []

    def _next(self, context=None):

        raise NotImplementedError

    def next(self, context=None):
        """
        Called to get next evaluation point from the algorithm.
        """
        if context is None:
            # call without context (algorithm might not allow context argument)
            next_x = self._next()
        else:
            # call with context
            next_x = self._next(context)

        if isinstance(next_x, tuple):
            x = next_x[0]
            additional_data = next_x[1]
        else:
            x = next_x
            additional_data = {}
        additional_data['t'] = self.t
        self.t += 1

        # for continous domains, check if x is inside box
        if isinstance(self.domain_all, ContinuousDomain):
            if (x > self.domain_all.u).any() or (x < self.domain_all.l).any():
                # logger.warning(f'Point outside domain. Projecting back into box.\nx is {x}, with limits {self.domain.l}, {self.domain.u}')
                x = np.maximum(np.minimum(x, self.domain_all.u), self.domain_all.l)

        return x, additional_data

    def add_data(self, data):
        """
        Add observation data to the algorithm.
        """
        if data['y'] > self.__best_y:
            self.__best_y = data['y']
            self.__best_x = data['x']

        self.initial_data.append(data)

    @property
    def dtype(self):
        """
        Returns:
            Numpy dtype of additional data return with next().

        """
        return np.dtype(self._get_dtype_fields())

    def _get_dtype_fields(self):
        """
        Fields used to define ``self.dtype``.

        Returns:

        """
        fields = [("t", "i")]
        return fields

    def finalize(self):
        return {'initial_data' : self.initial_data,
                'best_x' : self.best_predicted()}

    @property
    def requires_x0(self):
        """
        If true, algorithm requires initial evaluation from environment.
        By default set to False.
        """
        return False

    @property
    def exit(self):
        return self._exit

    def best_predicted(self):
        """
        If implemented, this should returns a point in the domain, which is currently believed to be best
        Returns:

        """
        return self.__best_x


class AcquisitionAlgorithm(Algorithm):
    """
    Algorithm which is defined through an acquisition function.
    """

    def initialize(self, **kwargs):
        super(AcquisitionAlgorithm, self).initialize(**kwargs)
        self._evaluate_x0 = True

        #self.solver = self._get_solver(domain=self.domain)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['solver']
        return self_dict

    def acquisition(self, x):
        raise NotImplementedError

    def acquisition_init(self):
        pass

    def acquisition_grad(self, x):
        raise NotImplementedError

    def _next(self, context=None):
        if self._evaluate_x0:
            self._evaluate_x0 = False
            if self.x0 is None:
                logger.error("Cannot evaluate x0, no initial point given")
            else:
                logger.info(f"{self.name}: Choosing initial point.")
                return self.x0

        # for contextual bandits, if domain changes, adjust solver (for now, just a new instance)
        if not context is None and 'domain' in context:
            self.solver = self._get_solver(context['domain'])

        self.acquisition_init()

        if self.solver.requires_gradients:
            acq = self.acquisition_grad
        else:
            acq = self.acquisition
        x, _ = self.solver.minimize(acq)
        return x

    def _get_solver(self, domain):
        solver = solvers.ScipySolver(domain= domain, initial_x=self.x0)
        return solver


class Greedy(ModelMixin, AcquisitionAlgorithm):
    """
    Implements the Upper Confidence Bound (UCB) algorithm.
    """

    def initialize(self, **kwargs):
        super(Greedy, self).initialize(**kwargs)

    def acquisition(self, X):
        X = X.reshape(-1, self.domain.d)
        return -(self.model.mean(X)-self.model.bias)/self.model.scale






