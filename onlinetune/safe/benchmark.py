import numpy as np
import os
import yaml
import time
from onlinetune.utils import logger
ts = int(time.time())
logger = logger.get_logger(__name__, 'log/train_simu_{}.log'.format(ts))



class BenchmarkEnvironment():
    """
    Base class for benchmark environments. Benchmark environment are implemented by specifying a single function _f(x)
    """

    def __init__(self, path=None):
        self._x = None  # current parameter set in the environment
        self._max_value = None  # max value achievable
        self._s = []
        self.lower_bound_objective = 0
        self._num_contexts = 1
        self._num_constraints = len(self._s) + int(self.lower_bound_objective is not None)
        self._env_parameters = {}
        self._init_seed()
        self._x0 = None
        self.random_x0 = False
        self.bias = 0
        self.scale = 1
        self.random_x0_min_value = None
        #self.seed = None


    def _get_random_initial_point(self):
        x0 = self.domain.l + self.domain.range * np.random.uniform(size=self.domain.d)
        if not self.random_x0_min_value is None:
            while self.f(x0) < self.random_x0_min_value:
                x0 = self.domain.l + self.domain.range * np.random.uniform(size=self.domain.d)

            logger.info("Found initial feasible point.")
        return  x0



    def _get_dtype_fields(self):

        if self.domain_all.d - self.domain.d > 0:
            fields = [('x', f'({self.domain_all.d},)f8'), ('y', 'f8'), ('context', '(1,%s)f' % (self.domain_all.d - self.domain.d))]
        else:
            fields =  [('x', f'({self.domain_all.d},)f8'), ('y', 'f8')]
        if self._num_constraints:
            fields += [('s', f"({self._num_constraints},)f8")]
        fields += [('y_exact', 'f8'), ('y_max', 'f8'), ('default', 'f8'), ('lower_bound_objective', 'f8')]
        return fields


    @property
    def dtype(self):
        return np.dtype(self._get_dtype_fields())


    def evaluate(self, x=None, knobs=None):
        evaluation = np.empty(shape=(), dtype=self.dtype)
        evaluation['y_exact'] = np.asscalar(self.f(x, knobs)) - self.lower_bound_objective
        self._x = self.new_x
        evaluation['x'] = self._x

        evaluation['y_max'] = self.max_value
        evaluation['lower_bound_objective'] = self.lower_bound_objective
        evaluation['default'] = self.lower_bound_objective / self.deviation
        evaluation['y'] = evaluation['y_exact']
        for i, s in enumerate(self._s):
            evaluation['s'][i] = s(self._x)

        if self.lower_bound_objective is not None:
            evaluation['s'][-1] = - evaluation['y_exact'] / np.abs(self.lower_bound_objective)

            #print ("lower_bound_objective: {}".format(self.lower_bound_objective))
            logger.info("evaluation['s'][-1]: {}".format(evaluation['s'][-1]))
        print (evaluation['y_exact']  / np.abs(self.lower_bound_objective))
        logger.info ( evaluation['y_exact']  / np.abs(self.lower_bound_objective))
        return evaluation

    def f(self, x):
        """
        Function to be implemented by actual benchmark.
        Args:
            x:

        Returns:

        """
        raise NotImplementedError

    @property
    def x0(self):
        return self._x0


    @property
    def max_value(self):
        if self._max_value is None:
            raise NotImplementedError

        return self._max_value*self.scale + self.bias

    @property
    def seed(self):
        """
        Provides a random seed for random generation of environments.
        """
        return self._seed

    @property
    def _requires_random_seed(self):
        """
        Overwrite this property and set to True to use random seed.
        """
        return False

    def _init_seed(self):
        """
        Initialize self.seed. First checks if self._path is provided, and if a file 'environment.yaml' exists in this path. If that file contains a dict {'seed', some_seed} this is used as seed. Else the the value from self.config.seed is taken, if this is None, a random integer is generated as seed. This seed, either randomly generated or from self.config.seed is saved in 'environment.yaml'.
        """
        # only if enviornment requires a random seed
        if not self._requires_random_seed:
            return

        # initialize seed
        self._seed = None

        # try to read seed from file
        if self._path:
            env_config_path = os.path.join(self._path, 'environment.yaml')
            if self._path and os.path.exists(env_config_path):
                with open(env_config_path, 'r') as f:
                    data = yaml.load(f)
                    self._seed = data['seed']
                    logger.info("Using random seed from environment.yaml.")
        else:
            logger.warning('Path not provided, cannot load/save seed.')

        # if seed was not loaded from file
        if self._seed is None:
            self._seed = self.seed
            # no seed given in configuration, pick a random one
            if self._seed is None:
                logger.info("No random seed provided in config, choosing a random random seed.")
                self._seed = np.random.randint(2 ** 32 - 1)

            # save seed
            if self._path:
                env_config_path = os.path.join(self._path, 'environment.yaml')
                data = {'seed': self._seed}
                with open(env_config_path, 'w') as f:
                    yaml.dump(data, f)
                    logger.info("Saved random seed to environment.yaml.")
        elif self.seed is not None and self._seed != self.seed:
            logger.warning(
                "Seed from saved environment file is different than seed in config. Using seed from environment file.")

