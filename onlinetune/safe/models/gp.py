from GPy.util.linalg import dtrtrs, tdot, dpotrs
import numpy as np
from GPy.models import GPRegression
import GPy
from scipy.optimize import minimize
import copy
from onlinetune.knobs import logger



def optimize_gp(experiment):
    experiment.algorithm.f.gp.kern.variance.fix()
    experiment.algorithm.f.gp.optimize()
    print(experiment.algorithm.f.gp)


class GP():
    """
    Base class for GP optimization.
    """

    def __init__(self, domain):
        self.domain = domain
        self.kernel = self._get_kernel()
        input_dim = domain.d
        self._X = np.zeros(input_dim).reshape(1, -1)
        self._Y = np.zeros(1).reshape(1, -1)
        self.gp = self._get_gp()
        self.t = 0
        self.kernel = self.kernel.copy()
        self._woodbury_chol = np.asfortranarray(self.gp.posterior._woodbury_chol)  # we create a copy of the matrix in fortranarray, such that we can directly pass it to lapack dtrtrs without doing another copy
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._beta_cached = 0.05
        self.beta_pre = self._beta_cached
        self._bias = 0
        self.optimize_flag = False
        self.calculate_gradients = True
        self.delta = 0.05
        self.modify_beta = False

    def _get_kernel(self):
        kernel = None
        action_num = self.domain.d - 10
        context_num = 10
        kernel1 = GPy.kern.Matern52(input_dim=action_num , variance=2., lengthscale=0.01, ctive_dims=[i for i in range(0, action_num )], ARD=True)
        kernel2 = GPy.kern.Linear(input_dim=context_num, active_dims=[i for i in range(action_num, action_num + context_num)], ARD=True)
        kernel = kernel1 + kernel2

        return kernel

    def _beta(self):
        return self._beta_cached

    @property
    def requires_std(self):
        return False

    @property
    def scale(self):
        if self.gp.kern.name == 'sum':
            return sum([part.variance for part in self.gp.kern.parts])
        else:
            return np.sqrt(self.gp.kern.variance)

    @property
    def bias(self):
        return self._bias

    def _get_gp(self):
        return GPRegression(self._X, self._Y, self.kernel, normalizer=True)

    def add_data(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y).reshape(-1,1)
        self._Y = np.vstack([self._Y, y])  # store unbiased data
        self._X = np.vstack([self._X, x])
        self._bias = 0#self._Y.mean()
        self._var = 1#self._Y.std()
        model = copy.deepcopy(self.gp)
        flag = True
        try:
            model.set_XY(self._X, (self._Y-self._bias)*self._var)
            model.optimize()
        except:
            flag = False
            print ("pass training")

        if flag:
            self.gp = model
            self._update_beta()
            self.t += y.shape[1]

        return self.gp


    def train_gp(self,x , y):
        self._Y = y.reshape(-1,1)  # store unbiased data
        self._X = x
        self._bias = 0  # self._Y.mean()
        self._var = 1  # self._Y.std()
        self.gp.set_XY(self._X, (self._Y - self._bias) * self._var)
        self.gp.optimize(messages=True,max_f_eval = 3000)
        self._update_beta()

        return self.gp


    def set_beta(self):
        logger.info("check beta!")
        if not self.check_beta():
            logger.info("Set beta from {} to the previous {}".format(self._beta_cached, self.beta_pre))
            self._beta_cached = self.beta_pre
            while not self.check_beta():
                self._beta_cached = self._beta_cached * 0.99
                logger.info("reduce beta to {}".format(self._beta_cached))
        else:
            return

    def optimize(self):
        if self.config.optimize_bias:
            self._optimize_bias()
        if self.config.optimize_var:
            self._optimize_var()

        self.beta_pre = self._beta()

        self._update_beta()



    def _update_cache(self):
        self._woodbury_chol = np.asfortranarray(self.gp.posterior._woodbury_chol)
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        self._X = self.gp.X.copy()

        if self.optimize_flag:
            self.optimize()


    def _optimize_bias(self):
        self._bias = minimize(self._bias_loss, self._bias, method='L-BFGS-B')['x'].copy()
        self._set_bias(self._bias)

    def _bias_loss(self, c):
        new_woodbury_vector,_= dpotrs(self._woodbury_chol, self._Y - c, lower=1)
        K = self.gp.kern.K(self.gp.X)
        mean = np.dot(K, new_woodbury_vector)
        norm = new_woodbury_vector.T.dot(mean)
        # loss is least_squares_error + norm
        return np.asscalar(np.sum(np.square(mean + c - self._Y)) + norm)

    def _set_bias(self, c):
        self._bias = c
        self.gp.set_Y(self._Y - c)
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()


    def check_beta(self):
        if 'X_eval_embedded' in dir(self):
            mean, var = self.mean_var(self.X_eval_embedded)
            ucb = mean + self._beta() * var
            safe_index = np.where((self.delta_y>0) & (mean<0))
            safe_mean = mean[safe_index]
            safe_var = var[safe_index]
            safe_ucb = ucb[safe_index]
            if safe_ucb.shape[0] >0 and (safe_ucb >= 0).all():
                logger.info("wrong line")
                index = np.where(safe_ucb>=0)
                logger.info("mean:{},var:{},ucb{}".format(safe_mean[index], safe_var[index], safe_ucb[index]))
                return False
        if 'y_added' in dir(self) and self.y_added > 0:
            mean, var = self.mean_var(self.x_added)
            ucb = mean + self._beta() * var
            if ucb < 0:
                return True
            else:
                if mean > 0:
                    logger.info("wrong center mean")
                    return True
                logger.info("wrong center")
                logger.info("mean:{},var:{},ucb:{}".format(mean, var, ucb))
                return False
        else:
            return True




    def _update_beta(self):
        logdet = self._get_logdet()
        self._woodbury_vector = self.gp.posterior._woodbury_vector.copy()
        logdet_priornoise = self._get_logdet_prior_noise()
        self._beta_cached = np.asscalar(np.sqrt(2 * np.log(1 / self.delta) + (logdet - logdet_priornoise)) + self._norm())
        if np.isnan(self._beta_cached):
            logger.info("beta is nan, use previous beta")
            self._beta_cached = self.beta_pre
        logger.info(f"Updated beta to {self._beta_cached}")
        self.beta_pre = self._beta_cached

    def _optimize_var(self):
        # fix all parameters
        for p in self.gp.parameters:
            p.fix()

        if self.gp.kern.name == 'sum':
            for part in self.gp.kern.parts:
                part.variance.unfix()
        else:
            self.gp.kern.variance.unfix()
        self.gp.optimize()
        if self.gp.kern.name == 'sum':
            values = []
            for part in self.gp.kern.parts:
                values.append(np.asscalar(part.variance.values))
        else:
            values = np.asscalar(self.gp.kern.variance.values)

        for p in self.gp.parameters:
            p.unfix()

    def _get_logdet(self):
        return 2.*np.sum(np.log(np.diag(self.gp.posterior._woodbury_chol)))

    def _get_logdet_prior_noise(self):
        return self.t * np.log(self.gp.likelihood.variance.values)


    def mean_var(self, x):
        """Recompute the confidence intervals form the GP.
        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        x = np.atleast_2d(x)

        mean,var = self.gp.predict_noiseless(x)
        if var.min() < 0:
            try:
                min_postive_var = var[np.where(var>0)].min()
            except:
                min_postive_var = 1e-9
            var[np.where(var <= 0)] = min_postive_var
            logger.info("modify negative var!!!")

        return (mean + self._bias)* self._var, var

    def mean_var_grad(self, x):
        return self.gp.predictive_gradients(x)

    def var(self, x):
        return self.mean_var(x)[1]

    def std(self, x):
        return np.sqrt(self.var(x))


    def mean(self, x):
        return self.mean_var(x)[0]

    def set_data(self, X, Y, append=True):
        if append:
            X = np.concatenate((self.gp.X, X))
            Y = np.concatenate((self.gp.Y, Y))
        self.gp.set_XY(X, Y)
        self.t = X.shape[0]
        self._update_cache()


    def _norm(self):
        norm = self._woodbury_vector.T.dot(self.gp.kern.K(self.gp.X)).dot(self._woodbury_vector)
        return np.asscalar(np.sqrt(norm))



    def __getstate__(self):
        self_dict = self.__dict__.copy()
        return self_dict

    @property
    def beta(self):
        """
        Scaling Factor beta
        """
        return self._beta()



    def ucb(self, x):
        mean, var = self.mean_var(x)
        std = np.sqrt(var)
        return mean + self.beta * std

    def lcb(self, x):
        mean, var = self.mean_var(x)
        std = np.sqrt(var)
        return mean - self.beta * std

    def ci(self, x):

        mean, var = self.mean_var(x)
        std = np.sqrt(var)
        beta = self.beta
        return mean - beta * std, mean + beta * std


