from onlinetune.safe.algorithm import Algorithm
from onlinetune.safe.model import ModelMixin
import numpy as np
import os
from onlinetune.knobs import gen_continuous
import matplotlib.pyplot as plt
from onlinetune.safe.utils import maximize, dimension_setting_helper, plot_parameter_changes, plot_model_changes
from .subdomain import LineDomain, TrustRegionDomain
from onlinetune.knobs import logger



def sample_grad_gp(model, x0, scale, eps=0.01):
    points = x0 + np.eye(len(x0))*scale*eps
    points = np.vstack((x0, points))
    Y = model.gp.posterior_samples_f(points, size=1).flatten()
    return (Y[1:] - Y[0])/(scale*eps)

def mean_grad_gp(model, x0, scale, eps=0.01):
    if not self.context is None:
        x0 = np.hstack((x0.reshape(1,-1), self.context))
    return model.gp.predictive_gradients(x0.reshape(1,-1))[0].flatten()


def ts(model, X):
    return model.gp.posterior_samples_f(X, size=1)

def ucb(model, X):
    return model.ucb(X)

class SubDomainBO(ModelMixin, Algorithm):
    """
    This class is used to run a 1-dim version of BO on a Sumdomain
    """

    def initialize(self, **kwargs):
        super(SubDomainBO, self).initialize(**kwargs)
        self.points_in_max_interval_to_stop = 10
        self.min_queries_line = 5
        self.max_queries_line = 10
        self.min_queries_tr = 5
        self.max_queries_tr = 10 #self.domain.d /
        self.tr_radius = 0.04
        self.tr_method = 'grad'
        self.line_boundary_margin = 0.1
        self.plot = False
        self.plot_every_step = False

        self.acquisition = 'febo.algorithms.subdomainbo.acquisition.ts'
        self._best_x = self.x0.copy()

        self._best_direction = None
        self._phase = 'best'
        self._iteration = 0

        self._parameter_names = kwargs.get('parameter_names')

        self._max_queries_line = dimension_setting_helper(self.max_queries_line, self.domain.d)
        self._min_queries_line = dimension_setting_helper(self.min_queries_line, self.domain.d)
        self._max_queries_tr = dimension_setting_helper(self.max_queries_tr, self.domain.d)
        self._minx_queries_tr = dimension_setting_helper(self.min_queries_tr, self.domain.d)
        self._point_type_addition = ''
        self.plot = False
        self.__acquisition = ucb
        self.best_y_evaluated = {}
        self.best_x_evauated = {}
        self.line_boundary_margin = 0.05
        self.successive_success = 0
        self.successive_fail = 0
        self.pre_improve = 0




    def _add_context(self, x):
        if np.array(self.context).shape[0] == 0 :
            return x
        context = self.context
        if context is None:
            return x
        context = np.atleast_2d(context)
        x =np.atleast_2d(x)
        num_contexts = len(self.domain_all.l) - len(self.domain.l)
        num_feature= len(self.domain.l)

        x2 = np.empty((x.shape[0], num_feature + num_contexts), dtype=float)
        x2[:, :num_feature] = x[:, :num_feature]
        x2[:, num_feature:] = context
        return x2

    def _next(self, context=None):

        self.context = context
        additional_data = {'iteration' : self._iteration}
        # sampling phases
        logger.info ("next: enter {} phase".format(self._phase))
        print ("enter {} phase".format(self._phase))
        #pdb.set_trace()
        if self._phase == 'best':
            additional_data['point_type'] = 'best'
            if '_tr_domain' in dir(self):
                self._best_x = self._tr_solver_best()
            elif '_line_domain' in dir(self):
                self._best_x = self._line_solver_best()

            self._best_x = self._add_context(self._best_x.reshape(1, -1)).flatten()
            self._point_type_addition = 'best'
            return self._best_x, additional_data

        if self._phase == 'line':
            additional_data['point_type'] = 'line'  + self._point_type_addition
            self._point_type_addition = ''
            x_next = self._line_solver_step()
            # if a tuple is returned, it contains x,m
            if isinstance(x_next, tuple):
                x_next, m = x_next
                additional_data['m'] = m
                logger.info(f"Choosing {m} measurements.")

            return x_next, additional_data

        if self._phase == 'tr':
            additional_data['point_type'] = 'tr' + self._point_type_addition
            self._point_type_addition = ''
            x_next = self._tr_solver_step()
            if isinstance(x_next, tuple):
                x_next, m = x_next
                additional_data['m'] = m
                logger.info(f"Choosing {m} measurements.")
            return x_next, additional_data



    def add_data(self, data):
        knob = gen_continuous(data['x'][:self.domain.d])
        if knob in self.knob_modelL:
            logger.info("evluated knobs, skip adding data to model")
        else:
            logger.info("new knobs, adding data to model")
            super().add_data(data)
            self.knob_modelL.append(knob)

        if -data['s'][-1] > self.pre_improve:
            self.successive_fail = 0
            self.successive_success = self.successive_success + 1
        elif -data['s'][-1] < self.pre_improve:
            self.successive_success = 0
            self.successive_fail =self.successive_fail + 1

        self.pre_improve = -data['s'][-1]

        if self._point_type_addition == "-expander":
            if  data['s'][-1] < - 0.05:
                self.tr_radius = self.tr_radius + 0.005
                logger.info("explore safe, increase tr_radius to {}".format(self.tr_radius))
            elif data['s'][-1] > 0  and self.tr_radius > 0.02:
                self.tr_radius = self.tr_radius - 0.01
                logger.info("explore un safe, decrease tr_radius to {}".format(self.tr_radius))

        if self.successive_fail > 4:
            self.tr_radius = self.tr_radius / 1.5
            self.successive_fail = 0
            logger.info("decrease tr_radius to {}".format(self.tr_radius))
            print ("decrease tr_radius to {}".format(self.tr_radius))
        if self.successive_success > 3:
            self.tr_radius = self.tr_radius * 1.1
            self.successive_success = 0
            logger.info("increase tr_radius to {}".format(self.tr_radius))
            print ("increase tr_radius to {}".format(self.tr_radius))

        if self.tr_radius > 0.3:
            self.tr_radius = 0.3

        #self.s[0].set_beta()
        #pdb.set_trace()
        # evaluate stopping conditions
        logger.info("Add data for phase:{}, tps:{}".format(self._phase, data['y']))
        if tuple(self.context) not in self.best_x_evauated:
            self.best_x_evauated[tuple(self.context)] = data['x']
            self.best_y_evaluated[tuple(self.context)] = data['y']
        else:
            if data['y'] > self.best_y_evaluated[tuple(self.context)]:
                self.best_x_evauated[tuple(self.context)] = data['x']
                self.best_y_evaluated[tuple(self.context)] = data['y']

        '''if data['y'] > self.best_y:
            self.best_y = data['y']
            self._best_x = data['x']'''


        if self._phase == 'line':
            # add line data
            self._line_add_data(data)
            self._best_x = self._line_solver_best()
            self._best_x_list.append(self._best_x.copy())

            if self._line_solver_stop():
                self._line_solver_finalize()
                self._phase = 'best'

        elif self._phase == 'tr':
            # add tr data
            self._tr_add_data(data)
            self._best_x = self._tr_solver_best()

            if self._tr_solver_stop():
                # compute best direction
                self._best_direction = self._tr_solver_best_direction()
                self._tr_solver_finalize()
                self._phase = 'best'

        elif self._phase =='best':
            self._iteration += 1
            self._phase, subdomain = self._get_new_subdomain()
            try:
                self.context = data['context']
            except:
                pass
            subdomain.context = self.context
            #logger.info(f'best_x evaluate, y={data["y"]}')


            if self._phase == 'line':
                self._line_solver_init(subdomain)
            elif self._phase == 'tr':
                self._tr_solver_init(subdomain)

            logger.info(f'starting {self._iteration}, {self._phase}-solver.')


    def get_data_center_from_context(self):
        best_x = self.default_x.copy()
        lcb_best, ucp_best = self.get_joined_constrained_cb(self._add_context(best_x.reshape(1, -1)))
        for context_source in self.best_x_evauated.keys():
            x_source = self._add_context(self.best_x_evauated[context_source])
            s_lcb, s_ucb = self.get_joined_constrained_cb(x_source.reshape(1, -1))
            if s_lcb < lcb_best:
                lcb_best = s_lcb
                best_x = self.best_x_evauated[context_source]
        knob = gen_continuous(best_x)
        print ("Region center change to:{}".format(knob))
        return best_x

    def _get_new_subdomain(self):
        if self._iteration % 2 == 0:
            return 'tr',
        else:
            return 'line',

    def _line_solver_init(self, line_domain):
        self._line_data = []
        self._best_x_list = []
        self._line_domain = line_domain


    def _line_add_data(self, data):
        self._line_data.append(data)


    def _line_solver_stop(self):
        # don't stop below min-queries
        if len(self._line_data) <= self._min_queries_line:
            return False

        # accuracy of maximum < 1%
        if self._line_max_ucb() - self.model.lcb(self._best_x) < 0.01*self.model.mean(self._best_x):
            logger.warning("Uncertainty at best_x reduced to 1%, stopping line.")
            return True

        # best_x didn't change after half the samples
        # flexible_query_range = max(self._max_queries_line - self._min_queries_line, 6)
        # if len(self._line_data) >= self._min_queries_line + flexible_query_range/4:
        #     # maximum distance of last few best_x did not change by more than 2 % on domain
        #     if np.max(pdist(self._best_x_list[-flexible_query_range // 2:], w=1/self.domain.range**2)) < 0.02:
        #         logger.warning(f"No best_x change in {flexible_query_range // 2} steps. Stopping line.")
        #         return True

        # stop at max queries
        if len(self._line_data) >= self._max_queries_line:
            logger.info("stop at max queries")

        return len(self._line_data) >= self._max_queries_line

    def _line_solver_best(self):
        boundary_margin = self._line_domain.range * self.line_boundary_margin
        def mean(X):
            # return model mean on line, but ignore a margin at the boundary to account for boundary effects of the gp
            return -self.model.mean(self._line_domain.embed_in_domain(X)) \
                   + 10e10*np.logical_or(X < self._line_domain.l + boundary_margin, X > self._line_domain.u - boundary_margin)
        x_line, res = self._line_solver.minimize(mean)
        logger.info("res: {}".format(res))
        return self._line_domain.embed_in_domain(x_line).flatten()


    def _line_solver_finalize(self):
        if self.plot:
            self._save_line_plot()

    def _save_line_plot(self, with_step_num=False):
        """
        Save a plot of the current line. The plot is generated in .plot_line(...)
        """
        f = plt.figure()
        axis = f.gca()

        self.plot_line(axis=axis)

        # save plot
        group_id = self.experiment_info.get("group_id", "")
        if group_id is None: # group_id might be set to None already in self.experiment_dir
            group_id = ""

        path = os.path.join(self.experiment_info["experiment_dir"], "plots", str(group_id), str(self.experiment_info.get("run_id", "")))
        os.makedirs(path, exist_ok=True)
        f.subplots_adjust(top=0.71)
        if with_step_num:
            path = os.path.join(path, f'Iteration_{self._iteration}_{self.t}.pdf')
        else:
            path = os.path.join(path, f'Iteration_{self._iteration}.pdf')
        f.savefig(path)
        logger.info(f'Saved line plot to {path}')
        plt.close()

    def plot_line(self, axis, steps=300):
        """
        This function uses the datapoints measured in one dim and plots these together with the standard deviation
        and mean of the model to check the lengthscale. It returns the plots into a folder with one plot per line in the dropout algorithm

        :param axis: axis to plot on
        :param line_data:
        :param steps:
        """

        # first create evaluation grid with correct bounds on the sub-domain
        X_eval = np.linspace(self._line_domain.l[0], self._line_domain.u[0], steps)

        # then we evaluate the mean and the variance by projecting back to high-d space
        X_eval_embedded = self._line_domain.embed_in_domain(X_eval.reshape(-1, 1))
        mean, var = self.model.mean_var(X_eval_embedded)
        mean, std = mean.flatten(), np.sqrt(var).flatten()

        # we plot the mean, mean +/- std and the data points
        axis.fill_between(X_eval, mean - std, mean + std, alpha=0.4, facecolor='grey', color='C0')
        axis.plot(X_eval, mean, color='C0')


        data_x = [self._line_domain.project_on_line(p['x']).flatten() for p in self._line_data]
        data_y = [p['y'] for p in self._line_data]
        axis.scatter(data_x, data_y,marker='x', c='C0')

        # starting and best_predicted point
        axis.axvline(self._line_domain.project_on_line(self._line_domain.x0), color='C0', linestyle='--')
        axis.axvline(self._line_domain.project_on_line(self._best_x), color='C0')

        # add some information in the title
        axis.set_title(f'Iteration: {self._iteration}'
                       f'\nbeta= {round(self.model.beta,3)}, variance= {round(self.model.gp.kern.variance[0],3)}, '
                       f'\nnoise variance= {round(self.model.gp.Gaussian_noise.variance[0],5)}')

        return X_eval, X_eval_embedded, data_x

    def _tr_solver_init(self, tr_domain):
        self._tr_domain = tr_domain
        self._tr_data = []

    def _tr_add_data(self, data):
        self._tr_data.append(data)



    def _tr_solver_best_direction(self):
        direction = (self._best_x - self._tr_domain.x0)[:self.domain.d].reshape(1, -1)
        # if change is less  than 2% of tr-radius or increase is less then 0.5%, pick a random direction
        if np.linalg.norm(direction/self._tr_domain.radius) < 0.02:
            logger.warning('change in best_x < 2% of trust-region, picking other direction.')
            direction = self.get_other_direction()
        else:
            y_x0 = self.model.mean(self._tr_domain.x0)
            y_new = self.model.mean(self._best_x)
            if y_new/y_x0 < 1.005:
                logger.warning('predicted objective increase at best_x < 0.5%, picking other direction.')
                direction = self.get_other_direction()
            else:
                logger.info("pick descent direction")

        return direction

    def _tr_solver_stop(self):
        return len(self._tr_data) >= self._max_queries_tr

    def _tr_solver_finalize(self):
        return
)

    def _tr_save_plot(self, with_step_num=False):
        """
              Save a plot of the current line. The plot is generated in .plot_line(...)
              """
        fig, axis = plt.subplots(ncols=1, figsize=(self.domain.d, 4))



        plot_parameter_changes(axis, self._parameter_names, self._tr_domain.x0, self._best_x, self.domain.l, self.domain.u, self._tr_domain.radius, self.x0)

        x0 = self._tr_domain.x0
        xnew = self._best_x
        y_x0 = np.asscalar(self.model.mean(x0.reshape(1, -1)))
        y_xnew = np.asscalar(self.model.mean(xnew.reshape(1,-1)))
        # ucb_xnew = np.asscalar(self.model.ucb(xnew.reshape(1,-1)))
        std_xnew = np.asscalar(self.model.std(xnew.reshape(1,-1)))

        y_coord = np.empty(self.domain.d)
        # ucb_coord = np.empty(self.domain.d)
        for i in range(self.domain.d):
            axis_points = self._tr_domain.get_axis_points(i)
            y_coord[i] = np.max(self.model.mean(axis_points))
            # ucb_coord[i] = np.max(self.model.ucb(axis_points))

        plot_model_changes(axis, y_x0, y_xnew, std_xnew, y_coord)

        # save plot
        group_id = self.experiment_info.get("group_id", "")
        if group_id is None:  # group_id might be set to None already in self.experiment_dir
            group_id = ""

        path = os.path.join(self.experiment_info["experiment_dir"], "plots", str(group_id),
                            str(self.experiment_info.get("run_id", "")))
        os.makedirs(path, exist_ok=True)
        # fig.subplots_adjust(wspace=0.4)
        if with_step_num:
            path = os.path.join(path, f'Iteration_{self._iteration}_{self.t}.pdf')
        else:
            path = os.path.join(path, f'Iteration_{self._iteration}.pdf')
        fig.savefig(path, bbox_inches="tight")
        logger.info(f'Saved trust-region plot to {path}')
        plt.close()

    def global_acquisition(self, X):
        return self.__acquisition(self.model, X)

    def _get_dtype_fields(self):
        fields = super()._get_dtype_fields()
        fields += [('iteration', 'i')]
        fields += [('direction', '(1,%s)f' % self.domain.d)]
        fields += [('point_type', 'S25')]
        return fields

    def best_predicted(self):
        return self.domain_all.project(self._best_x)


    def get_other_direction(self):
        if np.random.rand() > 0.6:
            return  self.get_random_direction()
        else:
            try:
                return self.get_important_direction()
            except:
                return self.get_random_direction()



    def get_random_direction(self):
        """
        creates a random directional vector in d = domain.d dimensions
        :return: return a vector in shape (1, self.domain.d)
        """
        x_c = self._best_x.copy()
        while True:
            direction = np.random.normal(size=self.domain.d).reshape(1, -1)
            direction /= np.linalg.norm(direction)
            direction *= self.domain.range  # scale direction with parameter ranges, such that the expected change in each direction has the same relative magnitude
            line_domain = LineDomain(self.domain, x_c, direction)
            if line_domain.upper - line_domain.lower > 0:
                break

        return direction

    def get_important_direction(self):
        label = self.cluster.get_label(self.cluster.current_id)
        flag, id = self.cluster.get_important_knobs(label)
        if flag:
            direction = np.eye(self.domain.d)[id].reshape(1, -1)
        else:
            direction = self.get_random_direction()
        return direction


class AscentLineBO(SubDomainBO):
    """
    Bayesian Optimization with alternateting trust-region and line-search.
    """

    def _get_new_subdomain(self):
        radius = self.domain.range * self.tr_radius
        if self.context_change:
            logger.info("Context change, use re-predicted x0 as center")
            x_c = self.get_data_center_from_context()
            x_c = self._add_context(x_c).flatten()
        else:
            x_c = self._best_x.copy()
            x_c = self.rule.check_safe_action(x_c, no_explore=True)
            knob =  gen_continuous(x_c)
            x_c = self._add_context(x_c).flatten()
            logger.info("Center change to best x: {}".format(knob))

        if self._iteration % 2 == 1:
            print (radius)
            logger.info("Use Trust region")
            return 'tr', TrustRegionDomain(self.domain, x_c, radius=radius)
        else:
            logger.info("Use Line region")
            line_domain = LineDomain(self.domain, x_c, self._best_direction)
            if line_domain.upper - line_domain.lower > 0:
                return 'line', line_domain
            else:
                while True:
                    direction = np.random.normal(size=self.domain.d).reshape(1, -1)
                    direction /= np.linalg.norm(direction)
                    direction *= self.domain.range  # scale direction with parameter ranges, such that the expected change in each direction has the same relative magnitude
                    line_domain = LineDomain(self.domain, x_c, direction)
                    if line_domain.upper - line_domain.lower > 0:
                        break
                return 'line', line_domain



