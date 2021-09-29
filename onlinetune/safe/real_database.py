from onlinetune.safe.domain import ContinuousDomain
from onlinetune.safe.benchmark import BenchmarkEnvironment
from onlinetune.knobs import logger
from onlinetune.dbenv import generate_knobs
from onlinetune.knobs import knob2action
import numpy as np
import joblib

class DatabaseBenchmark(BenchmarkEnvironment):
    def __init__(self, env, path=None, dynamic=True ):
        super().__init__(path)
        self._bench = env
        self._num_contexts = 10
        self.action_num = len(self._bench.default_knobs.keys())
        l = np.array([0] * self.action_num)
        u = np.array([1] * self.action_num)
        self.domain = ContinuousDomain(l, u)
        if dynamic:
            w_min = [1.5240603685379028, -2.614138126373291, 2.7521350383758545, -0.9760016798973083, -3.746350049972534, -2.523404836654663, 0.0, 0.0, 0.0, 0]
            w_max = [4.760213851928711, -0.9297570586204529, 3.904289722442627, 1.9165186882019043, -1.5053110122680664, 0.634671151638031, 236.8788270452624, 236.8788270452624, 0.9999839485712222, 10.978309071830996]
            l_all = np.array([0] * self.action_num + w_min)
            u_all = np.array([1] * self.action_num + w_max)
            self.domain_all = ContinuousDomain(l_all, u_all)
        else:
            self.domain_all = self.domain
        self._x0 =  knob2action(self._bench.default_knobs)
        self.count = 0
        self.y_default = 0
        self.context_count = 0
        self.new_x = None
        self.model_tune = joblib.load('tune.model')


    def initialize(self):
        env_info = {}
        if self.x0 is None:
            self._x0 = self.domain.l + self.domain.range/2

        if self.random_x0:
            logger.info("Using random initial point.")
            self._x0, _ = self._get_random_initial_point()

        env_info['x0'] = self.x0
        env_info['domain_all'] = self.domain_all
        env_info['domain'] = self.domain
        env_info['num_constraints'] = self._num_constraints

        return env_info


    def f(self, x, knobs=None):
        self.new_x = x
        if knobs == None:
            knobs = generate_knobs(x[:self.action_num], 'gp')

        external_metrics, internal_metrics, resource, knobs_rule = self._bench.step_GP(knobs, no_change=False)
        if not knobs_rule == knobs:
            logger.info('knobs is modified!')
            action_new = knob2action(knobs_rule)
            x_new = np.hstack((action_new.flatten(), x[self.action_num:].flatten()))
            self.new_x = x_new

        if self._bench.workload['name'] == 'job':
            tps = - external_metrics[1]
        else:
            tps = external_metrics[0]

        return np.array(tps)


    def get_context(self, dynamic=True):
        self._bench.flush_status()
        if dynamic:
            workload_vec, plan, context = self._bench.collec_default(get_embedding=dynamic)
        self.context = context
        self.count = self.count + 1
        self.y_default = self.model_tune.predict(self.context.reshape(1, -1))

        return self.context, self.y_default


