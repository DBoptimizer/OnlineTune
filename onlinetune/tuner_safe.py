from onlinetune.safe.real_database import DatabaseBenchmark
from onlinetune.safe.safeopt import LineSafetyMixin, TrustRegionSafetyMixin, SafetyMixin
from onlinetune.safe.utils import join_dtypes, join_dtype_arrays
from onlinetune.knobs import logger
from onlinetune.knobs import gen_continuous, knob2action
from onlinetune.dbscan import Cluster
from time import time
import numpy as np
import os



class MySQLTuner:
    def __init__(self, env, data='', y_variable='tps'):
        self.env = env
        ts = time()
        self.expr_name = 'train_{}'.format(ts)
        MySQLTuner.create_output_folders()
        self.data = data
        self.y_variable = y_variable


    @staticmethod
    def create_output_folders():
        output_folders = ['log', 'save_knobs']
        for folder in output_folders:
            if not os.path.exists(folder):
                os.mkdir(folder)


class SafeMySQLTuner(MySQLTuner):

    def __init__(self, env, data='',  y_variable='tps'):
        super().__init__(env, data, y_variable)
        self.dynamic_workload = True
        self.env_line = DatabaseBenchmark(env, dynamic=self.dynamic_workload )
        env_info = self.env_line.initialize()
        algo_kwargs = {}
        for k, v in env_info.items():
            if not k in algo_kwargs:
                algo_kwargs[k] = v

        self.algo_kwargs = algo_kwargs
        self.n_trust_regions = 5
        self.algoL = []
        self.cluster = Cluster(self.data, self.env.knobs_config, algo_kwargs, self.dynamic_workload)
        self.cluster.current_log = self.env_line._bench.log_file
        self.trust_region_init_dynamic(algo_kwargs)
        self.best_predicted_every = 16
        self._time_dtype = np.dtype([('time_acq', 'f'), ('time_data', 'f')])
        self.evaluation_dtype = join_dtypes(self.algoL[0].dtype, self.env_line.dtype, self._time_dtype)

        self.t = 0
        self._data = []
        self.dset = None
        self.T = 100000
        self._exit = False
        self.knobs_pre = self.env_line._bench.default_knobs



    def trust_region_init_dynamic(self, algo_kwargs):
        label_set = list(set(self.cluster.dbscan_label))
        self.algoL = []
        self.cluster.label_id = {}
        self.n_trust_regions = len(label_set) if len(label_set) > 0 else 1

        for i in range(self.n_trust_regions):
            logger.info("init trust regions {}:".format(i))
            algorithm = SafetyMixin()
            if not self.dynamic_workload:
                algo_kwargs['domain_all'] = algo_kwargs['domain']
            algo_kwargs['knobs_detail'] = self.env_line._bench.knobs_detail
            algorithm.initialize(**algo_kwargs)

            if len(label_set) > 0:
                obj_model = self.cluster.clusters_to_gps_objective_one_label(label_set[i], dynamic=self.dynamic_workload)
                con_model = self.cluster.clusters_to_gps_constraint_one_label(label_set[i], dynamic=self.dynamic_workload)
                algorithm.model = obj_model
                algorithm.s = [con_model]
                algorithm._best_x = self.cluster.get_cluster_best(label_set[i])
                self.cluster.label_id[label_set[i]] = i
            else:
                self.cluster.label_id[-1] = 0
            algorithm.best_y = 0
            algorithm.context_change = True
            algorithm.knobL = []
            algorithm.ydeltaL = []
            algorithm.xL = []
            algorithm.knob_modelL = []
            algorithm.default_x = self.env_line.x0
            algorithm.cluster = self.cluster
            self.algoL.append(algorithm)
            self.pre_algo = 0


        logger.info("finish init trust region!")




    def tune(self):
        logger.info(f"Starting optimization: {self.algoL[0].name}")
        while not self._exit:
            self._run_step()



    def _run_step(self):
        logger.debug("Starting iteration %s" % self.t)
        try:

            evaluation = self._run_interaction()

            if not self.dset is None:
                self.dset.add(evaluation)
            else:
                self._data.append(evaluation)  # if no dset is provided, manually record data

            self.t += 1
        except (Exception, KeyboardInterrupt) as e:
            self._handle_exception(e)

        if self.algoL[self.cluster.current_id].exit:
            logger.info(f"Algorithm terminated.")

        self._exit = (self.t >= self.T) or self.algoL[self.cluster.current_id].exit



    def choose_trust_region_svm(self):

        if self.dynamic_workload and self.cluster.ifReCluster():
            logger.info("Recluster !!!!")
            self.cluster.append_count = 0
            self.cluster.dbscan_(self.cluster.contextL)
            self.trust_region_init_dynamic(self.algo_kwargs)


        label = self.cluster.predict_cluster(self.context)
        self.cluster.current_id = self.cluster.label_id[label]
        logger.info("Choose algorithm {}".format(self.cluster.current_id))
        print ("Choose algorithm {}".format(self.cluster.current_id))

        return self.cluster.current_id


    def _run_interaction(self):
        if not self.dynamic_workload:
            self.context_change = False

        if self.env_line._num_contexts > 0:
            if True:
                self.context, self.y_default = self.env_line.get_context(dynamic=self.dynamic_workload)
                self.cluster.current_log = self.env_line._bench.log_file
                self.clear_algo()
                logger.info("context change to: {}, lower_bound_objective change to {}".format(list(self.context), self.y_default))
                print ("context change to: {}, lower_bound_objective change to {}".format(list(self.context), self.y_default))
                self.cluster.current_id = self.choose_trust_region_svm()

            id = self.cluster.current_id

            self.algoL[id].s[0].modify_beta = True
            self.env_line.lower_bound_objective = self.y_default

            if '_line_domain' in dir(self.algoL[id]):
              self.algoL[id]._line_domain.context = self.context
            if '_tr_domain' in dir(self.algoL[id]):
                self.algoL[id]._tr_domain.context = self.context


            start = time()
            x, additional_data = self.algoL[id].next(self.context)
            knobs = gen_continuous(x[:self.env_line.action_num])
            knobs_next = self.knob_diff(self.knobs_pre, knobs)
            time_acq = time() - start
        else:
            # call algorithm without context
            start = time()
            x, additional_data = self.algorithm.next()
            time_acq = time() - start

        env_evaluation = self.env_line.evaluate(x, knobs_next)
        knobs_next = gen_continuous(self.env_line.new_x)
        self.knobs_pre = knobs_next
        if env_evaluation['s'][-1] < 0:
            self.algoL[id].context_change = False
            self.env_line._bench.rule.feedback(safe=True)
        else:
            self.env_line._bench.rule.feedback(safe=False)

        self.algoL[id].rule = self.env_line._bench.rule
        evaluation = join_dtype_arrays(env_evaluation, additional_data, self.evaluation_dtype).view(np.recarray)

        start = time()


        self.algoL[id].knobL.append(knobs_next)
        self.algoL[id].ydeltaL.append(evaluation['y'] / np.abs(self.y_default))
        self.algoL[id].xL.append(x[:self.env_line.action_num])
        self.algoL[id].s[0].x_added = x
        self.algoL[id].s[0].y_added = evaluation['y'] / np.abs(self.y_default)
        self.algoL[id].add_data(evaluation)
        delta = evaluation['y'] / np.abs(self.y_default)
        improve = (self.cluster.tpsL - self.cluster.defaultL) / np.abs(self.cluster.defaultL)
        if delta > np.percentile(improve, 90) and delta > 0.1:
            knob_tmp = gen_continuous(evaluation['x'][:self.algoL[id].domain.d])
            for i in list(self.cluster.label_id.values()):
                if not i == id and not knob_tmp in self.algoL[i].knobL:
                    self.algoL[i].model.add_data(evaluation['x'], evaluation['y'])
                    logger.info('good knobs, also add to algoL {}'.format(i))



        time_data = time() - start

        evaluation['time_acq'] = time_acq
        evaluation['time_data'] = time_data

        logger.debug(f"Objective value {evaluation.y}.")
        logger.debug(f"Completed step {self.t}.")

        return True, evaluation

    def _handle_exception(self, e):
        raise e  # by default, just raise the exception

    def clear_algo(self):
        for i in range(self.n_trust_regions):
            self.algoL[i].best_y = 0
            self.algoL[i].context_change = True
            self.algoL[i].knobL = []
            self.algoL[i].ydeltaL = []
            self.algoL[i].xL = []
            self.algoL[i].knob_modelL = []


    def knob_diff(self, k1, k2):
        logger.info("Diff from previous:")
        for k in k2.keys():
            if self.env_line._bench.knobs_detail[k]['type'] == 'integer':
                range = self.env_line._bench.knobs_detail[k]['max'] - self.env_line._bench.knobs_detail[k]['min']
                if np.abs(k1[k] - k2[k])/range > 1e-9 :
                    logger.info ('{}:{} -> {}'.format(k, k1[k], k2[k]))
                else:
                    k2[k] = k1[k]
            elif not  k1[k] == k2[k]:
                logger.info('{}:{} -> {}'.format(k, k1[k], k2[k]))


        logger.info("Diff from default:")
        for k in k2.keys():
            if self.env_line._bench.knobs_detail[k]['type'] == 'integer':
                range = self.env_line._bench.knobs_detail[k]['max'] - self.env_line._bench.knobs_detail[k]['min']
                if np.abs(self.env_line._bench.default_knobs[k] - k2[k])/range > 1e-9 :
                    logger.info ('{}:{} -> {}'.format(k, self.env_line._bench.default_knobs[k], k2[k]))
                else:
                    k2[k] = self.env_line._bench.default_knobs[k]
            elif not  k1[k] == self.env_line._bench.default_knobs[k]:
                logger.info('{}:{} -> {}'.format(k, self.env_line._bench.default_knobs[k], k2[k]))
        return k2


