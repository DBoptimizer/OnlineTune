import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.svm import SVC
import pandas as pd
import math
from sklearn import preprocessing
import ConfigSpace
from sklearn.preprocessing import StandardScaler
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from sklearn.decomposition import PCA
from onlinetune.safe.models.gp import GP
from onlinetune.safe.domain import ContinuousDomain
from onlinetune.knobs import knob2action, initialize_knobs, gen_continuous, get_default_knobs
from onlinetune.knobs import logger

encoder = False
context_len = 10


class Cluster():
    def __init__(self, file, knob_f, algo_kwargs=None, dynamic=True):
        self.knobs_detail =  initialize_knobs(knob_f, -1)
        self.svm = SVC(kernel="rbf", gamma="auto")
        self.path = file
        if os.path.exists(file):
            self.load_context(file)
            if dynamic:
                self.dbscan_(self.contextL)
                self.learn_boundary()
            else:
                self.dbscan_label = np.ones(self.tpsL.shape) * -1
        else:
            self.init_empty_cluster()

        self.domain = algo_kwargs['domain']
        self.domain_all = algo_kwargs['domain_all']
        self.append_count = 0
        self.current_log = 0
        self.no_cluster = True



    def init_empty_cluster(self):
        self.weightL = np.empty((0, 5))
        self.contextL = np.empty((0, context_len))
        self.defaultL = np.empty(0)
        self.tpsL = np.empty(0)
        self.knobL = np.empty((0, 40))
        self.embeddingL = np.empty((0, 6))
        self.logL = np.empty(0)
        self.dbscan_label = np.empty(0)
        logger.info("init empty cluster")




    def load_context(self, file): #load from res: knobs, tps, context, default
        f = open(file)
        lines = f.readlines()
        defaultL, tpsL, knobL, weightL, contextL, lines_filter, embeddingL, logL = [], [], [], [], [], [], [], []
        for line in lines:
            if 'tps_0|' in line:
                continue

            tmp = line.split('|')
            #context
            tmp_context = tmp[-2].replace('inf', '0')
            context = eval(tmp_context.split('_')[-1])
            #weight
            tmp_weight = tmp[-4].replace('inf', '0')
            weight = eval(tmp_weight.split('_')[-1])
            # plan
            plan =  eval(tmp[-3].split('_')[-1])
            # default
            default = plan[-1]

            #tps
            try:
                tps = eval(tmp[1].split('_')[1])
            except:
                continue
            #knob
            knob = eval(tmp[0])
            if knob['key_buffer_size'] < 4096:
                logger.info (line)
                continue
            knob = knob2action(knob)
            #embedding
            embedding = eval(tmp[-6].split('_')[1])
            #log
            log = eval(tmp[-5].split('_')[1])

            lines_filter.append(line)

            contextL.append(context)
            weightL.append(weight)
            defaultL.append(default)
            tpsL.append(tps)
            knobL.append(knob)
            embeddingL.append(embedding)
            logL.append(log)


        self.weightL = np.array(weightL)
        self.contextL = np.array(contextL)
        self.defaultL = np.array(defaultL)
        self.tpsL = np.array(tpsL)
        self.knobL = np.array(knobL)
        self.embeddingL = np.array(embeddingL)
        self.lines_filter = lines_filter
        self.logL = np.array(logL)


    def dbscan_(self, X, replace=True):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca = PCA(n_components=3)
        pca.fit(X)
        X = pca.transform(X)
        db = DBSCAN(eps=0.7, min_samples=8).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        if n_clusters_ == 0:
            silhouette_score = 0
        else:
            silhouette_score = metrics.silhouette_score(X, labels)


        if replace:
            if n_clusters_ == 0:
                self.no_cluster = True
            else:
                self.no_cluster = False

            logger.info (labels)
            logger.info('Estimated number of clusters: %d' % n_clusters_)
            logger.info('Estimated number of noise points: %d' % n_noise_)
            logger.info("Silhouette Coefficient: %0.3f" % silhouette_score)

            self.dbscan_label = labels
            self.silhouette_score = silhouette_score
            self.learn_boundary()
            return labels
        else:
            return labels, silhouette_score


    def clusters_to_files(self, labelL):
        lines = np.array(self.lines_filter)
        label_set = list(set(labelL))

        if not os.path.exists('cluster_res'):
            os.system('mkdir cluster_res')

        for label in label_set:
            idx = np.where(np.array(labelL) == label)[0]
            lines_filter = lines[idx]
            lines_filter = list(lines_filter)
            f_out = os.path.join('cluster_res', 'label{}.txt'.format(label))
            f = open(f_out, 'w')
            f.writelines(lines_filter)
            f.flush()
            f.close()


    def get_cluster_best(self, label):
        idx = np.where(np.array(self.dbscan_label) == label)[0]
        default = self.defaultL[idx]
        tps = self.tpsL[idx]
        action = self.knobL[idx]
        improve = (tps - default) / np.abs(default)
        best_index = np.argmax(improve)
        if improve[best_index] < 0.05:
            return knob2action(get_default_knobs())

        return action[best_index, :]


    def clusters_to_gps_objective_one_label(self, label, dynamic=True):
        idx_label = np.where(np.array(self.dbscan_label) == label)[0]
        improve = (self.tpsL - self.defaultL) / np.abs(self.defaultL)
        idx_add = np.where((improve > np.percentile(improve, 90)) & (improve > 0.1))[0]
        idx = np.array(list(set(idx_label) | set(idx_add)))
        X_0 = self.knobL[idx]
        default = self.defaultL[idx]
        tps = self.tpsL[idx]
        context = self.contextL[idx]
        if dynamic:
            X = np.hstack((X_0, context))
        else:
            X = X_0.copy()
        Y = tps - default
        model = GP(self.domain_all)#, self.knobs_detail)
        model.train_gp(X, Y.reshape(-1, 1))
        logger.info ('finish training for label {}'.format(label))
        return model


    def clusters_to_gps_objective(self, labelL):
        label_set = list(set(labelL))
        if not os.path.exists('cluster_gp_objective'):
            os.system('mkdir cluster_gp_objective')

        for label in label_set:
            self.clusters_to_gps_objective_one_label(label)


    def clusters_to_gps_constraint_one_label(self, label, dynamic=True):
        idx = np.where(np.array(self.dbscan_label) == label)[0]
        X_0 = self.knobL[idx]
        default = self.defaultL[idx]
        if default[0] < 0:
            default = default
        else:
            default = default
        tps = self.tpsL[idx]
        context = self.contextL[idx]
        if dynamic:
            X = np.hstack((X_0, context))
        else:
            X = X_0.copy()
        Y = - (tps - default) / np.abs(default)
        model = GP(self.domain_all)#, self.knobs_detail)
        model.train_gp(X, Y.reshape(-1, 1))
        logger.info ('finish training for label {}'.format(label))
        return model


    def clusters_to_gps_constraint(self, labelL):
        label_set = list(set(labelL))
        if not os.path.exists('cluster_gp_constraint'):
            os.system('mkdir cluster_gp_constraint')

        for label in label_set:
            self.clusters_to_gps_constraint_one_label(label)


    def learn_boundary(self):
        labels = self.dbscan_label
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_ == 0:
            self.no_cluster = True
        else:
            self.svm.fit(self.contextL, self.dbscan_label)
            self.no_cluster = False


    def predict_cluster(self, x):
        if self.no_cluster:
            label = -1
        else:
            label = self.svm.predict(x.reshape(-1, self.contextL[0].shape[0]))[0]
        logger.info ("cluster {} is chosen".format(label))
        return label


    def get_label(self, id):
        for label in self.label_id:
            if self.label_id[label] == id:
                return label


    def array_to_file(self):
        f = open(self.path, 'w')
        for i in range(self.tpsL.shape[0]):
            knobs = self.knobL[i]
            knobs = gen_continuous(knobs)
            context = list(self.contextL[i])
            tps = self.tpsL[i]
            default = self.defaultL[i]
            embedding = list(self.embeddingL[i])
            log = self.logL[i]
            line = "{}|tps_{}|embed_{}|log_{}|weight_{}|plan_{}|context_{}|65d\n".format(knobs,
                    tps, embedding, log,[], [default], context)
            f.write(line)
        f.close()


    def get_important_knobs(self, label):
        from fanova import fANOVA
        idx = np.where(np.array(self.dbscan_label) == label)[0]
        actions = self.knobL[idx]
        X = pd.DataFrame()
        for i in range(actions.shape[0]):
            knob = gen_continuous(actions[i])
            X = X.append(knob,  ignore_index=True)

        cs = ConfigSpace.ConfigurationSpace()
        le = preprocessing.LabelEncoder()

        for c in self.knobs_detail.keys():
            if self.knobs_detail[c]['type'] == 'enum':
                le.fit(self.knobs_detail[c]['enum_values'])
                X[c] = le.transform(X[c])
                default_transformed = le.transform(np.array(str(self.knobs_detail[c]['default'])).reshape(1))[0]
                list_transformed = le.transform(self.knobs_detail[c]['enum_values']).tolist()
                knob = CategoricalHyperparameter(c, list_transformed, default_value=default_transformed)
            else:
                X[c] = X[c].astype('float')
                if self.knobs_detail[c]['min'] > X[c].min():
                    X = X[X[c] > self.knobs_detail[c]['min']]
                    logger.info (c)
                if self.knobs_detail[c]['max'] < X[c].max():
                    X = X[X[c] < self.knobs_detail[c]['max']]
                    logger.info (c)

                if self.knobs_detail[c]['max'] > 2**20:
                    X[c] = (X[c] - self.knobs_detail[c]['min'] )/self.knobs_detail[c]['max']
                    knob = UniformFloatHyperparameter(c, 0, 1, default_value=(self.knobs_detail[c]['default'] -self.knobs_detail[c]['min'])/ self.knobs_detail[c]['max'])

                else:
                    knob = UniformIntegerHyperparameter(c, self.knobs_detail[c]['min'], self.knobs_detail[c]['max'], default_value=self.knobs_detail[c]['default'])


            cs.add_hyperparameter(knob)

        context = self.contextL[idx]

        for i in range(context.shape[1]):
            name = 'w_{}'.format(i)
            X[name] = context[:, i]
            lb = min(self.domain_all.l[40 + i], self.contextL[:,i].min())
            ub = max(self.domain_all.u[40 + i], self.contextL[:, i].max())
            knob = UniformFloatHyperparameter(name, lb, ub, default_value=(lb + ub)/2 )
            cs.add_hyperparameter(knob)

        Y = self.tpsL[idx]
        f = fANOVA(X, Y, config_space=cs)
        im_dir = {}
        for i in self.knobs_detail.keys():
            value = f.quantify_importance((i,))[(i,)]['total importance']
            if not math.isnan(value):
                im_dir[i] = value

        a = sorted(im_dir.items(), key=lambda x: x[1], reverse=True)
        logger.info('Knobs importance rank:')
        out_knob = {}
        for i in range(0, len(a)):
            if a[i][1] != 0:
                logger.info("Top{}: {}, its importance accounts for {:.4%}".format(i + 1, a[i][0],
                                                                             a[i][1]))
                knob = a[i][0]
                out_knob[knob] = a[i][0]


        if len(out_knob.keys()) == 0:
            return False, None
        else:
            i = 0
            while i < len(out_knob.keys()):
                if np.random.rand() < 0.6:
                    break

        important_knob = list(out_knob.keys())[i]
        id = list(self.knobs_detail.keys()).index(important_knob)
        logger.info ("{} is chosen".format(important_knob))

        return True, id


    def ifReCluster(self):
        if self.append_count < 10:
            return False
        label_cluster, silhouette_score_cluster = self.dbscan_(self.contextL, replace=False)
        if self.no_cluster or len(set(label_cluster)) == 1:
            silhouette_score_no_cluster = 0
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(self.contextL)
            pca = PCA(n_components=3)
            pca.fit(X)
            X = pca.transform(X)
            silhouette_score_no_cluster = metrics.silhouette_score(X, self.dbscan_label)
        adjusted_mutual_info = metrics.adjusted_mutual_info_score(label_cluster, self.dbscan_label)
        logger.info("Adjusted Mutual Information: {}, silhouette_score_no_cluster: {}, silhouette_score_cluster: {}".format(adjusted_mutual_info, silhouette_score_no_cluster, silhouette_score_cluster))
        logger.info ("no cluster labels{}".format(self.dbscan_label))
        logger.info ("cluster labels{}".format(label_cluster))
        if  silhouette_score_no_cluster  < silhouette_score_cluster and adjusted_mutual_info < 0.6:
            return True


    def add_data_to_cluster(self, default, context, tps, knobs, label):
        if np.array(context).shape[0] == 0:
            context = np.zeros(context_len)
        embedding = context[:6]
        self.contextL = np.vstack((self.contextL, context))
        self.embeddingL = np.vstack((self.embeddingL, embedding))
        self.knobL = np.vstack((self.knobL, knobs))
        self.defaultL = np.hstack((self.defaultL, default))
        self.tpsL = np.hstack((self.tpsL, tps))
        self.dbscan_label = np.hstack((self.dbscan_label, label))
        self.logL = np.hstack((self.logL, self.current_log))
        self.append_count = self.append_count + 1
