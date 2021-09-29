import numpy as np
from onlinetune.safe.models.gp import GP


class ModelMixin:
    """ Algorithm Class which provides a model and an optimizer instance, as configured in config.py
    """
    def initialize(self, **kwargs):
        super(ModelMixin, self).initialize(**kwargs)
        self._model_domain = kwargs.get('model_domain', self.domain_all)

        #config_manager.load_data(self.config.model_config)
        # if model was passed as kwarg, use it, else create a new model from config
        self.model = GP(self._model_domain)#, kwargs.get('knobs_detail'))

        self._has_constraints_model = True
        if self._has_constraints_model:
            #config_manager.load_data(self.config.constraints_model_config)
            self.s = [GP(self._model_domain) for _ in range(self.num_constraints)]


        # add initial data
        if not self.initial_data is None:
            for evaluation in self.initial_data:
                self._add_data_to_models(evaluation)

        self._initialize_best_prediction_algorithm(kwargs.copy())

    def _initialize_best_prediction_algorithm(self, greedy_initialize_kwargs):
        # avoid mutual imports since Greedy itself uses a ModelMixin
        from autotune.safe.algorithm import Greedy
        # initialize an algorithm to calculate best predicted point
        if not isinstance(self, Greedy):
            self._best_prediction_algorithm = None

            self._best_prediction_algorithm = Greedy()

            # do not pass any initial data to greedy algorithm, as we are using the current model
            if 'initial_data' in greedy_initialize_kwargs:
                del greedy_initialize_kwargs['initial_data']
            greedy_initialize_kwargs['model'] = self.model


            if self._has_constraints_model:
                greedy_initialize_kwargs['constraint_model'] = self.s

            self._best_prediction_algorithm.initialize(**greedy_initialize_kwargs)



    def add_data(self, data):
        """ by default just passes the observed data to the model """
        super(ModelMixin, self).add_data(data)
        default = float(data['default'])
        context = self.context
        tps = float(data['y']) + data['lower_bound_objective']
        knobs = data['x'].flatten()[:40]
        label = self.cluster.get_label(self.cluster.current_id)
        self.cluster.add_data_to_cluster(default, context, tps, knobs, label)
        self.cluster.array_to_file()


        self._add_data_to_models(data)

    def _add_data_to_models(self, data):
        x = self._get_x_from_data(data)
        if self.model.requires_std:
            self.model.add_data(x, data["y"], self._get_std(data))
        else:
            self.model.add_data(x, data["y"])
        if self.model._beta_cached > 10:
            self.model._beta_cached = 10

        if self._has_constraints_model and self.num_constraints:
            if self.s[0].requires_std:
                for m, s, s_std in zip(self.s, data['s'], data['s_std']):
                    m.add_data(x, s, s_std)
            else:
                for m, s in zip(self.s, data['s']):
                    m.add_data(x, s)



    def _get_x_from_data(self, data):
        return data['x']

    def _get_std(self, data):
        """
        get std of observations from data, potentially computed from some other model
        """
        return data["y_std"]

    def next(self, context=None):
        x, additional_data = super().next(context=context)
        return x, additional_data

    def optimize_model(self):
        self.model.optimize()

        if self._has_constraints_model:
            for s in self.s:
                s.optimize()

    def best_predicted(self):
        if self._best_prediction_algorithm is None:
            raise NotImplementedError

        return self._best_prediction_algorithm.next()[0]

    def _get_dtype_fields(self):
        fields = super()._get_dtype_fields()
        fields.append(('y_model', 'f8'))
        fields.append(('y_std_model', 'f8'))
        return fields

    def get_joined_constrained_cb(self, X):
        joined_ucb = np.empty(shape=(X.shape[0], self.num_constraints))
        joined_lcb = np.empty(shape=(X.shape[0], self.num_constraints))
        for i,s in enumerate(self.s):
            mean, var = s.mean_var(X)
            mean = mean.flatten()
            std = np.sqrt(var.flatten())
            joined_lcb[:,i], joined_ucb[:,i] = mean - s.beta * std, mean + s.beta * std
        return np.max(joined_lcb, axis=1).reshape(-1,1), np.max(joined_ucb, axis=1).reshape(-1,1)

