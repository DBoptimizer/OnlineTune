import numpy as np
import pdb

from onlinetune.knobs import logger
from onlinetune.safe.subdomainbo import AscentLineBO
from onlinetune.safe.utils import  maximize, minimize, plot_colored_region
from onlinetune.knobs import gen_continuous


class TrustRegionSafetyMixin(AscentLineBO):
    def initialize(self, **kwargs):

        super(TrustRegionSafetyMixin, self).initialize(**kwargs)
        self.bo_expander_ratio = 6

    def _tr_add_data(self, data):
        super(TrustRegionSafetyMixin, self)._tr_add_data(data)
        self._tr_compute_safe_region()

    def _tr_solver_init(self, tr_domain):
        super(TrustRegionSafetyMixin, self)._tr_solver_init(tr_domain)
        logger.info("Init trust region")
        self._tr_compute_safe_region()


    def _tr_solver_stop(self):
        self._tr_compute_safe_region()

        if (~self._tr_mask_safe).all():
            count = 0
            beta_pre = self.s[0]._beta_cached
            while (~self._tr_mask_safe).all():
                if self.s[0]._beta_cached > 0.5/0.6:
                    self.s[0]._beta_cached = self.s[0]._beta_cached * 0.6
                    self._tr_compute_safe_region()
                count = count + 1
                if count > 10:
                    logger.info("No safe condidates, stop!")
                    return True
                    break
            if count > 0:
                logger.info("reduce beta from {} to {}".format(beta_pre, self.s[0]._beta_cached))


        return super()._tr_solver_stop()


    def filter_same_interger(self, X_embebed):
        mask_discard = np.ones(X_embebed.shape[0], dtype='bool').reshape(-1, 1)
        delta_y = np.zeros(X_embebed.shape[0], dtype='float64').reshape(-1, 1)
        for i in range(X_embebed.shape[0]):
            action = X_embebed[i]
            knob = gen_continuous(action)

            if knob in self.knobL:
                mask_discard[i] = 0
                delta_y[i] = self.ydeltaL[self.knobL.index(knob)]

        self.integer_mask = mask_discard
        self.s[0].delta_y = delta_y
        return mask_discard



    def _tr_compute_safe_region(self):
        self._tr_X_eval = self._add_context(self._tr_domain.points)
        self._tr_domain.X_eval_embedded = self._tr_X_eval
        self.X_eval_embedded = self._tr_X_eval
        mask_discard = self.filter_same_interger(self._tr_X_eval).flatten()
        s_lcb, s_ucb = self.get_joined_constrained_cb(self._tr_X_eval)
        s_lcb, s_ucb = s_lcb[:, 0], s_ucb[:, 0]
        self._tr_mask_safe = np.logical_and(s_ucb <= 0, mask_discard)

        self.safe_candidate = self._tr_mask_safe.sum()


    def _tr_solver_step(self):
        if self.bo_expander_ratio > 1.5:
            self.bo_expander_ratio = self.bo_expander_ratio - 0.5
        self._tr_compute_safe_region()

        if (~self._tr_mask_safe).all():
            if self.context_change:
                logger.info("Context change, use re-predicted x0.")
                x_0 = self.get_data_center_from_context()
                return self._add_context(x_0.reshape(1, -1)).flatten()
            else:

                count = 0
                beta_pre = self.s[0]._beta_cached
                while (~self._tr_mask_safe).all():
                    self._tr_compute_safe_region()
                    if self.s[0]._beta_cached > 0.5 / 0.6 :
                        self.s[0]._beta_cached = self.s[0]._beta_cached * 0.6
                    count = count + 1
                    if count > 10:
                        logger.warning("No safe point found, choosing current region center.")
                        return self._add_context(self._tr_domain.x0).flatten()
                if count > 0:
                    logger.info("reduce beta from {} to {}".format(beta_pre, self.s[0]._beta_cached))

        logger.info("Safe candidates: {}, account for {}".format(self._tr_mask_safe.sum(), self._tr_mask_safe.sum()/np.size(self._tr_mask_safe)))
        x_acq,_ , x_acq_safe, _ = maximize(self.global_acquisition, self._tr_X_eval, self._tr_mask_safe, both=True)

        if (x_acq_safe == x_acq).all() and np.random.rand() < 0.6:
            self._point_type_addition = "-global_acq"
            logger.info("-global_acq")
            return x_acq_safe

        X_eval_retract = self._tr_domain.x0 + (x_acq - self._tr_domain.x0) * np.linspace(0,1, 200).reshape(-1,1)
        s_lcb_retract, s_ucb_retract = self.get_joined_constrained_cb(X_eval_retract)
        s_lcb, s_ucb = s_lcb_retract[:,0], s_ucb_retract[:,0]

        x_expander = x_acq_safe#self._tr_domain.x0
        for x, u in zip(X_eval_retract, s_ucb):
            if u > 0:
                break
            x_expander = x

        # uncertainty sampling between expander and safe acquisition point
        std_acq_safe = self.s[0].beta * self.s[0].std(x_acq_safe)[0]  # scale with beta from the safety models to have a fair comparision ??
        lcb_expander, ucb_expander = self.get_joined_constrained_cb(x_expander.reshape(1, -1))
        knob_expander = gen_continuous(x_expander)

        logger.info ("delta:{}".format(np.asscalar(self.bo_expander_ratio * std_acq_safe - (ucb_expander - lcb_expander))))
        if np.asscalar(self.bo_expander_ratio*std_acq_safe - (ucb_expander - lcb_expander)) >= 0\
                or knob_expander in self.knobL\
                or np.random.rand() > 0.5:
            self._point_type_addition = "-safe_acq"
            logger.info("-safe_acq")
            return x_acq_safe
        else:
            self._point_type_addition = "-expander"
            logger.info("-expander")
            return x_expander

    def _tr_solver_best(self):
        if (~self._tr_mask_safe).all():
            return self._tr_domain.x0
        return  maximize(self.model.mean, self._tr_X_eval, self._tr_mask_safe)[0]

#@assign_config(SafeOptConfigMixin)
class LineSafetyMixin():
    def initialize(self, **kwargs):

        super(LineSafetyMixin, self).initialize(**kwargs)
        self._line_constraints_summary = {}
        self.bo_expander_ratio = 6

    def plot_line(self, axis, steps=300):
        X_eval, X_eval_embedded, X_data = super().plot_line(axis, steps=steps)

        # plot constraint models
        s_lcb, s_ucb = self.get_joined_constrained_cb(X_eval_embedded)
        axis.axhline(0, color='red', alpha=0.5)
        axis.fill_between(X_eval, s_lcb.flatten(), s_ucb.flatten(), color='red', alpha=0.3)
        for s in self.s:
            axis.plot(X_eval, s.mean(X_eval_embedded), color='red')


        plot_colored_region(axis, self._line_domain.l, self._line_optimistic_boundary_l, color='black')
        plot_colored_region(axis, self._line_optimistic_boundary_l, self._line_boundary_l, color='orange')

        plot_colored_region(axis, self._line_boundary_l, self._line_boundary_r, color='green')

        # if self._line_constraints_summary['discard_r']:
        plot_colored_region(axis, self._line_optimistic_boundary_r, self._line_domain.u, color='black')
        plot_colored_region(axis, self._line_boundary_r, self._line_optimistic_boundary_r, color='orange')
        # else:
        #     color_region_helper(axis, self._line_constraints_summary['boundary_r'], self._line_domain.u, color='orange')


        # scatter safety data
        data_s = np.empty(shape=(len(self._line_data), self.num_constraints))
        for i,p in enumerate(self._line_data):
            data_s[i] = p['s']

        for data_s_sub in data_s.T:
            axis.scatter(X_data, data_s_sub, color='red', marker='x')

        return X_eval, X_eval_embedded, X_data

    def _compute_boundary_helper(self, triple_iterator):
        """
        walks through the iterator and determines safety boundary and optimistic boundary
        :param triple_iterator: iterator over x, lcb, ucb, where x[0] is safe
        :return:
        """

        boundary, optimistic_boundary = None, None
        boundary_fixed = False
        for x, l, u in triple_iterator:
            if not boundary_fixed and u < 0:
                boundary = x

            # first time lower bound crosses threshold, we fix the boundary
            if u > 0:
                boundary_fixed = True

            # keep updating the optimistic boundary, until it is above threshold for first time
            optimistic_boundary = x
            if l > 0:
                break

        return boundary, optimistic_boundary


    def _line_add_data(self, data):
        super(LineSafetyMixin, self)._line_add_data(data)
        self._line_compute_safe_region()

    def _line_solver_init(self, line_domain):
        super(LineSafetyMixin, self)._line_solver_init(line_domain)
        logger.info("Init line region")
        self._line_compute_safe_region()

    def _line_max_ucb(self):
        res = maximize(self.model.ucb, self._line_X_eval_embedded, self._line_mask_safe)[1]
        return - res

    def _line_solver_stop(self):
        count = 0
        beta_pre = self.s[0]._beta_cached
        while (~self._line_mask_safe).all():
            if self.s[0]._beta_cached > 0.5 / 0.6:
                self.s[0]._beta_cached = self.s[0]._beta_cached * 0.6
                self._line_compute_safe_region()
            count = count + 1
            if count > 10 or self.s[0]._beta_cached<1:
                logger.info("No safe condidates, stop!")
                return True
                break
        if count > 0:
            logger.info("reduce beta from {} to {}".format(beta_pre, self.s[0]._beta_cached))


        return super()._line_solver_stop()


    def _line_compute_safe_region(self):
        n_safety_grid = 300
        # compute boundary and optimistic boundary
        X_eval = np.linspace(self._line_domain.l, self._line_domain.u, n_safety_grid)
        X_eval_embedded = self._line_domain.embed_in_domain(X_eval.reshape(-1, 1))

        # mask for point left/right of center
        mask_l = X_eval <= self._line_domain.c
        mask_r = X_eval >= self._line_domain.c

        # discard mask
        #mask_discard = np.ones(n_safety_grid, dtype='bool').reshape(-1, 1)
        mask_discard = self.filter_same_interger(X_eval_embedded).reshape(-1 ,1)
        self.X_eval_embedded = X_eval_embedded
        self._line_domain.X_eval_embedded = X_eval_embedded
        lcb, ucb = self.model.ci(X_eval_embedded)
        max_lcb = np.max(lcb)

        # compute mean, var for all models
        s_lcb, s_ucb = self.get_joined_constrained_cb(X_eval_embedded)

        # take minimum lcb, ucb over all safety models
        if np.sum(mask_l) == 0:
            boundary_l = optimistic_boundary_l = self._line_domain.c
        else:
            boundary_l, optimistic_boundary_l = self._compute_boundary_helper(
                zip(X_eval[mask_l][::-1], s_lcb[mask_l][::-1], s_ucb[mask_l][::-1]))
        if np.sum(mask_r) == 0:
            boundary_r = optimistic_boundary_r = self._line_domain.c
        else:
            boundary_r, optimistic_boundary_r = self._compute_boundary_helper(
                zip(X_eval[mask_r], s_lcb[mask_r], s_ucb[mask_r]))

        if boundary_l is None or boundary_r is None:
            boundary_l = boundary_r = self._line_domain.c

        boundary_region_l_mask = np.logical_and(X_eval <= boundary_l, X_eval >= optimistic_boundary_l)
        boundary_region_r_mask = np.logical_and(X_eval >= boundary_r, X_eval <= optimistic_boundary_r)

        # compute discard region
        discard_left = np.sum(boundary_region_l_mask) > 0 and max_lcb > np.max(ucb[boundary_region_l_mask])
        if discard_left:
            mask_discard = np.logical_and(mask_discard, X_eval > optimistic_boundary_l)

        discard_right = np.sum(boundary_region_r_mask) > 0 and max_lcb > np.max(ucb[boundary_region_r_mask])
        if discard_right:
            mask_discard = np.logical_and(mask_discard, X_eval < optimistic_boundary_r)

        # compute safe region
        mask_safe = np.logical_and(X_eval >= boundary_l, X_eval <= boundary_r)
        #logger.info("mask_safe before discard {}".format(mask_safe.sum()) )
        mask_safe = np.logical_and(mask_safe, mask_discard)
        #print ((self._line_domain.indexL, self._line_domain.l, self._line_domain.u))
        #mask_safe[self._line_domain.indexL] = 0
        # save some info for plotting
        self._line_X_eval = X_eval
        self._line_X_eval_embedded = X_eval_embedded
        self._line_mask_safe = mask_safe
        self._line_boundary_l = boundary_l
        self._line_boundary_r = boundary_r
        self._line_optimistic_boundary_l = optimistic_boundary_l
        self._line_optimistic_boundary_r = optimistic_boundary_r
        self._line_discard_l = discard_left
        self._line_discard_r = discard_right

    def _line_solver_best(self):
        if (~self._line_mask_safe).all():
            return self._line_domain.x0
        x, res = maximize(self.model.mean, self._line_X_eval_embedded, self._line_mask_safe)
        return x

    def _line_solver_step(self):
        if self.bo_expander_ratio > 1.5:
            self.bo_expander_ratio = self.bo_expander_ratio - 0.5

        self._line_compute_safe_region()
        if (~self._line_mask_safe).all():
            self._point_type_addition = "-initial"

            if self.context_change:
                logger.info("Context change, use re-predicted x0.")
                x_0 = self.get_data_center_from_context()
                return self._add_context(x_0.reshape(1, -1)).flatten()
            else:
                #logger.warning("No safe point found, choosing current line center.")
                #return self._add_context(self._line_domain.x0).flatten()
                count = 0
                beta_pre = self.s[0]._beta_cached
                while (~self._line_mask_safe).all():
                    self._line_compute_safe_region()
                    if self.s[0]._beta_cached  > 0.5 / 0.6:
                        self.s[0]._beta_cached = self.s[0]._beta_cached * 0.6
                    count = count + 1
                    if count > 10:
                        logger.info("reduce beta from {} to {}".format(beta_pre, self.s[0]._beta_cached))
                        logger.warning("No safe point found, choosing current line center.")
                        return self._add_context(self._line_domain.x0).flatten()

                if count > 0:
                    logger.info("reduce beta from {} to {}".format(beta_pre, self.s[0]._beta_cached))

        logger.info("Safe candidates: {}, account for {}".format(self._line_mask_safe.sum(), self._tr_mask_safe.sum()/np.size(self._tr_mask_safe)))
        # compute acq points
        try:
            x_acq, _, x_acq_safe, _, _ = maximize(self.global_acquisition, self._line_X_eval_embedded, self._line_mask_safe, both=True, index=True)
        except:
           pdb.set_trace()

        # x_acq_safe, res_safe = maximize(self.global_acquisition, self._line_X_eval_embedded, self._line_mask_safe)
        # compute expander in direction of unconstrained acquisition point
        if self._line_domain.project_on_line(x_acq) <= self._line_domain.c:
            x_expander = self._line_domain.embed_in_domain(self._line_boundary_l)[0]
        else:
            x_expander = self._line_domain.embed_in_domain(self._line_boundary_r)[0]

        # if unconstraint acquisition point is same as safe acquisition point, return it
        if (x_acq == x_acq_safe).all() and np.random.rand() < 0.6:
            self._point_type_addition = "-global_acq"
            #logger.info("res_safe: {}".format(res_safe))
            return x_acq_safe

        # uncertainty sampling between expander and safe acquisition point
        # scale with beta from the safety models to have a fair comparision ??
        std_acq_safe = self.s[0].beta*self.s[0].std(x_acq_safe)[0]
        lcb_expander, ucb_expander = self.get_joined_constrained_cb(x_expander.reshape(1,-1))

        knob_expander = gen_continuous(x_expander)

        logger.info ("delta:{}".format(np.asscalar(self.bo_expander_ratio * std_acq_safe - (ucb_expander - lcb_expander))))
        if np.asscalar(self.bo_expander_ratio*std_acq_safe - (ucb_expander - lcb_expander)) >= 0\
                or (knob_expander in self.knobL)\
                or np.random.rand() > 0.5:
            self._point_type_addition = "-safe_acq"
            logger.info("safe_acq")
            return x_acq_safe
        else:
            self._point_type_addition = "-expander"
            logger.info("expander")
            return x_expander

class SafetyMixin(LineSafetyMixin, TrustRegionSafetyMixin):
    pass


class SafeAscentLineBO(SafetyMixin, AscentLineBO):
    pass