import math
import numpy as np
import scipy

class BaseConformalPredictor:
    def __init__(self, label_type, nonconformity_score_type):
        # The label and non-conformity score types (continuous or discrete)
        self.label_type = label_type
        self.nonconformity_score_type = nonconformity_score_type

    def set_train(self, X_train, y_train):
        # Train the machine learning model
        self.model.fit(X_train, y_train)

    def set_calibration(self, X_cal, y_cal, alpha):
        # Compute the variables dependent on the calibration dataset size and
        # the significance level
        self.n = X_cal.shape[0]
        self.n_alpha = math.ceil((1. - alpha) * (self.n + 1)) - 1

        # Compute the conformal threshold
        r_cal = self.get_nonconformity_score(X_cal, y_cal)
        r_cal.sort()
        r = np.append(r_cal, np.inf)
        self.threshold = r[self.n_alpha]

    def get_prediction_set_size_estimates_known_multiplicative_factor(
        self, X_access, y_access, gamma
    ):
        # Compute the variables dependent on the accessible dataset size
        k = X_access.shape[0]
        delta = math.sqrt(math.log(2. / gamma) / (2. * k))

        # Compute the accessible dataset non-conformity scores
        r_access = self.get_nonconformity_score(X_access, y_access)

        # The space of non-conformity scores over which to integrate/sum
        if self.nonconformity_score_type == 'continuous':
            # The accessible dataset non-confomrity scores, with r_min and
            # r_max, because the empirical distribution of non-conformity scores
            # is a step function over this space
            r_access.sort()
            r = np.insert(r_access, 0, self.r_min)
            r = np.append(r, self.r_max)
        elif self.nonconformity_score_type == 'discrete':
            # The space of non-conformity scores
            r = self.r_space

        # Compute the desired epirical distribution of non-conformity scores
        p = (r_access[np.newaxis, :] < r[:, np.newaxis]).mean(1)
        p_lower = (p - delta).clip(0., 1.)
        p_upper = (p + delta).clip(0., 1.)

        # Compute the binomial CDF terms
        binom_cdfs = scipy.stats.binom.cdf(
            self.n_alpha, self.n, [p, p_upper, p_lower]
        )

        # Compute the expected prediction set size estimates
        if self.nonconformity_score_type == 'continuous':
            # Compute the integral over the non-conformity score space
            int_mf = self.integrate_multiplicative_factor(r)
            sizes = (binom_cdfs[:, 1 :] * int_mf[np.newaxis, :]).sum(1)
        elif self.nonconformity_score_type == 'discrete':
            # Compute the sum over the non-conformity score space
            mf = self.multiplicative_factor(r)
            sizes = (binom_cdfs * mf[np.newaxis, :]).sum(1)

        # Return the expected prediction set size estimates
        return sizes

    def get_prediction_set_size_estimates_unknown_multiplicative_factor(
        self, X_access, y_access, gamma
    ):
        # Compute the variables dependent on the accessible dataset size
        k = X_access.shape[0]
        delta = math.sqrt(math.log(2. / gamma) / (2. * k))

        # Compute the accessible dataset non-conformity scores
        r_access = self.get_nonconformity_score(X_access, y_access)

        # Compute the accessible dataset non-conformity scores for all labels
        R_access = self.get_nonconformity_scores(X_access)

        # Compute the desired epirical distribution of non-conformity scores
        p = (
            r_access[np.newaxis, np.newaxis, :] < R_access[:, :, np.newaxis]
        ).mean(2)
        p_lower = (p - delta).clip(0., 1.)
        p_upper = (p + delta).clip(0., 1.)

        # Compute the binomial CDF terms
        binom_cdfs = scipy.stats.binom.cdf(
            self.n_alpha, self.n, [p, p_upper, p_lower]
        )

        # Compute the expected prediction set size estimates
        if self.label_type == 'continuous':
            # Compute the mean over the accessible dataset, and the integral
            # over the label space
            sizes = self.y_width * binom_cdfs.mean(1).sum(1)
        elif self.label_type == 'discrete':
            # Compute the mean over the accessible dataset, and the sum over the
            # label space
            sizes = binom_cdfs.mean(1).sum(1)

        # Return the expected prediction set size estimates
        return sizes

