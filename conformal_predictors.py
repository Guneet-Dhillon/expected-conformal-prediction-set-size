import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor

from base_conformal_predictor import BaseConformalPredictor

class LPRegressionConformalPredictor(BaseConformalPredictor):
    def __init__(self, y, p=1., model='randomforest'):
        super().__init__('continuous', 'continuous')

        # Set the order of the norm
        self.p = p

        # Set the minimum and the maximum non-conformity scores
        self.r_min = 0.
        self.r_max = (y.max() - y.min()) ** p

        # The machine learning model
        if model == 'randomforest':
            self.model = RandomForestRegressor()

        # The multiplicative factor is known
        self.get_prediction_set_size_estimates = \
            self.get_prediction_set_size_estimates_known_multiplicative_factor

    def get_model_prediction(self, X):
        # Compute the machine learning model prediction
        yh = self.model.predict(X)
        return yh

    def get_nonconformity_score(self, X, y):
        # Compute the non-conformity score
        yh = self.get_model_prediction(X)
        r = np.abs(yh - y) ** self.p
        return r

    def get_prediction(self, X_test):
        # Compute the conformal prediction set
        yh_test = self.get_model_prediction(X_test)
        C = [
            yh_test - (self.threshold ** (1. / self.p)),
            yh_test + (self.threshold ** (1. / self.p))
        ]
        return C

    def integrate_multiplicative_factor(self, r):
        # Compute the integral of the multiplicative factor over each
        # (r[i], r[i + 1]] region
        int_mf = 2. * ((r[1 :] ** (1. / self.p)) - (r[: -1] ** (1. / self.p)))
        return int_mf

class ZeroOneClassificationConformalPredictor(BaseConformalPredictor):
    def __init__(self, num_classes, model='randomforest'):
        super().__init__('discrete', 'discrete')

        # Set the number of classes
        self.num_classes = num_classes

        # Set the space of non-conformity scores
        self.r_space = np.array([0., 1.])

        # The machine learning model
        if model == 'randomforest':
            self.model = RandomForestClassifier()

        # The multiplicative factor is known
        self.get_prediction_set_size_estimates = \
            self.get_prediction_set_size_estimates_known_multiplicative_factor

    def get_model_prediction(self, X):
        # Compute the machine learning model prediction
        yh = self.model.predict(X)
        return yh

    def get_nonconformity_score(self, X, y):
        # Compute the non-conformity score
        yh = self.get_model_prediction(X)
        r = (yh != y).astype(float)
        return r

    def get_prediction(self, X_test):
        # Compute the conformal prediction set
        n_test = X_test.shape[0]
        if self.threshold == 0.:
            yh_test = self.get_model_prediction(X_test)
            C = np.zeros((n_test, self.num_classes), bool)
            C[np.arange(n_test), yh_test] = 1.
        else:
            C = np.ones((n_test, self.num_classes), bool)
        return C

    def multiplicative_factor(self, r):
        # Compute the multiplicative factor for each non-conformity score
        mf = (r == 0.) + ((self.num_classes - 1) * (r == 1.))
        return mf

class QuantileRegressionConformalPredictor(BaseConformalPredictor):
    def __init__(self, y, y_num=100, model='randomforest'):
        super().__init__('continuous', 'continuous')

        # Set the minimum and the maximum labels
        self.y_min = y.min()
        self.y_max = y.max()

        # Set the number of labels to use for the expected prediction set size
        # estimates
        self.y_num = y_num

        # The machine learning model
        if model == 'randomforest':
            self.model = RandomForestQuantileRegressor()

        # The multiplicative factor is unknown
        self.get_prediction_set_size_estimates = \
            self.get_prediction_set_size_estimates_unknown_multiplicative_factor

    def set_calibration(self, X_cal, y_cal, alpha):
        # Set the significance level
        self.alpha = alpha
        super().set_calibration(X_cal, y_cal, alpha)

    def get_model_prediction(self, X):
        # Compute the machine learning model prediction
        quantiles = [self.alpha / 2., 1. - (self.alpha / 2.)]
        Yh = self.model.predict(X, quantiles=quantiles)
        return Yh

    def get_nonconformity_score(self, X, y):
        # Compute the non-conformity score
        Yh = self.get_model_prediction(X)
        r = np.maximum(Yh[:, 0] - y, y - Yh[:, 1])
        return r

    def get_nonconformity_scores(self, X):
        # Compute all labels to use
        ys, self.y_width = np.linspace(
            self.y_min, self.y_max, self.y_num + 1, retstep=True
        )
        ys += (self.y_width / 2.)
        ys = ys[: -1]

        # Compute the non-conformity scores for all labels
        Yh = self.get_model_prediction(X)
        R = []
        for y in ys:
            r = np.maximum(Yh[:, 0] - y, y - Yh[:, 1])
            R.append(r[:, np.newaxis])
        R = np.hstack(R)
        return R

    def get_prediction(self, X_test):
        # Compute the conformal prediction set
        Yh_test = self.get_model_prediction(X_test)
        C = [Yh_test[:, 0] - self.threshold, Yh_test[:, 1] + self.threshold]
        return C

class LACClassificationConformalPredictor(BaseConformalPredictor):
    def __init__(self, num_classes, model='randomforest'):
        super().__init__('discrete', 'continuous')

        # Set the number of classes
        self.num_classes = num_classes

        # The machine learning model
        if model == 'randomforest':
            self.model = RandomForestClassifier()

        # The multiplicative factor is unknown
        self.get_prediction_set_size_estimates = \
            self.get_prediction_set_size_estimates_unknown_multiplicative_factor

    def get_model_prediction(self, X):
        # Compute the machine learning model prediction
        Yh = self.model.predict_proba(X)
        return Yh

    def get_nonconformity_score(self, X, y):
        # Compute the non-conformity score
        n = X.shape[0]
        Yh = self.get_model_prediction(X)
        r = 1. - Yh[np.arange(n), y]
        return r

    def get_nonconformity_scores(self, X):
        # Compute the non-conformity scores for all labels
        Yh = self.get_model_prediction(X)
        R = 1. - Yh
        return R

    def get_prediction(self, X_test):
        # Compute the conformal prediction set
        R_test = self.get_nonconformity_scores(X_test)
        C = R_test <= self.threshold
        return C

class APSClassificationConformalPredictor(BaseConformalPredictor):
    def __init__(self, num_classes, model='randomforest'):
        super().__init__('discrete', 'continuous')

        # Set the number of classes
        self.num_classes = num_classes

        # The machine learning model
        if model == 'randomforest':
            self.model = RandomForestClassifier()

        # The multiplicative factor is unknown
        self.get_prediction_set_size_estimates = \
            self.get_prediction_set_size_estimates_unknown_multiplicative_factor

    def get_model_prediction(self, X):
        # Compute the machine learning model prediction
        Yh = self.model.predict_proba(X)
        return Yh

    def get_nonconformity_score(self, X, y):
        # Compute the non-conformity score
        n = X.shape[0]
        Yh = self.get_model_prediction(X)
        yh = Yh[np.arange(n), y]
        r = \
            ((yh[:, np.newaxis] < Yh) * Yh).sum(1) \
            + (np.random.uniform(size=n) * yh)
        return r

    def get_nonconformity_scores(self, X):
        # Compute the non-conformity scores for all labels
        n = X.shape[0]
        Yh = self.get_model_prediction(X)
        R = []
        for y in range(self.num_classes):
            yh = Yh[:, y]
            r = \
                ((yh[:, np.newaxis] < Yh) * Yh).sum(1) \
                + (np.random.uniform(size=n) * yh)
            R.append(r[:, np.newaxis])
        R = np.hstack(R)
        return R

    def get_prediction(self, X_test):
        # Compute the conformal prediction set
        R_test = self.get_nonconformity_scores(X_test)
        C = R_test <= self.threshold
        return C

