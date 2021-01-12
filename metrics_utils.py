import os
from tensorflow import where as tfwhere, zeros_like as tfzeros_like
from tensorflow.keras.metrics import (Metric as MetricTfKeras, Accuracy as AccuracyTfKeras,
                                      FalsePositives, TruePositives, TrueNegatives, FalseNegatives, Precision, Recall)
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.utils import metrics_utils as metrics_utils_tf_keras
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import init_ops, math_ops
import numpy as np
os.environ['SM_FRAMEWORK'] = 'tf.keras'  # will tell segmentation models to use tensorflow's keras
from segmentation_models.base import Metric as MetricSM, functional
import time


SMOOTH = 1e-5
assert SMOOTH <= 1e-5

# 0.5 is default prediction threshold for most metrics which use a threshold value
# and the threshold value is also effectively ignored for one hot metrics
global_threshold = 0.5
assert 0.0 <= global_threshold <= 1.0

# In summary, to achieve one hot metrics:
# 1. For a metric class who via definition inherits tf.keras.metrics.Metric or tf.keras.metric.MeanMetricWrapper, for
#    one hot conversion in which this metric class is inherited by a sub-class one hot version:
#    -  in tf2, place 1H at __ call __ method or update_state method (or both), followed by corresponding super().
#    -  in tf1, place 1H at update_state method, followed by corresponding super().
# 2. For a metric class who via definition does NOT inherit tf.keras.metrics.Metric or tf.keras.metric.MeanMetricWrapper
#    (e.g., instead, inherits segmentation_models.metrics.Metric), for one hot conversion in which this metric class is
#    inherited by a sub-class one hot version (note, the class instance will be treated as a function and automatically
#    wrapped with tf.keras.metrics.MeanMetricWrapper during model.compile) :
#    -  in tf2, place 1H at __ call __ method, followed by corresponding super(). Interestingly in tf2, the result is
#   independent of whether or not the update_state method result has a return statement.
#    -  in tf1, place 1H at __ call __ method, followed by corresponding super().


# one hot classes are intended to act as pass-throughs. 1H (argmax) proceeds after thresholding, as done in infer.

# `MeanMetricWrapper` inheritance in custom metric: do not need to remove 'return' from `def update_state` in tf2.0
class OneHotAccuracyTfKeras(AccuracyTfKeras):
    def __init__(self, name='accuracy_tfkeras_1H', dtype=None):
        super().__init__(name=name, dtype=dtype)

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, global_threshold), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)


class OneHotFalseNegatives(FalseNegatives):
    def __init__(self, thresholds=None, name='FN_1H', dtype=None):
        super().__init__(
            thresholds=thresholds,
            name=name,
            dtype=dtype
        )

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)


class OneHotFalsePositives(FalsePositives):
    def __init__(self, thresholds=None, name='FP_1H', dtype=None):
        super().__init__(
            thresholds=thresholds,
            name=name,
            dtype=dtype
        )

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)


class OneHotTrueNegatives(TrueNegatives):
    def __init__(self, thresholds=None, name='TN_1H', dtype=None):
        super().__init__(
            thresholds=thresholds,
            name=name,
            dtype=dtype
        )

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)


class OneHotTruePositives(TruePositives):
    def __init__(self, thresholds=None, name='TP_1H', dtype=None):
        super().__init__(
            thresholds=thresholds,
            name=name,
            dtype=dtype
        )

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))    # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)


class OneHotPrecision(Precision):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='precision_1H',
                 dtype=None):
        super().__init__(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype)

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)


class OneHotRecall(Recall):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='recall_1H',
                 dtype=None):
        super().__init__(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype)

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location# based on tf.keras binary_accuracy
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)


# based on Keras/tf.keras precision and recall class definitions found at (depending on import source):
# keras: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/metrics.py#L1154
# tf.keras: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/metrics.py#L1134
class FBetaScore(MetricTfKeras):
    """Abstract base class for F1Score.
    For additional information, see the
    following: https://en.wikipedia.org/wiki/F1_score#Definition

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `top_k` is set, we'll calculate precision as how often on average a class
    among the top-k classes with the highest predicted values of a batch entry is
    correct and can be found in the label for that entry.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold and/or in the
    top-k highest predictions, and computing the fraction of them for which
    `class_id` is indeed a correct label."""

    '''
    Arguments
        beta: The F-measure was derived so that F_β "measures the effectiveness of
            retrieval with respect to a user who attaches β times as much importance to recall as precision".
            beta=1 gives F_1 score, and is also known as the Sørensen–Dice coefficient or Dice similarity
            coefficient (DSC).
        thresholds: (Optional) A float value or a python list/tuple of float
            threshold values in [0, 1]. A threshold is compared with prediction
            values to determine the truth value of predictions (i.e., above the
            threshold is `true`, below is `false`). One metric value is generated
            for each threshold value. If neither thresholds nor top_k are set, the
            default is to calculate precision with `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    '''

    def __init__(self,
                 beta=1,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        name = name or str('f' + str(beta) + 'score')
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.beta = beta
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils_tf_keras.NEG_INF
        self.thresholds = metrics_utils_tf_keras.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # for tf v1, use 'return metrics_...'. for tf v2, use 'metrics_...' (for inherited keras/tf.keras Metric class)
        metrics_utils_tf_keras.update_confusion_matrix_variables(
            {
                metrics_utils_tf_keras.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils_tf_keras.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils_tf_keras.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        denominator = ((1 + self.beta * self.beta) * self.true_positives + self.beta * self.beta * self.false_negatives
                       + self.false_positives)
        numerator = (1 + self.beta * self.beta) * self.true_positives
        result = math_ops.div_no_nan(numerator, denominator)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'beta': self.beta,
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OneHotFBetaScore(FBetaScore):
    def __init__(self,
                 beta=1,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        name = name or str('f' + str(beta) + 'score_1H')
        super().__init__(
            beta=beta,
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype)

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)


# based on Keras/tf.keras precision and recall class definitions found at (depending on import source):
# keras: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/metrics.py#L1154
# tf.keras: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/metrics.py#L1134
class IoUScore(MetricTfKeras):
    """Computes the mean Intersection-Over-Union metric.
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation, which first computes the IOU for each semantic class and then
    computes the average over classes. IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `top_k` is set, we'll calculate precision as how often on average a class
    among the top-k classes with the highest predicted values of a batch entry is
    correct and can be found in the label for that entry.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold and/or in the
    top-k highest predictions, and computing the fraction of them for which
    `class_id` is indeed a correct label."""

    '''
    # Arguments
        thresholds: (Optional) A float value or a python list/tuple of float
            threshold values in [0, 1]. A threshold is compared with prediction
            values to determine the truth value of predictions (i.e., above the
            threshold is `true`, below is `false`). One metric value is generated
            for each threshold value. If neither thresholds nor top_k are set, the
            default is to calculate precision with `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    '''

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='iou_score',
                 dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils_tf_keras.NEG_INF
        self.thresholds = metrics_utils_tf_keras.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # for tf v1, use 'return metrics_...'. for tf v2, use 'metrics_...' (for inherited keras/tf.keras Metric class)
        metrics_utils_tf_keras.update_confusion_matrix_variables(
            {
                metrics_utils_tf_keras.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils_tf_keras.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils_tf_keras.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        denominator = (self.true_positives + self.false_negatives + self.false_positives)
        numerator = self.true_positives
        result = math_ops.div_no_nan(numerator, denominator)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OneHotIoUScore(IoUScore):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='iou_score_1H',
                 dtype=None):
        super().__init__(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype)

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)


# VERSION 2 CLASSBINARYACCURACY METHOD, BASED ON KERAS PACKAGE -- ACCUMULATED OVER EPOCH (inherit KERAS.METRIC)
# based on Keras/tf.keras precision and recall class definitions found at (depending on import source):
# keras: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/metrics.py#L1154
# tf.keras: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/metrics.py#L1134
class ClassBinaryAccuracyTfKeras(MetricTfKeras):
    r"""
    .. math:: Binary Accuracy = (TN + TP)/(TN+TP+FN+FP) = Number of correct assessments/Number of all assessments,
    for given class for more than one class input, output becomes mean accuracy (similar but not same as categorical)
    # Arguments
        thresholds: (Optional) A float value or a python list/tuple of float
            threshold values in [0, 1]. A threshold is compared with prediction
            values to determine the truth value of predictions (i.e., above the
            threshold is `true`, below is `false`). One metric value is generated
            for each threshold value. If neither thresholds nor top_k are set, the
            default is to calculate precision with `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
    """

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='class_all_binary_accuracy_tfkeras',
                 dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils_tf_keras.NEG_INF
        self.thresholds = metrics_utils_tf_keras.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # for tf v1, use 'return metrics_...'. for tf v2, use 'metrics_...' (for inherited keras/tf.keras Metric class)
        metrics_utils_tf_keras.update_confusion_matrix_variables(
            {
                metrics_utils_tf_keras.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils_tf_keras.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils_tf_keras.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
                metrics_utils_tf_keras.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        denominator = (self.true_positives + self.false_negatives + self.false_positives + self.true_negatives)
        numerator = self.true_positives + self.true_negatives
        result = math_ops.div_no_nan(numerator, denominator)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OneHotClassBinaryAccuracyTfKeras(ClassBinaryAccuracyTfKeras):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='class_all_binary_accuracy_tfkeras_1H',
                 dtype=None):
        super().__init__(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype)

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction, **kwargs):
        prediction = tfwhere(math_ops.greater(prediction, self.thresholds), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices,
                                      K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot, **kwargs)


# VERSION 1 CLASSBINARYACCURACY METHOD, BASED ON SEGMENTATION_MODELS PACKAGE -- AVERAGED OVER EPOCH
# adapted from: s_m.IOUScore() from github.com/qubvel/segmentation_models/blob/master/segmentation_models/metrics.py
class ClassBinaryAccuracySM(MetricSM):
    r"""
    .. math:: Binary Accuracy = (TN + TP)/(TN+TP+FN+FP) = Number of correct assessments/Number of all assessments,
    for given class for more than one class input, output becomes mean accuracy (similar but not same as categorical)
    Args:
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round
    Returns:
       A callable ``class_binary_accuracy`` instance. Can be used in ``model.compile(...)`` function.
    Example:
    .. code:: python
        metric = ClassBinaryAccuracy()
        model.compile('SGD', loss=loss, metrics=[metric])
    """

    def __init__(
            self,
            class_weights=None,
            class_indexes=None,
            threshold=None,
            per_image=False,
            smooth=SMOOTH,
            name=None
    ):
        self.name = name or 'class_all_binary_accuracy_sm'
        super().__init__(name=self.name)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):

        backend = self.submodules['backend']

        gt, pr = functional.gather_channels(gt, pr, indexes=self.class_indexes, **self.submodules)
        pr = functional.round_if_needed(pr, self.threshold, **self.submodules)
        axes = functional.get_reduce_axes(self.per_image, **self.submodules)

        # score calculation (assumed pr are 1-hot in practice)
        tp = backend.sum(gt * pr, axis=axes)
        fp = backend.sum(pr, axis=axes) - tp
        fn = backend.sum(gt, axis=axes) - tp
        tn = backend.sum((-gt + 1) * (-pr + 1), axis=axes)
        score = (tp + tn) / (tp + tn + fp + fn + self.smooth)
        # score is averaged over whole batch here (unlike Keras, where score is accumulated over batch)
        score = functional.average(score, self.per_image, self.class_weights, **self.submodules)

        return score


class OneHotClassBinaryAccuracySM(ClassBinaryAccuracySM):
    def __init__(
            self,
            class_weights=None,
            class_indexes=None,
            threshold=None,
            per_image=False,
            smooth=SMOOTH,
            name=None
    ):
        self.name = name or 'class_all_binary_accuracy_sm_1H'
        super().__init__(
            class_weights=class_weights,
            class_indexes=class_indexes,
            threshold=threshold,
            per_image=per_image,
            smooth=smooth,
            name=self.name)

    # call redirects to parent class following one hot conversion
    def __call__(self, groundtruth, prediction):
        prediction = tfwhere(math_ops.greater(prediction, self.threshold), prediction, tfzeros_like(prediction))  # based on tf.keras binary_accuracy
        prediction_onehot_indices = K.argmax(prediction, axis=-1)  # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC
        return super().__call__(groundtruth, prediction_onehot)


class timecallback(Callback):
    def __init__(self):
        # use this value as reference to calculate cumulative time taken
        self.timetaken = time.perf_counter()

    def on_epoch_begin(self, epoch, logs):
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs):
        self.epoch_end_time = time.perf_counter()

        logs['epoch_time_in_sec'] = self.epoch_end_time - self.epoch_start_time
        logs['total_elapsed_time_in_sec'] = self.epoch_end_time - self.timetaken
