from keras.optimizers import Adam
from keras.metrics import accuracy, Accuracy, binary_accuracy, categorical_accuracy, binary_crossentropy, categorical_crossentropy
from keras import backend as K
from segmentation_models import Unet
from segmentation_models.metrics import iou_score, IOUScore, f1_score, f2_score, FScore, precision, Precision, recall, Recall
from segmentation_models.losses import jaccard_loss, JaccardLoss, dice_loss, DiceLoss, CategoricalCELoss
from segmentation_models.base import KerasObject, Metric
from segmentation_models.base import functional

SMOOTH = 1e-5


class OneHotMetricWrapper(KerasObject):
    def __init__(
            self,
            name=None,
            metric_class_instance=None,   # DO I NEED A COMMA HERE?
    ):
        self.metric_class_instance = metric_class_instance
        self.name = name or self.metric_class_instance.name
        super().__init__(name=self.name)

    def __call__(self, groundtruth, prediction):   # assuming 4D tensor is BHWC
        # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot_indices = K.argmax(prediction, axis=-1)
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[-1])  # assume 4D tensor is BHWC

        return self.metric_class_instance(groundtruth, prediction_onehot)


def generate_compiled_segmentation_model(model_name, model_parameters, num_classes, loss, optimizer,
                                         weights_to_load=None):

    # These are the only model, loss, and optimizer currently supported
    assert model_name == 'Unet'
    assert loss == 'cross_entropy'
    assert optimizer == 'adam'

    model = Unet(input_shape=(None, None, 1), classes=num_classes, **model_parameters)

    crossentropy = binary_crossentropy if num_classes == 1 else categorical_crossentropy
    loss_fn = crossentropy

    all_metrics = []
    for class_num in range(num_classes + 1):
        if class_num == 0:    # all class metrics
            # note, `loss_fn` for all classes placed before `all_metrics` in lineup of command window metrics and plots
            all_metrics.extend([CategoricalCELoss(),
                                OneHotMetricWrapper(metric_class_instance=Accuracy()),
                                OneHotMetricWrapper(metric_class_instance=ClassBinaryAccuracy(threshold=None)),
                                categorical_accuracy,
                                OneHotMetricWrapper(metric_class_instance=IOUScore()),
                                OneHotMetricWrapper(metric_class_instance=FScore(beta=1)),
                                OneHotMetricWrapper(metric_class_instance=Precision())])
            all_metrics[0].name = str('sm_cat_cross_entropy')
        else:    # per class metrics
            all_metrics.append(CategoricalCELoss(class_indexes=class_num - 1))
            all_metrics[-1].name = str('class' + str(class_num - 1) + '_binary_cross_entropy')
            all_metrics.append(OneHotMetricWrapper(metric_class_instance=ClassBinaryAccuracy(name=str('class' + str(class_num - 1) + '_binary_accuracy'),
                                                                                             class_indexes=class_num - 1,
                                                                                             threshold=None)))
            all_metrics.append(OneHotMetricWrapper(metric_class_instance=IOUScore(name=str('class' + str(class_num - 1) + '_iou_score'),
                                                                                  class_indexes=class_num - 1)))
            all_metrics.append(OneHotMetricWrapper(metric_class_instance=JaccardLoss(class_indexes=class_num - 1)))
            all_metrics[-1].name = str('class' + str(class_num - 1) + '_jaccard_loss')
            all_metrics.append(OneHotMetricWrapper(metric_class_instance=FScore(name=str('class' + str(class_num - 1) + '_f1_score'),
                                                                                class_indexes=class_num - 1,
                                                                                beta=1)))
            all_metrics.append(OneHotMetricWrapper(metric_class_instance=DiceLoss(class_indexes=class_num - 1,
                                                                                  beta=1)))
            all_metrics[-1].name = str('class' + str(class_num - 1) + '_dice_loss')
            all_metrics.append(OneHotMetricWrapper(metric_class_instance=Precision(name=str('class' + str(class_num - 1) + '_precision'),
                                                                                   class_indexes=class_num - 1)))
            all_metrics.append(OneHotMetricWrapper(metric_class_instance=Recall(name=str('class' + str(class_num - 1) + '_recall'),
                                                                                class_indexes=class_num - 1)))
        if num_classes == 1:
            break

    model.compile(optimizer=Adam(),
                  loss=loss_fn,
                  metrics=all_metrics)

    if weights_to_load:
        model.load_weights(weights_to_load)

    return model


# adapted from: IOUScore() from https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/metrics.py
class ClassBinaryAccuracy(Metric):
    r"""
    .. math:: Binary Accuracy = (TN + TP)/(TN+TP+FN+FP) = Number of correct assessments/Number of all assessments, for given class
    for more than one class input, output becomes mean accuracy (similar but not same as categorical)
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
            name=None,
    ):
        name = name or 'class_i_binary_accuracy'
        super().__init__(name=name)
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

        # score calculation (assumed pr are 1-hot)
        tp = backend.sum(gt * pr, axis=axes)
        fp = backend.sum(pr, axis=axes) - tp
        fn = backend.sum(gt, axis=axes) - tp
        tn = backend.sum((-gt + 1) * (-pr + 1), axis=axes)
        score = (tp + tn) / (tp + tn + fp + fn + self.smooth)
        score = functional.average(score, self.per_image, self.class_weights, **self.submodules)

        return score


# alias
class_binary_accuracy = ClassBinaryAccuracy()
