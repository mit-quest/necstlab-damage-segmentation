from keras.optimizers import Adam
from keras.metrics import accuracy, binary_accuracy, categorical_accuracy, binary_crossentropy, categorical_crossentropy
from keras import backend as K
from segmentation_models import Unet
from segmentation_models.metrics import iou_score, IOUScore, f1_score, f2_score, FScore, precision, Precision, recall, Recall
from segmentation_models.losses import jaccard_loss, JaccardLoss, dice_loss, DiceLoss, CategoricalCELoss
from segmentation_models.base import KerasObject, Metric
from segmentation_models.base import functional

SMOOTH = 1e-5


class OneHotMetricWrapper:
    def __init__(
            self,
            name_1h=None,              # segmentation_models and keras attribute, but post-init for s_m.losses
            class_weights_1h=None,     # segmentation_models only attributes:
            class_indexes_1h=None,       # Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
            threshold_1h=None,
            beta_1h=1,
            per_image_1h=False,
            smooth_1h=SMOOTH,
            dtype_1h=None,             # Keras only attributes:
            from_logits_1h=False,
            label_smoothing_1h=0
    ):
        self.name_1h = name_1h
        self.class_weights_1h = class_weights_1h if class_weights_1h is not None else 1
        self.class_indexes_1h = class_indexes_1h
        self.threshold_1h = threshold_1h
        self.beta_1h = beta_1h
        self.per_image_1h = per_image_1h
        self.smooth_1h = smooth_1h
        self.dtype_1h = dtype_1h
        self.from_logits_1h = from_logits_1h
        self.label_smoothing_1h = label_smoothing_1h
        # apply correct pre-defined metric that requires one-hot predictions:
        if self.name_1h == 'accuracy':
            self.metric_class_instance_1h = accuracy
        elif self.name_1h == 'iou_score':
            self.metric_class_instance_1h = iou_score
        elif self.name_1h == 'IOUScore':
            self.metric_class_instance_1h = IOUScore(name=str('class' + str(self.class_indexes_1h) + '_iou_score'),
                                                     class_indexes=self.class_indexes_1h)
        elif self.name_1h == 'f1_score':
            self.metric_class_instance_1h = f1_score
        elif self.name_1h == 'f2_score':
            self.metric_class_instance_1h = f2_score
        elif self.name_1h == 'FScore':
            self.metric_class_instance_1h = FScore(name=str('class' + str(self.class_indexes_1h) + '_f1_score'),
                                                   class_indexes=self.class_indexes_1h, beta=self.beta_1h)
        elif self.name_1h == 'precision':
            self.metric_class_instance_1h = precision
        elif self.name_1h == 'Precision':
            self.metric_class_instance_1h = Precision(name=str('class' + str(self.class_indexes_1h) + '_precision'),
                                                      class_indexes=self.class_indexes_1h)
        elif self.name_1h == 'recall':
            self.metric_class_instance_1h = recall
        elif self.name_1h == 'Recall':
            self.metric_class_instance_1h = Recall(name=str('class' + str(self.class_indexes_1h) + '_recall'),
                                                   class_indexes=self.class_indexes_1h)
        elif self.name_1h == 'jaccard_loss':
            self.metric_class_instance_1h = jaccard_loss
        elif self.name_1h == 'JaccardLoss':
            self.metric_class_instance_1h = JaccardLoss(class_indexes=self.class_indexes_1h)
            self.metric_class_instance_1h.name = str('class' + str(self.class_indexes_1h) + '_jaccard_loss')
        elif self.name_1h == 'dice_loss':
            self.metric_class_instance_1h = dice_loss
        elif self.name_1h == 'DiceLoss':
            self.metric_class_instance_1h = DiceLoss(class_indexes=self.class_indexes_1h, beta=self.beta_1h)
            self.metric_class_instance_1h.name = str('class' + str(self.class_indexes_1h) + '_dice_loss')
        elif self.name_1h == 'CategoricalCELoss':
            self.metric_class_instance_1h = CategoricalCELoss(class_indexes=self.class_indexes_1h)
            self.metric_class_instance_1h.name = str('class' + str(self.class_indexes_1h) + '_binary_cross_entropy')
        elif self.name_1h == 'ClassBinaryAccuracy':   # can use threshold instead of one hot
            self.metric_class_instance_1h = ClassBinaryAccuracy(name=str('class' + str(self.class_indexes_1h) +
                                                                         '_binary_accuracy'),
                                                                class_indexes=self.class_indexes_1h, threshold=None)
        else:
            self.metric_class_instance_1h = None  # results in error

    def __call__(self, groundtruth, prediction):   # assuming 4D tensor is BHWC
        # based on keras.metrics.categorical_accuracy to determine max pred index (1 of channels) at each HW location
        prediction_onehot_indices = K.cast(K.argmax(prediction, axis=-1), K.floatx())
        prediction_onehot = K.one_hot(prediction_onehot_indices, K.int_shape(prediction)[3])  # assume 4D tensor is BHWC

        return self.metric_class_instance_1h(groundtruth, prediction_onehot)


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
        if class_num == 0:
            # note, `loss_fn` for all classes placed before `all_metrics` in lineup of command window metrics and plots
            all_metrics.extend([CategoricalCELoss(), OneHotMetricWrapper(name_1h='accuracy'), categorical_accuracy,
                                OneHotMetricWrapper(name_1h='iou_score'), OneHotMetricWrapper(name_1h='f1_score'),
                                OneHotMetricWrapper(name_1h='precision'), OneHotMetricWrapper(name_1h='recall')])
            all_metrics[0].name = str('sm_cat_cross_entropy')
        else:
            all_metrics.append(OneHotMetricWrapper(name_1h='CategoricalCELoss', class_indexes_1h=class_num - 1))
            all_metrics.append(OneHotMetricWrapper(name_1h='ClassBinaryAccuracy', class_indexes_1h=class_num - 1,
                                                   threshold_1h=None))
            all_metrics.append(OneHotMetricWrapper(name_1h='IOUScore', class_indexes_1h=class_num - 1))
            all_metrics.append(OneHotMetricWrapper(name_1h='JaccardLoss', class_indexes_1h=class_num - 1))
            all_metrics.append(OneHotMetricWrapper(name_1h='FScore', class_indexes_1h=class_num - 1, beta_1h=1))
            all_metrics.append(OneHotMetricWrapper(name_1h='DiceLoss', class_indexes_1h=class_num - 1, beta_1h=1))
            all_metrics.append(OneHotMetricWrapper(name_1h='Precision', class_indexes_1h=class_num - 1))
            all_metrics.append(OneHotMetricWrapper(name_1h='Recall', class_indexes_1h=class_num - 1))
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
