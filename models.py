import os
from scipy.optimize import minimize_scalar
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import (Accuracy as AccuracyTfKeras, BinaryAccuracy, CategoricalAccuracy,
                                      BinaryCrossentropy as BinaryCrossentropyM,
                                      CategoricalCrossentropy as CategoricalCrossentropyM,
                                      FalsePositives, TruePositives, TrueNegatives, FalseNegatives, Precision, Recall)
from tensorflow.keras.losses import (BinaryCrossentropy as BinaryCrossentropyL,
                                     CategoricalCrossentropy as CategoricalCrossentropyL)
from metrics_utils import (OneHotAccuracyTfKeras, OneHotFalseNegatives, OneHotFalsePositives,
                           OneHotTrueNegatives, OneHotTruePositives, OneHotPrecision, OneHotRecall,
                           ClassBinaryAccuracyTfKeras, OneHotClassBinaryAccuracyTfKeras, ClassBinaryAccuracySM,
                           OneHotClassBinaryAccuracySM, FBetaScore, OneHotFBetaScore, IoUScore, OneHotIoUScore,
                           global_threshold)
os.environ['SM_FRAMEWORK'] = 'tf.keras'  # will tell segmentation models to use tensorflow's keras
from segmentation_models import Unet
from segmentation_models.losses import CategoricalCELoss


def generate_compiled_segmentation_model(model_name, model_parameters, num_classes, loss, optimizer,
                                         weights_to_load=None, optimizing_threshold_flag=False,
                                         optimizing_class_id=None, optimizing_input_threshold=None,
                                         class_optimized_thresholds=None):

    # These are the only model, loss, and optimizer currently supported
    assert model_name == 'Unet'
    assert loss == 'cross_entropy'
    assert optimizer == 'adam'

    loss_fn = BinaryCrossentropyL()

    all_metrics = []    # one-hot versions are generally preferred for given metric
    # make first metric a copy of loss, to continually verify `val_loss` is correct
    if isinstance(loss_fn, BinaryCrossentropyL):
        all_metrics.append(BinaryCrossentropyM(name='binary_ce_metric'))
    else:
        all_metrics.append(CategoricalCrossentropyM(name='categ_ce_metric'))

    # standard thresholded version (default threshold is 0.5) also kept below, in case it's desired in certain scenario
    for class_num in range(num_classes + 1):
        if class_num == 0:    # all class metrics
            # note, `loss_fn` for all classes placed before `all_metrics` in lineup of command window metrics and plots
            if not isinstance(loss_fn, BinaryCrossentropyL):
                all_metrics.extend([CategoricalCELoss()])
                all_metrics[1].name = str('categ_cross_entropy_sm')
            all_metrics.extend([
                AccuracyTfKeras(),
                OneHotAccuracyTfKeras(),  # `global_threshold` built-in
                ClassBinaryAccuracyTfKeras(thresholds=global_threshold),
                OneHotClassBinaryAccuracyTfKeras(thresholds=global_threshold),
                ClassBinaryAccuracySM(threshold=global_threshold),
                OneHotClassBinaryAccuracySM(threshold=global_threshold),
                BinaryAccuracy(threshold=global_threshold),
                CategoricalAccuracy(),
                FalseNegatives(name='false_neg', thresholds=global_threshold),
                OneHotFalseNegatives(name='false_neg_1H', thresholds=global_threshold),
                TrueNegatives(name='true_neg', thresholds=global_threshold),
                OneHotTrueNegatives(name='true_neg_1H', thresholds=global_threshold),
                FalsePositives(name='false_pos', thresholds=global_threshold),
                OneHotFalsePositives(name='false_pos_1H', thresholds=global_threshold),
                TruePositives(name='true_pos', thresholds=global_threshold),
                OneHotTruePositives(name='true_pos_1H', thresholds=global_threshold),
                Recall(name='recall', thresholds=global_threshold),
                OneHotRecall(name='recall_1H', thresholds=global_threshold),
                Precision(name='precision', thresholds=global_threshold),
                OneHotPrecision(name='precision_1H', thresholds=global_threshold),
                FBetaScore(name='f1_score', beta=1, thresholds=global_threshold),
                OneHotFBetaScore(name='f1_score_1H', beta=1, thresholds=global_threshold),
                IoUScore(name='iou_score', thresholds=global_threshold),
                OneHotIoUScore(name='iou_score_1H', thresholds=global_threshold)
            ])
        else:    # per class metrics
            if class_optimized_thresholds is None or class_optimized_thresholds[str('class_'+str(class_num-1))] is None:
                class_threshold = global_threshold
            else:
                class_threshold = class_optimized_thresholds[str('class_'+str(class_num-1))]

            all_metrics.append(CategoricalCELoss(class_indexes=class_num - 1))
            all_metrics[-1].name = str('class' + str(class_num - 1) + '_binary_cross_entropy')
            all_metrics.append(ClassBinaryAccuracySM(name=str('class' + str(class_num - 1) + '_binary_accuracy_sm'),
                                                     class_indexes=class_num - 1, threshold=class_threshold))
            all_metrics.append(OneHotClassBinaryAccuracySM(name=str('class' + str(class_num - 1) + '_binary_accuracy_sm_1H'),
                                                           class_indexes=class_num - 1, threshold=class_threshold))
            all_metrics.append(ClassBinaryAccuracyTfKeras(name=str('class' + str(class_num - 1) + '_binary_accuracy_tfkeras'),
                                                          class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(OneHotClassBinaryAccuracyTfKeras(name=str('class' + str(class_num - 1) + '_binary_accuracy_tfkeras_1H'),
                                                                class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(IoUScore(name=str('class' + str(class_num - 1) + '_iou_score'),
                                        class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(OneHotIoUScore(name=str('class' + str(class_num - 1) + '_iou_score_1H'),
                                              class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(FBetaScore(name=str('class' + str(class_num - 1) + '_f1_score'),
                                          class_id=class_num - 1,
                                          beta=1, thresholds=class_threshold))
            all_metrics.append(OneHotFBetaScore(name=str('class' + str(class_num - 1) + '_f1_score_1H'),
                                                class_id=class_num - 1,
                                                beta=1, thresholds=class_threshold))
            all_metrics.append(Precision(name=str('class' + str(class_num - 1) + '_precision'),
                                         class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(OneHotPrecision(name=str('class' + str(class_num - 1) + '_precision_1H'),
                                               class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(Recall(name=str('class' + str(class_num - 1) + '_recall'),
                                      class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(OneHotRecall(name=str('class' + str(class_num - 1) + '_recall_1H'),
                                            class_id=class_num - 1, thresholds=class_threshold))

        if num_classes == 1:
            break

    if optimizing_threshold_flag:
        all_metrics = [OneHotIoUScore(name=str('class' + str(optimizing_class_id) + '_iou_score_1H'),
                                      class_id=optimizing_class_id, thresholds=optimizing_input_threshold)]

    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    model = Unet(input_shape=(None, None, 1), classes=num_classes, **model_parameters)
    model.compile(optimizer=Adam(),
                  loss=loss_fn,
                  metrics=all_metrics)

    if weights_to_load:
        model.load_weights(weights_to_load)

    if not optimizing_threshold_flag:
        print(model.summary())

    return model


class EvaluateModelForInputThreshold:
    def __init__(
            self,
            optimizing_class_id=None,
            train_config=None,
            dataset_generator=None,
            model_path=False,
            name=None
    ):
        self.name = name or 'optimizing_compiled_model'
        self.optimizing_class_id = optimizing_class_id
        self.train_config = train_config
        self.dataset_generator = dataset_generator
        self.model_path = model_path

    # evaluate model performance on specified dataset for specified prediction threshold
    def __call__(self, input_threshold):
        optimizing_model = generate_compiled_segmentation_model(
            self.train_config['segmentation_model']['model_name'],
            self.train_config['segmentation_model']['model_parameters'],
            len(self.dataset_generator.mask_filenames),
            self.train_config['loss'],
            self.train_config['optimizer'],
            weights_to_load=self.model_path,
            optimizing_threshold_flag=True,
            optimizing_class_id=self.optimizing_class_id,
            optimizing_input_threshold=input_threshold)

        all_results = optimizing_model.evaluate(self.dataset_generator,
                                                steps=np.ceil(len(self.dataset_generator) / 2).astype(int))
        assert len(all_results) == 2

        return 1 - all_results[-1]


# framework to fit prediction threshold
def fit_prediction_thresholds(optimizing_class_id, train_config, dataset_generator, model_path):
    optimizing_compiled_model = EvaluateModelForInputThreshold(optimizing_class_id, train_config, dataset_generator,
                                                               model_path)
    opt_bounds = [0, 1]
    opt_method = 'bounded'
    opt_tol = 0.01
    opt_options = {'maxiter' : 1000, 'disp' : True}
    optimization_configuration = {'opt_bounds': opt_bounds, 'opt_method': opt_method, 'opt_tol': opt_tol,
                                  'opt_options': opt_options}
    optimized_threshold = minimize_scalar(optimizing_compiled_model, bounds=(opt_bounds[0], opt_bounds[1]),
                                          method=opt_method, tol=opt_tol, options=opt_options)

    return optimized_threshold, optimization_configuration
