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


thresholds_training_history = {}
train_thresholds_counter = 0


def generate_compiled_segmentation_model(model_name, model_parameters, num_classes, loss, optimizer,
                                         weights_to_load=None, optimizing_threshold_class_metric=None,
                                         optimizing_class_id=None, optimizing_input_threshold=None,
                                         optimized_class_thresholds=None):

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
        if class_num == 0 and optimizing_threshold_class_metric is None:    # all class metrics
            # note, `loss_fn` for all classes placed before `all_metrics` in lineup of command window metrics and plots
            if not isinstance(loss_fn, BinaryCrossentropyL):
                all_metrics.extend([CategoricalCELoss()])
                all_metrics[1].name = str('categ_cross_entropy_sm')
            all_metrics.extend([
                AccuracyTfKeras(),
                # OneHotAccuracyTfKeras(),  # `global_threshold` built-in
                ClassBinaryAccuracyTfKeras(thresholds=global_threshold),
                # OneHotClassBinaryAccuracyTfKeras(thresholds=global_threshold),
                ClassBinaryAccuracySM(threshold=global_threshold),
                # OneHotClassBinaryAccuracySM(threshold=global_threshold),
                BinaryAccuracy(threshold=global_threshold),
                CategoricalAccuracy(),
                FalseNegatives(name='false_neg', thresholds=global_threshold),
                # OneHotFalseNegatives(name='false_neg_1H', thresholds=global_threshold),
                TrueNegatives(name='true_neg', thresholds=global_threshold),
                # OneHotTrueNegatives(name='true_neg_1H', thresholds=global_threshold),
                FalsePositives(name='false_pos', thresholds=global_threshold),
                # OneHotFalsePositives(name='false_pos_1H', thresholds=global_threshold),
                TruePositives(name='true_pos', thresholds=global_threshold),
                # OneHotTruePositives(name='true_pos_1H', thresholds=global_threshold),
                Recall(name='recall', thresholds=global_threshold),
                # OneHotRecall(name='recall_1H', thresholds=global_threshold),
                Precision(name='precision', thresholds=global_threshold),
                # OneHotPrecision(name='precision_1H', thresholds=global_threshold),
                FBetaScore(name='f1_score', beta=1, thresholds=global_threshold),
                # OneHotFBetaScore(name='f1_score_1H', beta=1, thresholds=global_threshold),
                IoUScore(name='iou_score', thresholds=global_threshold),
                # OneHotIoUScore(name='iou_score_1H', thresholds=global_threshold)
            ])
        elif class_num == 0 and optimizing_threshold_class_metric is not None:  # all class metrics
            continue
        else:    # per class metrics
            if optimizing_threshold_class_metric is not None:
                class_threshold = optimizing_input_threshold
                class_num = optimizing_class_id + 1
            elif optimized_class_thresholds is None:
                class_threshold = global_threshold
            else:
                class_threshold = optimized_class_thresholds[str('class' + str(class_num - 1))]

            all_metrics.append(CategoricalCELoss(class_indexes=class_num - 1))
            all_metrics[-1].name = str('class' + str(class_num - 1) + '_binary_cross_entropy')
            all_metrics.append(ClassBinaryAccuracySM(name=str('class' + str(class_num - 1) + '_binary_accuracy_sm'),
                                                     class_indexes=class_num - 1, threshold=class_threshold))
            all_metrics.append(ClassBinaryAccuracyTfKeras(name=str('class' + str(class_num - 1) + '_binary_accuracy_tfkeras'),
                                                          class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(IoUScore(name=str('class' + str(class_num - 1) + '_iou_score'),
                                        class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(FBetaScore(name=str('class' + str(class_num - 1) + '_f1_score'),
                                          class_id=class_num - 1,
                                          beta=1, thresholds=class_threshold))
            all_metrics.append(Precision(name=str('class' + str(class_num - 1) + '_precision'),
                                         class_id=class_num - 1, thresholds=class_threshold))
            all_metrics.append(Recall(name=str('class' + str(class_num - 1) + '_recall'),
                                      class_id=class_num - 1, thresholds=class_threshold))

            if optimizing_threshold_class_metric is not None:
                break

        if num_classes == 1:
            break

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = Unet(input_shape=(None, None, 1), classes=num_classes, **model_parameters)
    model.compile(optimizer=Adam(),
                  loss=loss_fn,
                  metrics=all_metrics)

    if weights_to_load:
        model.load_weights(weights_to_load)

    if optimizing_threshold_class_metric is None:
        print(model.summary())

    return model


class EvaluateModelForInputThreshold:
    def __init__(
            self,
            optimizing_class_id=None,
            optimizing_threshold_class_metric=None,
            train_config=None,
            dataset_generator=None,
            dataset_downsample_factor=1.0,
            model_path=False,
            name=None
    ):
        self.name = name or 'optimizing_compiled_model'
        self.optimizing_class_id = optimizing_class_id
        self.optimizing_threshold_class_metric = optimizing_threshold_class_metric
        self.train_config = train_config
        self.dataset_generator = dataset_generator
        self.dataset_downsample_factor = dataset_downsample_factor
        assert 0 < self.dataset_downsample_factor <= 1.0
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
            optimizing_threshold_class_metric=self.optimizing_threshold_class_metric,
            optimizing_class_id=self.optimizing_class_id,
            optimizing_input_threshold=input_threshold)
        all_results = optimizing_model.evaluate(self.dataset_generator,
                                                steps=np.ceil(self.dataset_downsample_factor *
                                                              len(self.dataset_generator)).astype(int))
        if hasattr(optimizing_model.loss, '__name__'):
            metric_names = [optimizing_model.loss.__name__] + [m.name for m in optimizing_model.metrics]
        elif hasattr(optimizing_model.loss, 'name'):
            metric_names = [optimizing_model.loss.name] + [m.name for m in optimizing_model.metrics]
        dict_results = dict(zip(metric_names, all_results))

        optimizing_result = dict_results[str('class' + str(self.optimizing_class_id) + '_'
                                             + self.optimizing_threshold_class_metric)]

        print(
            '\n For input threshold of {}, evaluation of optimizing class{}_{} is {}'.format(input_threshold,
                                                                                             self.optimizing_class_id,
                                                                                             self.optimizing_threshold_class_metric,
                                                                                             optimizing_result)
        )
        print('\n')

        global train_thresholds_counter
        if train_thresholds_counter == 0:
            thresholds_training_history[str('class' + str(self.optimizing_class_id))] = {}
        thresholds_training_history[str('class' + str(self.optimizing_class_id))][str(str(train_thresholds_counter) + '_threshold_metric')] = [float(input_threshold), float(optimizing_result)]
        train_thresholds_counter += 1

        return 1 - optimizing_result


# framework to train prediction threshold
def train_prediction_thresholds(optimizing_class_id, optimizing_threshold_class_metric, train_config,
                                dataset_generator, dataset_downsample_factor, model_path):

    optimizing_compiled_model = EvaluateModelForInputThreshold(optimizing_class_id, optimizing_threshold_class_metric,
                                                               train_config, dataset_generator,
                                                               dataset_downsample_factor, model_path)
    global train_thresholds_counter
    train_thresholds_counter = 0

    opt_bounds = [0, 1]
    opt_method = 'bounded'
    opt_tol = 1e-2
    opt_options = {'maxiter': 500, 'disp': 3}
    optimization_configuration = {'opt_bounds': opt_bounds, 'opt_method': opt_method, 'opt_tol': opt_tol,
                                  'opt_options': opt_options, 'opt_class_metric': optimizing_threshold_class_metric,
                                  'opt_dataset_generator': dataset_generator.dataset_directory,
                                  'opt_dataset_downsample_factor': dataset_downsample_factor}
    training_threshold_output = minimize_scalar(optimizing_compiled_model, bounds=(opt_bounds[0], opt_bounds[1]),
                                                method=opt_method, tol=opt_tol, options=opt_options)

    return training_threshold_output, optimization_configuration
