from keras.optimizers import Adam
from keras.metrics import accuracy, Accuracy, binary_accuracy, categorical_accuracy, CategoricalAccuracy, \
    BinaryCrossentropy as BinaryCrossentropyM, CategoricalCrossentropy as CategoricalCrossentropyM
from keras.metrics import FalsePositives, TruePositives, TrueNegatives, FalseNegatives, Precision, Recall
from keras.losses import binary_crossentropy, BinaryCrossentropy as BinaryCrossentropyL, categorical_crossentropy, \
    CategoricalCrossentropy as CategoricalCrossentropyL
from segmentation_models import Unet
# from segmentation_models.metrics import iou_score, IOUScore, f1_score, f2_score, FScore, precision, Precision,
# recall, Recall
from segmentation_models.losses import CategoricalCELoss
from metrics_utils import OneHotAccuracy, OneHotFalseNegatives, OneHotFalsePositives, OneHotTrueNegatives, \
    OneHotTruePositives, OneHotPrecision, OneHotRecall, ClassBinaryAccuracyKeras, OneHotClassBinaryAccuracyKeras, \
    ClassBinaryAccuracySM, OneHotClassBinaryAccuracySM
from metrics_utils import FBetaScore, OneHotFBetaScore, IoUScore, OneHotIoUScore

global_threshold = 0.5  # 0.5 is default prediction threshold for most metrics feat. this attribute
assert global_threshold <= 1.0


def generate_compiled_segmentation_model(model_name, model_parameters, num_classes, loss, optimizer,
                                         weights_to_load=None):

    # These are the only model, loss, and optimizer currently supported
    assert model_name == 'Unet'
    assert loss == 'cross_entropy'
    assert optimizer == 'adam'

    model = Unet(input_shape=(None, None, 1), classes=num_classes, **model_parameters)

    crossentropy = BinaryCrossentropyL() if num_classes == 1 else CategoricalCrossentropyL()
    loss_fn = crossentropy
    all_metrics = []
    if isinstance(loss_fn, BinaryCrossentropyL):
        all_metrics.append(BinaryCrossentropyM())
    else:
        all_metrics.append(CategoricalCrossentropyM())
    all_metrics[0].name = str(all_metrics[0].name + '_keras_metric')

    for class_num in range(num_classes + 1):
        if class_num == 0:    # all class metrics
            # note, `loss_fn` for all classes placed before `all_metrics` in lineup of command window metrics and plots
            all_metrics.extend([
                                CategoricalCELoss(),
                                Accuracy(),
                                OneHotAccuracy(name='accuracy_1H'),
                                CategoricalAccuracy(name='categ_acc_class'),
                                FalseNegatives(name='false_neg', thresholds=global_threshold),
                                OneHotFalseNegatives(name='false_neg_1H'),
                                TrueNegatives(name='true_neg', thresholds=global_threshold),
                                OneHotTrueNegatives(name='true_neg_1H'),
                                FalsePositives(name='false_pos', thresholds=global_threshold),
                                OneHotFalsePositives(name='false_pos_1H'),
                                TruePositives(name='true_pos', thresholds=global_threshold),
                                OneHotTruePositives(name='true_pos_1H'),
                                Recall(name='recall', thresholds=global_threshold),
                                OneHotRecall(name='recall_1H'),
                                Precision(name='precision', thresholds=global_threshold),
                                OneHotPrecision(name='precision_1H'),
                                FBetaScore(name='f1_score', beta=1, thresholds=global_threshold),
                                OneHotFBetaScore(name='f1_score_1H', beta=1),
                                IoUScore(name='iou_score', thresholds=global_threshold),
                                OneHotIoUScore(name='iou_score_1H')
                                ])
            all_metrics[1].name = str('categ_ce_sm_loss')
        else:    # per class metrics
            all_metrics.append(CategoricalCELoss(class_indexes=class_num - 1))
            all_metrics[-1].name = str('class' + str(class_num - 1) + '_binary_cross_entropy')
            all_metrics.append(ClassBinaryAccuracySM(name=str('class' + str(class_num - 1) + '_binary_accuracy_sm'),
                                                     class_indexes=class_num - 1, threshold=global_threshold))
            all_metrics.append(OneHotClassBinaryAccuracySM(name=str('class' + str(class_num - 1) + '_binary_accuracy_1H_sm'),
                                                           class_indexes=class_num - 1, threshold=global_threshold))
            all_metrics.append(ClassBinaryAccuracyKeras(name=str('class' + str(class_num - 1) + '_binary_accuracy_keras'),
                                                        class_id=class_num - 1, thresholds=global_threshold))
            all_metrics.append(OneHotClassBinaryAccuracyKeras(name=str('class' + str(class_num - 1) + '_binary_accuracy_1H_keras'),
                                                              class_id=class_num - 1, thresholds=global_threshold))
            all_metrics.append(OneHotIoUScore(name=str('class' + str(class_num - 1) + '_iou_score'),
                                              class_id=class_num - 1, thresholds=global_threshold))
            all_metrics.append(OneHotFBetaScore(name=str('class' + str(class_num - 1) + '_f1_score'),
                                                class_id=class_num - 1,
                                                beta=1, thresholds=global_threshold))
            all_metrics.append(OneHotPrecision(name=str('class' + str(class_num - 1) + '_precision'),
                                               class_id=class_num - 1, thresholds=global_threshold))
            all_metrics.append(OneHotRecall(name=str('class' + str(class_num - 1) + '_recall'),
                                            class_id=class_num - 1, thresholds=global_threshold))
        if num_classes == 1:
            break

    # manually check metric attributes and inheritance
    # print(all_metrics)
    # for m in all_metrics:
    #     if hasattr(m.__class__, '__name__'):
    #         print(m.name, m.__class__.__name__, m.__class__.__mro__)
    # input("pre-compile - press enter")

    model.compile(optimizer=Adam(),
                  loss=loss_fn,
                  metrics=all_metrics)

    if weights_to_load:
        model.load_weights(weights_to_load)

    return model
