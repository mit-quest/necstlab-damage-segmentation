from keras.optimizers import Adam
from keras.metrics import accuracy, binary_crossentropy, categorical_crossentropy
from segmentation_models import Unet
from segmentation_models.metrics import iou_score, IOUScore
from segmentation_models.losses import jaccard_loss, JaccardLoss, dice_loss, DiceLoss, CategoricalCELoss
from image_utils import class_binary_accuracy, ClassBinaryAccuracy


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
            all_metrics.extend([accuracy, iou_score, jaccard_loss])  # , dice_loss])
        else:
            all_metrics.append(CategoricalCELoss(class_indexes=class_num - 1))
            all_metrics[-1].name = str('class' + str(class_num - 1) + '_binary_cross_entropy')
            all_metrics.append(ClassBinaryAccuracy(name=str('class' + str(class_num - 1) + '_binary_accuracy'), class_indexes=class_num - 1))
            all_metrics.append(IOUScore(name=str('class' + str(class_num - 1) + '_iou_score'), class_indexes=class_num - 1))
            all_metrics.append(JaccardLoss(class_indexes=class_num - 1))
            all_metrics[-1].name = str('class' + str(class_num - 1) + '_jaccard_loss')
            # all_metrics.append(DiceLoss(class_indexes=class_num - 1))
            # all_metrics[-1].name = str('class' + str(class_num - 1) + '_dice_loss')
        if num_classes == 1:
            break

    model.compile(optimizer=Adam(),
                  loss=loss_fn,
                  metrics=all_metrics)

    if weights_to_load:
        model.load_weights(weights_to_load)

    return model
