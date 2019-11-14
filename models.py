from keras.optimizers import Adam
from keras.metrics import accuracy, binary_crossentropy, categorical_crossentropy
from segmentation_models import Unet
from segmentation_models.metrics import iou_score
from segmentation_models.losses import jaccard_loss, dice_loss


def generate_compiled_segmentation_model(model_name, model_parameters, num_classes, loss, optimizer,
                                         weights_to_load=None):

    # These are the only model, loss, and optimizer currently supported
    assert model_name == 'Unet'
    assert loss == 'cross_entropy'
    assert optimizer == 'adam'

    model = Unet(input_shape=(None, None, 1), classes=num_classes, **model_parameters)

    crossentropy = binary_crossentropy if num_classes else categorical_crossentropy
    loss_fn = crossentropy

    model.compile(optimizer=Adam(),
                  loss=loss_fn,
                  metrics=[accuracy, iou_score, jaccard_loss, dice_loss])

    if weights_to_load:
        model.load_weights(weights_to_load)

    return model
