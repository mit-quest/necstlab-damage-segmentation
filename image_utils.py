import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import io


class_RGB_mapping = {
    'class_0': None,
    'class_1': [255, 0, 0],  # blue: 0 degree ply damage
    'class_2': [0, 255, 0],  # green: 45 degree ply damage
    'class_3': [0, 0, 255],  # red: 90 degree ply damage
}

GV_RGB_mapping = {
    100: [255, 0, 0],  # blue: 0 degree ply damage
    175: [0, 255, 0],  # green: 45 degree ply damage
    250: [0, 0, 255],  # red: 90 degree ply damage
}


# def overlay_batch_predictions(image_file, prediction_array, threshold=None):
#     image = np.asarray(Image.open(image_file).convert('RGB'))
#     for c in batch_image_array[0].shape:
#
#
#         composite_image = Image.fromarray(composite_image)
#
#     return composite_image



def overlay_masks(images, masks):
    # assert type(images) == type(masks)
    if type(images) == np.ndarray:
        raise NotImplementedError
    else:  # assume is a file path
        image_file, mask_files = images, masks
        image = np.asarray(Image.open(image_file).convert('RGB'))
        masks = {c: np.asarray(Image.open(mask_files[c])) for c in mask_files}

        composite_image = image.copy()
        for c in masks:
            if class_RGB_mapping[c] is not None:
                composite_image[masks[c].astype(bool)] = class_RGB_mapping[c]

    composite_image = Image.fromarray(composite_image)
    return composite_image


def overlay_annotation(image_file, annotation_file):
    image = np.asarray(Image.open(image_file).convert('RGB'))
    annotation = np.asarray(Image.open(annotation_file))

    composite_image = image.copy()
    for annotation_gv in GV_RGB_mapping:
        composite_image[annotation == annotation_gv] = GV_RGB_mapping[annotation_gv]

    composite_image = Image.fromarray(composite_image)
    return composite_image


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, log_dir, images_and_masks_paths):
        super().__init__()
        self.log_dir = log_dir
        self.images_and_masks_paths = images_and_masks_paths

    def on_epoch_end(self, epoch, logs=None):

        writer = tf.summary.FileWriter(self.log_dir)

        for image_file, mask_files in self.images_and_masks_paths:
            composite_image = np.asarray(overlay_masks(image_file, mask_files))
            summary_image = make_image(composite_image)

            summary = tf.Summary(value=[tf.Summary.Value(tag=image_file.as_posix(), image=summary_image)])
            writer.add_summary(summary, epoch)

        writer.close()



# image_file = '/media/josh/Extra Drive 1/necstlab/data/easy_copy/prepared/train/images/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160_0018.tif'
# mask_files = {
#     'class_0': '/media/josh/Extra Drive 1/necstlab/data/easy_copy/prepared/train/masks/class_0/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160_dmg_labels_GV_0018.tif',
#     'class_1': '/media/josh/Extra Drive 1/necstlab/data/easy_copy/prepared/train/masks/class_1/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160_dmg_labels_GV_0018.tif',
#     'class_2': '/media/josh/Extra Drive 1/necstlab/data/easy_copy/prepared/train/masks/class_2/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160_dmg_labels_GV_0018.tif'
# }
#
# overlay_masks(image_file, mask_files).show()


# image_file = '/media/josh/Extra Drive 1/necstlab/data/processing/easy/downsampled/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160/images/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160_0018.tif'
# annotation_file = '/media/josh/Extra Drive 1/necstlab/data/processing/easy/downsampled/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160/annotations/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160_dmg_labels_GV_0018.tif'

# image_file = '/media/josh/Extra Drive 1/necstlab/data/processing/easy/resized/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160/images/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160_0018.tif'
# annotation_file = '/media/josh/Extra Drive 1/necstlab/data/processing/easy/resized/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160/annotations/8bit_AS4_CNT_S2_P1_L6_2560_1800_2160_dmg_labels_GV_0018.tif'

# image_file = '/media/josh/Extra Drive 1/necstlab/data/processing/easy/resized/THIN_REF_S2_P1_L3_2496_1563_2159/images/THIN_REF_S2_P1_L3_2496_1563_2159-0070.tif'
# annotation_file = '/media/josh/Extra Drive 1/necstlab/data/processing/easy/resized/THIN_REF_S2_P1_L3_2496_1563_2159/annotations/THIN_REF_S2_P1_L3_2496_1563_2159-dmg_labels_GV-0070.tif'
#
# image_file = '/media/josh/Extra Drive 1/necstlab/data/processing/easy/resized/THIN_REF_S2_P1_L3_2496_1563_2159/images/THIN_REF_S2_P1_L3_2496_1563_2159-0195.tif'
# annotation_file = '/media/josh/Extra Drive 1/necstlab/data/processing/easy/resized/THIN_REF_S2_P1_L3_2496_1563_2159/annotations/THIN_REF_S2_P1_L3_2496_1563_2159-dmg_labels_GV-0195.tif'
# #
# overlay_annotation(image_file, annotation_file).show()
