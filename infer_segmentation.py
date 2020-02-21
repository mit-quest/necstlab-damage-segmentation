import os
import shutil
import numpy as np
import yaml
from datetime import datetime
import pytz
from PIL import Image, ImageOps
from pathlib import Path
import git
from models import generate_compiled_segmentation_model

# infer can be run multiple times (labels, overlay), create new metadata each time
metadata_file_name = 'metadata_' + datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ') + '.yaml'
tmp_directory = Path('./tmp')

# rgb
class_colors = [
    [0, 0, 255],    # blue
    [255, 255, 0],  # yellow
    [255, 0, 0],    # red
    [0, 255, 0],    # green
    [255, 0, 255]   # magenta
]


def stitch_preds_together(tiles, target_size_1d, labels_output):
    n_tile_rows = len(tiles)
    n_tile_cols = len(tiles[0])
    stitched_array = np.zeros((target_size_1d * n_tile_rows, target_size_1d * n_tile_cols, 3))
    for i in range(n_tile_rows):
        for j in range(n_tile_cols):
            stitched_array[i*target_size_1d:(i+1)*target_size_1d, j*target_size_1d:(j+1)*target_size_1d, :] = tiles[i][j]
    if labels_output:
        stitched_image = Image.fromarray(np.mean(stitched_array, -1).astype('uint8'))
    else:
        stitched_image = Image.fromarray(stitched_array.astype('uint8'))
    return stitched_image


def prepare_image(image, target_size_1d):
    # make the image an event multiple of 512x512
    desired_size = target_size_1d * np.ceil(np.asarray(image.size) / target_size_1d).astype(int)
    delta_w = desired_size[0] - image.size[0]
    delta_h = desired_size[1] - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    padded_image = ImageOps.expand(image, padding, fill=int(np.asarray(image).mean()))

    # break into 512x512 tiles
    padded_image = np.asarray(padded_image)
    tiles = []
    for i in range(padded_image.shape[0] // target_size_1d):
        tiles.append([])
        for j in range(padded_image.shape[1] // target_size_1d):
            tiles[i].append(padded_image[i*target_size_1d:(i+1)*target_size_1d, j*target_size_1d:(j+1)*target_size_1d].copy())

    # scale the images to be between 0 and 1 if GV
    for i in range(len(tiles)):
        for j in range(len(tiles[i])):
            tiles[i][j] = tiles[i][j] * 1./255
    return tiles


def overlay_predictions(prepared_tiles, preds, prediction_threshold, background_class_index, labels_output):
    prediction_tiles = []
    for i in range(len(prepared_tiles)):
        prediction_tiles.append([])
        for j in range(len(prepared_tiles[i])):
            prediction_tiles[i].append(np.dstack((prepared_tiles[i][j], prepared_tiles[i][j], prepared_tiles[i][j])))
            prediction_tiles[i][j] = (prediction_tiles[i][j] * 255).astype(int)

            above_threshold_mask = preds[i][j].max(axis=2) > prediction_threshold
            best_class_by_pixel = preds[i][j].argmax(axis=2)
            color_counter = 0
            for class_i in range(preds[i][j].shape[-1]):
                above_threshold_and_best_class = above_threshold_mask & (best_class_by_pixel == class_i)
                if (background_class_index is not None) and (class_i == background_class_index):
                    continue
                if labels_output:
                    prediction_tiles[i][j][above_threshold_and_best_class] = int((color_counter + 1) *
                                                                                 np.floor(255/preds[i][j].shape[-1]))
                else:
                    prediction_tiles[i][j][above_threshold_and_best_class] = class_colors[color_counter]
                color_counter = (color_counter + 1) % len(class_colors)
    return prediction_tiles


def segment_image(model, image, prediction_threshold, target_size_1d, background_class_index, labels_output):

    prepared_tiles = prepare_image(image, target_size_1d)

    preds = []
    for i in range(len(prepared_tiles)):
        preds.append([])
        for j in range(len(prepared_tiles[i])):
            preds[i].append(model.predict(prepared_tiles[i][j].reshape(1, target_size_1d, target_size_1d, 1))[0, :, :, :])

    # make background black if labels only
    if labels_output:
        for i in range(len(prepared_tiles)):
            for j in range(len(prepared_tiles[i])):
                prepared_tiles[i][j] = prepared_tiles[i][j] * 0

    pred_tiles = overlay_predictions(prepared_tiles, preds, prediction_threshold, background_class_index, labels_output)
    stitched_pred = stitch_preds_together(pred_tiles, target_size_1d, labels_output)
    return stitched_pred


def main(gcp_bucket, model_id, background_class_index, stack_id, image_ids, prediction_threshold, labels_output):
    start_dt = datetime.now()

    assert "gs://" in gcp_bucket

    if background_class_index is not None:
        assert background_class_index >= 0

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    run_name = '{}_{}'.format(stack_id, model_id)

    local_model_dir = Path(tmp_directory, 'models', model_id)
    local_model_dir.mkdir(parents=True)
    local_processed_data_dir = Path(tmp_directory, 'processed-data', stack_id)
    local_processed_data_dir.mkdir(parents=True)
    local_inferences_dir = Path(tmp_directory, 'inferences', run_name)
    local_inferences_dir.mkdir(parents=True)
    output_dir = Path(local_inferences_dir, 'output')
    output_dir.mkdir(parents=True)

    os.system("gsutil -m cp -r '{}' '{}'".format(os.path.join(gcp_bucket, 'models', model_id),
                                                 Path(tmp_directory, 'models').as_posix()))
    os.system("gsutil -m cp -r '{}' '{}'".format(os.path.join(gcp_bucket, 'processed-data', stack_id),
                                                 Path(tmp_directory, 'processed-data').as_posix()))

    with Path(local_model_dir, 'config.yaml').open('r') as f:
        train_config = yaml.safe_load(f)['train_config']

    with Path(local_model_dir, 'metadata.yaml').open('r') as f:
        model_metadata = yaml.safe_load(f)

    image_folder = Path(local_processed_data_dir, 'images')
    assert model_metadata['target_size'][0] == model_metadata['target_size'][1]
    target_size_1d = model_metadata['target_size'][0]
    num_classes = model_metadata['num_classes']

    compiled_model = generate_compiled_segmentation_model(
        train_config['segmentation_model']['model_name'],
        train_config['segmentation_model']['model_parameters'],
        num_classes,
        train_config['loss'],
        train_config['optimizer'],
        Path(local_model_dir, "model.hdf5").as_posix())

    if image_ids is None:
        images_list = list(Path(image_folder).iterdir())
    else:
        images_list = image_ids.split(',')

    n_images = len(list(Path(image_folder).iterdir()))
    for i, image_file in enumerate(sorted(Path(image_folder).iterdir())):
        if image_file.parts[-1] in images_list:
            print('Segmenting image {} of {}...'.format(i, n_images))

            image = Image.open(image_file)

            segmented_image = segment_image(compiled_model, image, prediction_threshold,
                                            target_size_1d, background_class_index, labels_output)

            if labels_output:
                image_file_ext = image_file.parts[-1].split('.')[-1]
                segmented_image.save(Path(output_dir, str(image_file.parts[-1].split('.')[0] + '_labels.' + image_file_ext)).as_posix())
            else:
                segmented_image.save(Path(output_dir, image_file.name).as_posix())

    metadata = {
        'gcp_bucket': gcp_bucket,
        'model_id': model_id,
        'background_class_index': background_class_index,
        'stack_id': stack_id,
        'image_ids': image_ids,
        'prediction_threshold': prediction_threshold,
        'labels_output': labels_output,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1)
    }

    with Path(local_inferences_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    os.system("gsutil -m cp -r '{}' '{}'".format(Path(tmp_directory, 'inferences').as_posix(), gcp_bucket))

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--gcp-bucket',
        type=str,
        help='The GCP bucket where the raw data is located and to use to store the processed stacks.')
    argparser.add_argument(
        '--model-id',
        type=str,
        help='The model ID.')
    argparser.add_argument(
        '--background-class-index',
        type=int,
        default=None,
        help='For this model, indicate background class index if used during model training, to exclude background overlay.')
    argparser.add_argument(
        '--stack-id',
        type=str,
        help='The stack ID (must already be processed).')
    argparser.add_argument(
        '--image-ids',
        type=str,
        default=None,
        help='For these images, the corresponding stack ID (must already be processed).')
    argparser.add_argument(
        '--prediction-threshold',
        type=float,
        default=0.5,
        help='Threshold to apply to the prediction to classify a pixel as part of a class.')
    argparser.add_argument(
        '--labels-output',
        type=bool,
        default=False,
        help='If false, will output overlaid image (RGB); if true, will output labels only image (GV).')

    main(**argparser.parse_args().__dict__)
