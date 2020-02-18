import os
import shutil
import random
import math
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import git
from datetime import datetime
import pytz
from gcp_utils import remote_folder_exists


metadata_file_name = 'metadata.yaml'
tmp_directory = Path('./tmp')

max_tries_rand_crop_per_class = 1000  # stopgap soln
max_num_crop_per_img = 36  # suits 4600 x 2048 img with 512 x 512 target


def copy_processed_data_locally_if_missing(scans, processed_data_remote_source, processed_data_local_dir):
    Path(processed_data_local_dir).mkdir(parents=True, exist_ok=True)
    for scan in scans:
        if scan not in [p.name for p in Path(processed_data_local_dir).iterdir()]:
            os.system("gsutil -m cp -r '{}/{}' '{}'".format(processed_data_remote_source, scan, processed_data_local_dir))


def copy_and_downsample_processed_data_to_preparation_if_missing(scans, processed_data_local_dir,
                                                                 data_prep_local_dir, downsampling_params):
    Path(data_prep_local_dir, 'downsampled').mkdir(parents=True, exist_ok=True)
    assert 'type' in downsampling_params
    for scan in scans:
        if scan not in [p.name for p in Path(data_prep_local_dir, 'downsampled').iterdir()]:
            scan_image_files = sorted(Path(processed_data_local_dir, scan, 'images').iterdir())
            scan_annotation_files = sorted(Path(processed_data_local_dir, scan, 'annotations').iterdir())
            assert len(scan_image_files) == len(scan_annotation_files)
            assert 'num_skip_beg_slices' in downsampling_params
            assert 'num_skip_end_slices' in downsampling_params
            assert len(scan_image_files) > (downsampling_params['num_skip_beg_slices']
                                            + downsampling_params['num_skip_end_slices'])
            assert downsampling_params['num_skip_beg_slices'] >= 0
            assert downsampling_params['num_skip_end_slices'] >= 0
            total_images = (len(scan_image_files)
                            - downsampling_params['num_skip_beg_slices']
                            - downsampling_params['num_skip_end_slices'])

            if 'number_of_images' in downsampling_params:
                num_images = downsampling_params['number_of_images']
                assert num_images <= total_images
                assert 'frac' not in downsampling_params
            elif 'frac' in downsampling_params:
                num_images = math.ceil(downsampling_params['frac'] * total_images)
                assert num_images <= total_images
                assert 'number_of_images' not in downsampling_params
            else:  # all eligible images used
                num_images = total_images

            if downsampling_params['type'] == 'None':
                file_inds_to_copy = range(downsampling_params['num_skip_beg_slices'],
                                          total_images + downsampling_params['num_skip_beg_slices'])
            elif downsampling_params['type'] == 'random':
                file_inds_to_copy = random.sample(range(downsampling_params['num_skip_beg_slices'],
                                                        total_images + downsampling_params['num_skip_beg_slices']),
                                                  k=num_images)
            elif downsampling_params['type'] == 'linear':
                file_inds_to_copy = np.floor(np.linspace(downsampling_params['num_skip_beg_slices'],
                                                         total_images + downsampling_params['num_skip_beg_slices'] - 1,
                                                         num_images)).astype(int)
            elif downsampling_params['type'] == 'from_start':
                file_inds_to_copy = range(downsampling_params['num_skip_beg_slices'], num_images)
            elif downsampling_params['type'] == 'from_end':
                file_inds_to_copy = range(total_images + downsampling_params['num_skip_beg_slices'] - num_images,
                                          total_images + downsampling_params['num_skip_beg_slices'])
            else:
                raise ValueError("Unknown downsampling type: {}".format(downsampling_params['type']))

            Path(data_prep_local_dir, 'downsampled', scan, 'images').mkdir(parents=True)
            Path(data_prep_local_dir, 'downsampled', scan, 'annotations').mkdir(parents=True)
            for image_ind in file_inds_to_copy:
                shutil.copy(scan_image_files[image_ind].as_posix(),
                            Path(data_prep_local_dir, 'downsampled', scan, 'images', scan_image_files[
                                image_ind].name).as_posix())
                shutil.copy(scan_annotation_files[image_ind].as_posix(),
                            Path(data_prep_local_dir, 'downsampled', scan, 'annotations', scan_annotation_files[
                                image_ind].name).as_posix())


# adapted from: https://github.com/matterport/Mask_RCNN/issues/230
def random_crop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask


def resize_and_crop(data_prep_local_dir, target_size, image_cropping_params, class_annotation_mapping):
    Path(data_prep_local_dir, 'resized').mkdir(parents=True, exist_ok=True)
    assert 'type' in image_cropping_params
    assert target_size[0] > 0
    assert target_size[1] > 0
    for scan in [p.name for p in Path(data_prep_local_dir, 'downsampled').iterdir()]:
        if scan not in [p.name for p in Path(data_prep_local_dir, 'resized').iterdir()]:
            scan_image_files = sorted(Path(data_prep_local_dir, 'downsampled', scan, 'images').iterdir())
            scan_annotation_files = sorted(Path(data_prep_local_dir, 'downsampled', scan, 'annotations').iterdir())
            assert len(scan_image_files) == len(scan_annotation_files)
            Path(data_prep_local_dir, 'resized', scan, 'images').mkdir(parents=True)
            Path(data_prep_local_dir, 'resized', scan, 'annotations').mkdir(parents=True)
            for image_ind in range(len(scan_image_files)):
                image = Image.open(scan_image_files[image_ind])
                annotation = Image.open(scan_annotation_files[image_ind])
                if image_cropping_params['type'] == 'None':
                    image.thumbnail(target_size)
                    annotation.thumbnail(target_size)
                    image.save(Path(data_prep_local_dir, 'resized', scan, 'images', scan_image_files[
                        image_ind].name).as_posix())
                    annotation.save(Path(data_prep_local_dir, 'resized', scan, 'annotations', scan_annotation_files[
                        image_ind].name).as_posix())
                elif image_cropping_params['type'] == 'random':
                    assert 'num_per_image' in image_cropping_params
                    assert image_cropping_params['num_per_image'] > 0
                    assert image_cropping_params['num_per_image'] <= max_num_crop_per_img
                    for counter_crop in range(image_cropping_params['num_per_image']):
                        image_crop, annotation_crop = random_crop(np.asarray(image), np.asarray(annotation), target_size[0], target_size[1])
                        image_crop = Image.fromarray(image_crop)
                        annotation_crop = Image.fromarray(annotation_crop)
                        image_crop.save((Path(data_prep_local_dir, 'resized', scan, 'images', scan_image_files[
                            image_ind].name).as_posix()).replace('.', ('_crop' + str(counter_crop) + '.')))
                        annotation_crop.save((Path(data_prep_local_dir, 'resized', scan, 'annotations', scan_annotation_files[
                            image_ind].name).as_posix()).replace('.', ('_crop' + str(counter_crop) + '.')))
                elif image_cropping_params['type'] == 'linear':  # do not train with pad, some overlap okay (still aug'd)
                    assert 'num_per_image' in image_cropping_params
                    assert image_cropping_params['num_per_image'] > 0
                    assert image_cropping_params['num_per_image'] <= max_num_crop_per_img
                    img = np.asarray(image)
                    mask = np.asarray(annotation)
                    num_tiles_hor = np.int(np.ceil(img.shape[1] / target_size[0]))
                    num_tiles_ver = np.int(np.ceil(img.shape[0] / target_size[1]))
                    horiz_counter = 0  # gets reset (cyclic)
                    vert_counter = 0  # does not get reset (not cyclic)
                    for counter_crop in range(image_cropping_params['num_per_image']):  # L to R, then move down by trgt
                        if horiz_counter < (num_tiles_hor - 1):
                            x_crop_lhs = horiz_counter * target_size[0]
                        elif horiz_counter == (num_tiles_hor - 1):
                            x_crop_lhs = img.shape[1] - target_size[0]
                        if vert_counter < (num_tiles_ver - 1):
                            y_crop_top = vert_counter * target_size[1]
                        elif vert_counter == (num_tiles_ver - 1):
                            y_crop_top = img.shape[0] - target_size[1]
                        image_crop = img[y_crop_top:y_crop_top+target_size[1], x_crop_lhs:x_crop_lhs+target_size[0]]
                        annotation_crop = mask[y_crop_top:y_crop_top+target_size[1], x_crop_lhs:x_crop_lhs+target_size[0]]
                        image_crop = Image.fromarray(image_crop)
                        annotation_crop = Image.fromarray(annotation_crop)
                        image_crop.save((Path(data_prep_local_dir, 'resized', scan, 'images', scan_image_files[
                            image_ind].name).as_posix()).replace('.', ('_crop' + str(counter_crop) + '.')))
                        annotation_crop.save(
                            (Path(data_prep_local_dir, 'resized', scan, 'annotations', scan_annotation_files[
                                image_ind].name).as_posix()).replace('.', ('_crop' + str(counter_crop) + '.')))
                        if horiz_counter == (num_tiles_hor - 1):  # reached rhs of img, move back to lhs & down by trgt
                            horiz_counter = 0
                            vert_counter += 1
                            if vert_counter > (num_tiles_ver - 1):
                                break  # regardless of num crops input, the bot rhs of img has been reached
                        else:
                            horiz_counter += 1
                        assert horiz_counter < num_tiles_hor  # prevent movement off image
                        assert vert_counter < num_tiles_ver   # prevent movement off image
                elif image_cropping_params['type'] == 'all':  # do not train with pad, some overlap okay (still aug'd)
                    img = np.asarray(image)
                    mask = np.asarray(annotation)
                    num_tiles_hor = np.int(np.ceil(img.shape[1] / target_size[0]))
                    num_tiles_ver = np.int(np.ceil(img.shape[0] / target_size[1]))
                    counter_crop = 0
                    for vert_counter in range(num_tiles_ver):  # L to R, then move down by trgt
                        for horiz_counter in range(num_tiles_hor):
                            if horiz_counter < (num_tiles_hor - 1):
                                x_crop_lhs = horiz_counter * target_size[0]
                            elif horiz_counter == (num_tiles_hor - 1):
                                x_crop_lhs = img.shape[1] - target_size[0]
                            if vert_counter < (num_tiles_ver - 1):
                                y_crop_top = vert_counter * target_size[1]
                            elif vert_counter == (num_tiles_ver - 1):
                                y_crop_top = img.shape[0] - target_size[1]
                            image_crop = img[y_crop_top:y_crop_top+target_size[1], x_crop_lhs:x_crop_lhs+target_size[0]]
                            annotation_crop = mask[y_crop_top:y_crop_top+target_size[1], x_crop_lhs:x_crop_lhs+target_size[0]]
                            image_crop = Image.fromarray(image_crop)
                            annotation_crop = Image.fromarray(annotation_crop)
                            image_crop.save((Path(data_prep_local_dir, 'resized', scan, 'images', scan_image_files[
                                image_ind].name).as_posix()).replace('.', ('_crop' + str(counter_crop) + '.')))
                            annotation_crop.save(
                                (Path(data_prep_local_dir, 'resized', scan, 'annotations', scan_annotation_files[
                                    image_ind].name).as_posix()).replace('.', ('_crop' + str(counter_crop) + '.')))
                            counter_crop += 1
                elif image_cropping_params['type'] == 'class':  # 'smart' crop: do not train with pad, some overlap okay (still aug'd)
                    assert 'num_pos_per_class' in image_cropping_params
                    assert 'num_neg_per_class' in image_cropping_params
                    assert image_cropping_params['num_pos_per_class'] > 0  # logical choice
                    assert image_cropping_params['num_pos_per_class'] <= max_num_crop_per_img
                    assert image_cropping_params['num_neg_per_class'] >= 0  # logical choice if 0
                    assert image_cropping_params['num_neg_per_class'] <= max_num_crop_per_img
                    assert 'min_num_class_pos_px' in image_cropping_params
                    for c, gvs_in_c in class_annotation_mapping.items():
                        assert "class_" in c
                        assert "_annotation_GVs" in c, "'_annotation_GVs' must be in the class name to indicate these are gray values"
                        class_name = c[:-len('_annotation_GVs')]
                        assert image_cropping_params['min_num_class_pos_px'][class_name + '_pos_px'] > 0  # logical choice, min thresh that defines each class pos pixel qty per crop
                        if np.size(np.asarray(annotation)[np.isin(np.asarray(annotation), gvs_in_c)]) >= image_cropping_params['min_num_class_pos_px'][class_name + '_pos_px']:  # if class-pos is present in full annot
                            for counter_classpos_crop in range(image_cropping_params['num_pos_per_class']):
                                flag_crop_pass = 0
                                counter_classpos_tries = 0  # getting this far presents no guarantees of ok crop selection
                                while flag_crop_pass == 0 and counter_classpos_tries < max_tries_rand_crop_per_class:  # stopgap soln:
                                    image_crop, annotation_crop = random_crop(np.asarray(image), np.asarray(annotation), target_size[0], target_size[1])
                                    if np.size(annotation_crop[np.isin(annotation_crop, gvs_in_c)]) >= image_cropping_params['min_num_class_pos_px'][class_name + '_pos_px']:
                                        image_crop = Image.fromarray(image_crop)
                                        annotation_crop = Image.fromarray(annotation_crop)
                                        image_crop.save((Path(data_prep_local_dir, 'resized', scan, 'images', scan_image_files[
                                            image_ind].name).as_posix()).replace('.', ('_pos_' + str(class_name) + '_crop' + str(counter_classpos_crop) + '.')))
                                        annotation_crop.save((Path(data_prep_local_dir, 'resized', scan, 'annotations', scan_annotation_files[
                                            image_ind].name).as_posix()).replace('.', ('_pos_' + str(class_name) + '_crop' + str(counter_classpos_crop) + '.')))
                                        flag_crop_pass = 1
                                    counter_classpos_tries += 1
                        if np.size(np.asarray(annotation)[np.isin(np.asarray(annotation), gvs_in_c, invert=True)]) >= target_size[0] * target_size[1]:  # if class-neg of target size is present
                            for counter_classneg_crop in range(image_cropping_params['num_neg_per_class']): # won't run if `num_neg_per_class` is 0
                                flag_crop_pass = 0
                                counter_classneg_tries = 0
                                while flag_crop_pass == 0 and counter_classneg_tries < max_tries_rand_crop_per_class:  # stopgap soln
                                    image_crop, annotation_crop = random_crop(np.asarray(image), np.asarray(annotation), target_size[0], target_size[1])
                                    if np.all(np.isin(annotation_crop, gvs_in_c, invert=True)):
                                        image_crop = Image.fromarray(image_crop)
                                        annotation_crop = Image.fromarray(annotation_crop)
                                        image_crop.save((Path(data_prep_local_dir, 'resized', scan, 'images', scan_image_files[
                                            image_ind].name).as_posix()).replace('.', ('_neg_' + str(class_name) + '_crop' + str(counter_classneg_crop) + '.')))
                                        annotation_crop.save((Path(data_prep_local_dir, 'resized', scan, 'annotations', scan_annotation_files[
                                            image_ind].name).as_posix()).replace('.', ('_neg_' + str(class_name) + '_crop' + str(counter_classneg_crop) + '.')))
                                        flag_crop_pass = 1
                                    counter_classneg_tries += 1
                else:
                    raise ValueError("Image cropping type: {}".format(image_cropping_params['type']))


def create_class_masks(data_prep_local_dir, class_annotation_mapping):
    for scan in [p.name for p in Path(data_prep_local_dir, 'resized').iterdir()]:
        if 'masks' not in [p.name for p in Path(data_prep_local_dir, 'resized', scan).iterdir()]:
            scan_annotation_files = sorted(Path(data_prep_local_dir, 'resized', scan, 'annotations').iterdir())
            for c, gvs_in_c in class_annotation_mapping.items():
                assert "class_" in c
                assert "_annotation_GVs" in c, "'_annotation_GVs' must be in the class name to indicate these are grayvalues"
                class_name = c[:-len('_annotation_GVs')]
                Path(data_prep_local_dir, 'resized', scan, 'masks', class_name).mkdir(parents=True, exist_ok=True)
                for scan_annotation_file in scan_annotation_files:
                    annotation = np.asarray(Image.open(scan_annotation_file))
                    mask = np.zeros(annotation.shape, dtype=bool)
                    for gv in gvs_in_c:
                        assert type(gv) is int
                        mask += (annotation == gv)
                    mask = Image.fromarray(mask.astype('uint8'))
                    mask.save(Path(data_prep_local_dir, 'resized', scan, 'masks',
                                   class_name, scan_annotation_file.name).as_posix())


def recursive_copy_directory(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError:
        shutil.copy(src, dst)


def split_prepared_data(data_prep_local_dir, prepared_dataset_local_dir, dataset_split):
    for split, scans_in_split in dataset_split.items():
        if split not in [p.name for p in prepared_dataset_local_dir.iterdir()]:
            images_dest_dir = Path(prepared_dataset_local_dir, split, 'images')
            images_dest_dir.mkdir(parents=True)
            for scan in scans_in_split:
                for file in Path(data_prep_local_dir, 'resized', scan, 'images').iterdir():
                    shutil.copy(file.as_posix(), Path(images_dest_dir, file.name).as_posix())
                for c in [p.name for p in Path(data_prep_local_dir, 'resized', scan, 'masks').iterdir()]:
                    mask_class_dest_dir = Path(prepared_dataset_local_dir, split, 'masks', c)
                    mask_class_dest_dir.mkdir(parents=True, exist_ok=True)
                    for file in Path(data_prep_local_dir, 'resized', scan, 'masks', c).iterdir():
                        shutil.copy(file.as_posix(), Path(mask_class_dest_dir, file.name).as_posix())


def copy_dataset_to_remote_dest(prepared_dataset_location, prepared_dataset_remote_dest, dataset_id):
    print('Copying dataset {} to gcp bucket...'.format(dataset_id))
    os.system("gsutil -m cp -r '{}' '{}'".format(prepared_dataset_location.as_posix(),
                                                 os.path.join(prepared_dataset_remote_dest, dataset_id)))


def prepare_dataset(gcp_bucket, config_file):
    """
    The ordering of the steps is important because it assumes a certain directory structure is progressively created!
    """
    start_dt = datetime.now()

    with Path(config_file).open('r') as f:
        dataset_config = yaml.safe_load(f)['dataset_config']

    dataset_id = Path(config_file).name.split('.')[0]

    assert "gs://" in gcp_bucket

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    processed_data_remote_source = os.path.join(gcp_bucket, 'processed-data')
    processed_data_local_dir = Path(tmp_directory, 'processed-data')
    processed_data_local_dir.mkdir()

    data_prep_local_dir = Path(tmp_directory, 'preparing')
    data_prep_local_dir.mkdir()

    prepared_dataset_local_dir = Path(tmp_directory, 'datasets', )
    prepared_dataset_local_dir.mkdir(parents=True)

    prepared_dataset_remote_dest = os.path.join(gcp_bucket, 'datasets')

    with Path(prepared_dataset_local_dir, 'config.yaml').open('w') as f:
        yaml.safe_dump({'dataset_config': dataset_config}, f)

    all_scans = []
    for _, scans in dataset_config['dataset_split'].items():
        all_scans += scans
    all_scans = sorted(set(all_scans))

    assert not remote_folder_exists(prepared_dataset_remote_dest, dataset_id)

    copy_processed_data_locally_if_missing(all_scans, processed_data_remote_source, processed_data_local_dir)

    copy_and_downsample_processed_data_to_preparation_if_missing(
        all_scans, processed_data_local_dir, data_prep_local_dir, dataset_config['stack_downsampling'])

    resize_and_crop(data_prep_local_dir, dataset_config['target_size'], dataset_config['image_cropping'], dataset_config['class_annotation_mapping'])

    create_class_masks(data_prep_local_dir, dataset_config['class_annotation_mapping'])

    split_prepared_data(data_prep_local_dir, prepared_dataset_local_dir, dataset_config['dataset_split'])

    metadata = {
        'gcp_bucket': gcp_bucket,
        'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
        'number_of_images': {
            'train': len(list(Path(prepared_dataset_local_dir, 'train', 'images').iterdir())),
            'validation': len(list(Path(prepared_dataset_local_dir, 'validation', 'images').iterdir())),
        },
        'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha,
        'original_config_filename': config_file,
        'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1)
    }
    try:
        metadata['number_of_images']['test'] = len(list(Path(prepared_dataset_local_dir, 'test', 'images').iterdir()))
    except FileNotFoundError:
        pass  # does not necessarily have to be test data

    with Path(prepared_dataset_local_dir, metadata_file_name).open('w') as f:
        yaml.safe_dump(metadata, f)

    copy_dataset_to_remote_dest(prepared_dataset_local_dir, prepared_dataset_remote_dest, dataset_id)

    shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import argparse
    import sys

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument(
        '--gcp-bucket',
        type=str,
        help='The GCP bucket where the processed data is located and to use to store the prepared dataset.')
    argparser.add_argument(
        '--config-file',
        type=str,
        help='The location of the data preparation configuration file.')

    prepare_dataset(**argparser.parse_args().__dict__)
