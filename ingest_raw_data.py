import os
from pathlib import Path
from PIL import Image
import shutil
from datetime import datetime
import pytz
import gcp_utils
import yaml
import git

# we will now have two metadata files, one for the annotations folder and one for the images folder
metadata_file_name_annotations = 'metadata_annotations.yaml'
metadata_file_name_images = 'metadata_images.yaml'
#metadata_file_name = 'metadata.yaml'

tmp_directory = Path('./tmp')

def process_zips(gcp_bucket):

    files = gcp_utils.list_files(gcp_bucket.split('gs://')[1], 'raw-data')

    for zipped_stack in files:
        if zipped_stack == 'raw-data/':
            continue
        process_zip(gcp_bucket, os.path.join(gcp_bucket, zipped_stack))


def process_zip(gcp_bucket, zipped_stack):

    start_dt = datetime.now()

    assert "gs://" in zipped_stack
    assert "gs://" in gcp_bucket

    # clean up the tmp directory
    try:
        shutil.rmtree(tmp_directory.as_posix())
    except FileNotFoundError:
        pass
    tmp_directory.mkdir()

    is_annotation = 'dmg' in zipped_stack

    stack_id = Path(zipped_stack).name.split('.')[0]
    split_strings = ['_8bit', '-', '_dmg']
    for s in split_strings:
        stack_id = stack_id.split(s)[0]

    stack_dir = Path(tmp_directory, stack_id)

    metadata_file_name = metadata_file_name_annotations if is_annotation else metadata_file_name_images
    check_metadata_file = gcp_utils.remote_folder_exists(os.path.join(gcp_bucket, 'processed-data'),
                                                         '/'.join([stack_id]),
                                                         sample_file_name=metadata_file_name):

    if not is_annotation and check_metadata_file:
        print("{} has already been processed! Skipping...".format(os.path.join(stack_id, "images")))

    elif is_annotation and check_metadata_file:

        print("{} has already been processed! Skipping...".format(os.path.join(stack_id, "annotations")))

    else:
        os.system("gsutil -m cp -r '{}' '{}'".format(zipped_stack, tmp_directory.as_posix()))

        os.system("7za x -y -o'{}' '{}'".format(stack_dir.as_posix(), Path(tmp_directory, Path(zipped_stack).name).as_posix()))
        os.remove(Path(tmp_directory, Path(zipped_stack).name).as_posix())
        unzipped_dir = next(stack_dir.iterdir())

        original_number_of_files_in_zip = len(list(unzipped_dir.iterdir()))

        for f in Path(unzipped_dir).iterdir():
            if f.name[-4:] != '.tif':
                # remove any non-image files
                os.remove(f.as_posix())
            else:
                # convert all images to greyscale (some are already and some aren't)
                Image.open(f).convert("L").save(f)

        shutil.move(unzipped_dir.as_posix(),
                    Path(unzipped_dir.parent, 'annotations' if is_annotation else 'images').as_posix())

        # get metadata file, if exists
        os.system("gsutil -m cp -r '{}' '{}'".format(os.path.join(gcp_bucket, 'processed-data/', stack_id, metadata_file_name),
                                                     Path(tmp_directory, stack_id).as_posix()))

        number_of_images = len(list(Path(unzipped_dir.parent, 'annotations' if is_annotation else 'images').iterdir()))

        if original_number_of_files_in_zip == number_of_images:  # this guarantees that the images were all copied localy
            try:
                with Path(tmp_directory, stack_id, metadata_file_name).open('r') as f:
                    metadata = yaml.safe_load(f)
            except FileNotFoundError:
                metadata = {}

            metadata.update({'annotations' if is_annotation else 'images': {
                'gcp_bucket': gcp_bucket,
                'zipped_stack_file': zipped_stack,
                'created_datetime': datetime.now(pytz.UTC).strftime('%Y%m%dT%H%M%SZ'),
                'original_number_of_files_in_zip': original_number_of_files_in_zip,
                'number_of_images': number_of_images,
                'git_hash': git.Repo(search_parent_directories=True).head.object.hexsha},
                'elapsed_minutes': round((datetime.now() - start_dt).total_seconds() / 60, 1)
            })

            with Path(tmp_directory, stack_id, metadata_file_name).open('w') as f:
                yaml.safe_dump(metadata, f)

            # os.system("gsutil -m cp -n -r '{}' '{}'".format(unzipped_dir.parent.as_posix(),
            #                                                os.path.join(gcp_bucket, 'processed-data/')))

            # instead of copying it, we should sync it
            os.system("gsutil -m rsync -r '{}' '{}'".format(unzipped_dir.parent.as_posix(),
                                                            os.path.join(gcp_bucket, 'processed-data/')))

            print('\n Ingest Raw Data Metadata:')
            print(metadata)
            print('\n')

        else:
            print('Ingestion of ' + stack_dir + ' was not completed. ' + original_number_of_files_in_zip + 'files were detected in the zipped folder and ' + original_number_of_files_in_zip + 'were moved.')

        shutil.rmtree(tmp_directory.as_posix())


if __name__ == "__main__":
    import sys
    import argparse

    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument(
        '--gcp-bucket',
        type=str,
        help='The GCP bucket where the raw data is located and to use to store the processed stacks.')

    argparser.add_argument(
        '--zipped-stack',
        type=str,
        default='',
        help='The zipped stack (.zip or .7z) to be processed.')

    kw_args = argparser.parse_args().__dict__

    if kw_args['zipped_stack'] == '':
        process_zips(gcp_bucket=kw_args['gcp_bucket'])
    else:
        process_zip(gcp_bucket=kw_args['gcp_bucket'],
                    zipped_stack=kw_args['zipped_stack'])
