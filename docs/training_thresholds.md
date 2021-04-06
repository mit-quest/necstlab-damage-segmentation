# Training Prediction Thresholds

Prerequisite artifacts:
* A dataset (in a GCP bucket) that we will use to train the model thresholds
* A pretrained damage segmentation model (in a GCP bucket) to train thresholds on, based on performance of a selected optimizing metric

Infrastructure that will be used:
* A GCP bucket where the prepared dataset for training will be accessed from
* A GCP bucket where the trained model thresholds will be stored (same as model bucket)
* A GCP virtual machine to run the threshold training on

## Workflow
1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). This will copy the current code base from your local machine to the GCP machine so make sure any changes to the configuration file are saved before this step is run.
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To train model thresholds, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `python3 train_segmentation_model_prediction_thresholds.py  --gcp-bucket <gcp_bucket> --dataset-directory <dataset_directory> --model-id <model_id> --optimizing-class-metric <optimizing_class_metric> --dataset-downsample-factor <dataset_downsample_factor>`.
1. Once threshold training has finished, you should visit the folder `<gcp_bucket>/models/<model_ID>` (created and populated previously during model training), in which you should see `model_thresholds_<time-stamp>.yaml` that contains optimization results and metadata. The output filename with  `_<timestamp>` supports expected usage that threshold training may be performed multiple times using different optimization setups.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed. 

### Summary of command line arguments of `train_segmentation_model_prediction_thresholds.py`:

* `--gcp-bucket`:
        type=str,
        help='The GCP bucket where the raw data is located and to use to store the processed stacks.'
* `--dataset-directory`:
        type=str,
        help='The dataset ID + "/validation" or "/test".'
* `--model-id`:
        type=str,
        help='The model ID.')
* `--batch-size`:
        type=int,
        default=16,
        help='The batch size to use during inference.'
* `--optimizing-class-metric`:
        type=str,
        default='iou_score',
        help='Use single class metric if training prediction threshold.'
* `--dataset-downsample-factor`:
        type=float,
        default=1.0,
        help='Accelerate optimization via using subset of dataset.'
* `'--random-module-global-seed`:
        type=int,
        default=None,
        help='The setting of random.seed(global seed), where global seed is int or default None (no seed given).')
* `'--numpy-random-global-seed`:
        type=int,
        default=None,
        help='The setting of np.random.seed(global seed), where global seed is int or default None (no seed given).')
* `'--tf-random-global-seed`:
        type=int,
        default=None,
        help='The setting of tf.random.set_seed(global seed), where global seed is int or default None (no seed given).')
* `'--message`:
        type=str,
        default=None,
        help='A str message the used wants to leave, the default is None.')  

### Example command line inputs:

```
python3 train_segmentation_model_prediction_thresholds.py --gcp-bucket gs://sandbox --dataset-directory dataset-composite_0123/validation --model-id segmentation-model-composite_0123_20200321T154533Z --batch-size 16 --optimizing-class-metric iou_score --dataset-downsample-factor 0.1

python3 train_segmentation_model_prediction_thresholds.py --gcp-bucket gs://sandbox --dataset-directory dataset-composite_0123/test --model-id segmentation-model-composite_0123_20200321T154533Z --batch-size 16 --optimizing-class-metric f1_score --dataset-downsample-factor 0.5
```
        
### Notes:

- Batch size of 16 works with P100 GPU (8 for K80), but batch size of 20 is too large for P100 GPU.
- Selected `optimizing-class-metric` needs to be among those setup in `models.py` before `model.compile`.
- For class metrics (binarized), there is no difference between one-hot and non-one-hot.
- Optimization (via `scipy` `minimize_scalar`) configured within `models.py`.
- For multi-class models, each class threshold is trained independently. If one class threshold fails to converge, then program will exit with error. Adjust optimization configuration and retry.

### Tips:

- In VM SSH, use `nano` text editor to edit scripts previously uploaded to VM. _E.g.,_ `nano configs/dataset-medium.yaml` to edit text in `dataset-medium.yaml`
- To create a VM without destroying others (assuming `terraform apply` seeks to create & destroy), use `target` flag: `terraform apply -lock=false -target=google_compute_instance.vm[<#>]` to create VM #. Similar syntax with `terraform destroy` to specify target. 
