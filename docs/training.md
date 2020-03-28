# Training

Prerequisite artifacts:
* A dataset (in a GCP bucket) that we will use to train the model
* A train configuration file (on your local machine)

Infrastructure that will be used:
* A GCP bucket where the prepared dataset for training will be accessed from
* A GCP bucket where the trained model will be stored
* A GCP virtual machine to run the training on

## Workflow
1. If the stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Either edit the configuration file `configs/train_config.yaml` or create your own configuration file and place it in the `configs` folder.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`). This will copy the current code base from your local machine to the GCP machine so make sure any changes to the configuration file are saved before this step is run.
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To train a new model, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `python3 train_segmentation_model.py  --gcp-bucket <gcp_bucket> --config-file configs/<config_filename>.yaml`. 
1. Once model training has finished, you should see the folder `<gcp_bucket>/models/<model_ID>-<timestamp>` has been created and populated, where `<model_ID>` was defined in `configs/train_config.yaml`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

### Summary of command line arguments of `train_segmentation_model.py`:

* `--gcp-bucket`:
        type=str,
        help='The GCP bucket where the prepared data is located and to use to store the trained model.'
* `--config-file`:
        type=str,
        help='The location of the train configuration file.'
        
### Example command line inputs:

```
python3 train_segmentation_model.py  --gcp-bucket gs://sandbox --config-file configs/config_sandbox/train-composite_0123.yaml
```

### Notes:

- Batch size of 16 works with P100 GPU (8 for K80), but batch size of 20 is too large for P100 GPU.
- Metrics will be computed based on `global_threshold` in `metrics_utils.py` until threshold training occurs.

### Tips:

- In VM SSH, use `nano` text editor to edit scripts previously uploaded to VM. _E.g.,_ `nano configs/dataset-medium.yaml` to edit text in `dataset-medium.yaml`
- To create a VM without destroying others (assuming `terraform apply` seeks to create & destroy), use `target` flag: `terraform apply -lock=false -target=google_compute_instance.vm[<#>]` to create VM #. Similar syntax with `terraform destroy` to specify target. 


## Issues

* #32 enable pre-trained weights: potentially with `--pre-trained-weights <model_ID>` in command line. If this parameter is excluded, then the default is no pre-training and training operates as it does currently from random initialized weights. Need to think about metadata yet associated with pre-training, though initial thought could be copy everything from that pre-training model training metadata.
