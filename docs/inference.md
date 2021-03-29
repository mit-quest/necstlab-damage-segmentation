# Inference

Prerequisite artifacts:
* Unannotated stacks (in a GCP bucket) that we wish to perform damage segmentation on
* A pretrained damage segmentation model (in a GCP bucket) to use for inference

Infrastructure that will be used:
* A GCP bucket where the stored unsegmented stacks will be accessed from
* A GCP bucket where the stacks with inferred damage segmentation will be stored
* A GCP virtual machine to run inference on

## Workflow

1. If the unsegmented stacks are not in a GCP bucket, see the previous workflow `Copying the raw data into the cloud for storage and usage`.
1. Use Terraform to start the appropriate GCP virtual machine (`terraform apply`).
1. Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been created named `<project_name>-<user_name>` where `<project_name>` is the name of your GCP project and `<user_name>` is your GCP user name.
1. To infer (segment) the damage of the stacks, SSH into the virtual machine `<project_name>-<user_name>`, start tmux (`tmux`), `cd` into the code directory (`cd necstlab-damage-segmentation`), and run `python3 infer_segmentation.py --gcp-bucket <gcp_bucket> --stack-id <stack_id> --image-ids <image_ids> --model-id <model_id>`. _Optional:_ Use `--trained-thresholds-id <model_thresholds>.yaml` to test using pretrained class thresholds. 
1. Once inference has finished, you should see the folder `<gcp_bucket>/inferences/<inference_ID>` has been created and populated, where `<inference_ID>` is `<stack_id>_<model_id>`.  Since inference can be performed multiple times (on same image(s)) using different prediction thresholds, output `metadata.yaml` filename and `output` directory name are appended with `_<timestamp>`.
1. Use Terraform to terminate the appropriate GCP virtual machine (`terraform destroy`). Once Terraform finishes, you can check the GCP virtual machine console to ensure a virtual machine has been destroyed.

### Summary of command line arguments of `infer_segmentation.py`:

* `--gcp-bucket`:
        type=str,
        help='The GCP bucket where the raw data is located and to use to store the processed stacks.'
* `--model-id`:
        type=str,
        help='The model ID.'
* `--background-class-index`:
        type=int,
        default=None,
        help='For this model, indicate background class index if used during model training, to exclude background overlay.'
* `--stack-id`:
        type=str,
        help='The stack ID (must already be processed).'
* `--image-ids`:
        type=str,
        default=None,
        help='For these images, the corresponding stack ID (must already be processed).'
* `--user-specified-prediction-thresholds`:
        type=float,
        nargs='+',
        default=None,
        help='Threshold(s) to apply to the prediction to classify a pixel as part of a class. _E.g.,_ 0.5 or 0.5 0.3 0.6'
* `--labels-output`:
        type=str,
        default='False',
        help='If false, will output overlaid image (RGB); if true, will output labels only image (GV).'
* `--pad-output`:
        type=str,
        default='False',
        help='If false, will output inference identical to input image size.'
* `--trained-thresholds-id`:
        type=str,
        default=None,
        help='The specified trained thresholds file id.'

### Example command line inputs:

```
python3 infer_segmentation.py --gcp-bucket gs://sandbox --stack-id composite_1234 --model-id segmentation-model-composite_0123_20200321T154533Z --image-ids composite_1234-209.tif,composite_1234-2089.tif,composite_1234-2189.tif --labels-output False --pad-output False --trained-thresholds-id model_thresholds_20200321T181016Z.yaml

python3 infer_segmentation.py --gcp-bucket gs://sandbox --stack-id composite_1234 --model-id segmentation-model-composite_0123_20200321T154533Z --labels-output True --pad-output False --user-specified-prediction-thresholds 0.5

python3 infer_segmentation.py --gcp-bucket gs://sandbox --stack-id composite_1234 --model-id segmentation-model-composite_0123_20200321T154533Z --labels-output True --pad-output True --user-specified-prediction-thresholds 0.5 0.25 0.02        (3-class example)

python3 infer_segmentation.py --gcp-bucket gs://sandbox --stack-id composite_1234 --model-id segmentation-model-composite_0123_20200321T154533Z --labels-output True --pad-output True
```

### Notes:

- Predictions will be based on `global_threshold` in `metrics_utils.py` unless pretrained or arbitrary thresholds are specified.
- Default assumption is that background is NOT a class explicitly trained on by model

### Tips:

- In VM SSH, use `nano` text editor to edit scripts previously uploaded to VM. _E.g.,_ `nano configs/dataset-medium.yaml` to edit text in `dataset-medium.yaml`
- To create a VM without destroying others (assuming `terraform apply` seeks to create & destroy), use `target` flag: `terraform apply -lock=false -target=google_compute_instance.vm[<#>]` to create VM #. Similar syntax with `terraform destroy` to specify target. 
- `infer_segmentation.py` looks for `<stack_id>` in `processed-data` inside bucket
- More secure local copying: download inference locally by (e.g.): 
    ```
    until gsutil -m cp -c -L log.txt -r gs://necstlab-sandbox/inferences/<inference_ID> <local_storage_location>; do
        sleep 1
    done
    ```
     Or `-L filecopy.log -r` or `-L cp.log -r`
- Less secure local copying: download inference locally by (e.g.) `gsutil -m cp -n -r gs://necstlab-sandbox/inferences/<inference_ID> <local_storage_location>`
