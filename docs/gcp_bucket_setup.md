# Setting up your GCP bucket

To store all of the artifacts (data, models, results, etc.) that are required for and result from the workflows, we will use a GCP bucket. To set up a GCP bucket to be used for these workflows, see the instructions [here](). TODO: add link/link content

As you run the workflows you'll see the following directory structure be automatically created and populated inside of your bucket:
```
<GCP bucket>/
    datasets/         (this is where any prepared datasets for training will be stored)
        <dataset_ID>/
            test/
                annotations/
                    ...
                images/
                    ...
            train/
                annotations/
                    ...
                images/
                    ...
            validate/
                annotations/
                    ...
                images/
                    ...
            config.yaml
            metadata.yaml
    inferences/       (this is where any stack segmentations will be stored)
        <inference_ID>-<timestamp>/
            logs/
            output/
                ...
            metadata.yaml
    models/           (this is where any trained segmentation models will be stored)
        <model_ID>-<timestamp>/
            logs/
            plots/
                ...
            config.yaml
            model.hdf5
            metadata.yaml
            metrics.csv
    processed-data/
        <stack_ID>/
            annotations/
                ...
            images/
                ...
            config.yaml
            metadata.yaml
    raw-data/         (this is where any raw data files will be stored)
        ...
    tests/         (this is where any analysis that results from testing will be stored)
        <test_ID>/
            logs/
            plots/
                ...
            config.yaml
            metrics.csv
            metadata.yaml
```
where `<test_ID>`, `<dataset_ID>`, `<inference_ID>`, and `<model_ID>` are defined inside of configuration files and `<timestamp>` is the time the process was started and is automatically generated. The `<stack_ID>`s will be created as raw data is moved into the cloud. 
