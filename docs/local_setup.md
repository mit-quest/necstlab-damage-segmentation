# Setting up your local machine

## Terraform

To programmatically set up and destroy cloud resources (virtual machines, buckets, etc.), we will use a tool called Terraform. For instructions on how to install Terraform, see [here](). TODO: add link/link content

## GCP

All of the workflows use Google Cloud Platform (GCP) for storage (buckets) and compute (virtual machines). To allow the code to programmatically interact with GCP, we will set up a Software Development Kit (SDK) on your local machine. To install the GCP SDK follow the instructions [here](). TODO: add link/link content

To set up and destroy virtual machines, Terraform requires access to GCP. For instructions on how to download GCP credentials for Terraform, see [here](). TODO: add link/link content

Edit the `terraform.tfvars` file with your `username`, `gcp_key_file_location`, `public_ssh_key_location`, `private_ssh_key_location`. For more information on how to generate a public and private SSH key pair, see [here](https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

## Clone this repository locally

To copy this repository locally, in a terminal window, enter and clone the repository using the command: `git clone git@github.com:mit-quest/necstlab-damage-segmentation.git`. If you do not have `git` installed, see [here]() for installation instructions. TODO: add link/link content

All commands will assume to be run from the `necstlab-damage-segmentation` directory, which you can `cd` into using: `cd necstlab-damage-segmentation`
