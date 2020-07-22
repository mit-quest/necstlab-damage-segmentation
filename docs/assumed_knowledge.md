# Assumed knowledge
The workflows contained in this repository assume:
* You know how to interact within terminal and use Git Commands. If you are unfamilair, see [here]() for a  reference/tutorial
* You know how to check the status of GCP virtual machines using the GCP compute engine dashboard. If you are unfamiliar with how to do this, see [here](https://cloud.google.com/compute/docs/instances) for instructions. 
* You know how to SSH into a GCP virtual machine. If you are unfamiliar with how to do this, see [here](https://cloud.google.com/compute/docs/instances/connecting-to-instance) for instructions to use local SSH client or [here](https://cloud.google.com/compute/docs/ssh-in-browser) for their browser client.
* You know how to check the contents of a GCP bucket using the GCP storage dashboard. If you are unfamiliar with how to do this, see [here](https://cloud.google.com/storage/docs/listing-objects) for instructions. 
* You know how to create and destroy resources using Terraform. If you are unfamiliar with how to do this, see [here](https://www.terraform.io/docs/commands/index.html) for instructions.
* You are familiar with image annotations and how they are used in image segmentation. If you are unfamiliar with this, see [here]() for more information. TODO: add link/link content 
* You are familiar with how datasets are used in Machine Learning (for example, splitting your data into train, validation, and test). If you are unfamiliar with this, see [here]() for more information. TODO: add link/link content  
* You are familiar with how use tmux on a remote machine and how we will use it to keep processes running even if the SSH window is closed or disconnected. If you are unfamiliar with this, see [here](https://tmuxcheatsheet.com/) for more information.
* The codebase is meant to be run on a virtual machine so it installs the python package user-wide. If you wish to run the code locally, we suggest using `virtualenv` (see [here](virtual_environment.md) for instructions).
