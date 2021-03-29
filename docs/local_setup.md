# Setting up your local machine on a Windows Machine

## Terraform

To programmatically set up and destroy cloud resources (virtual machines, buckets, etc.), we will use a tool called Terraform. Follow the link [here](https://www.terraform.io/downloads.html) for the terraform download. Extract the executable and add it to your PATH.

## Github Desktop

This software easily allows integration with GitHub version control and seamless cloning of the repository. Follow the download link [here](https://desktop.github.com/). During installation, ensure Git and Git Bash are installed and sign into your GitHub account once the installation is finished.

## Git/GitBash

Git Bash is a terminal that we use to rectify Unix line endings within windows machines. Git should have been installed with GitHub Desktop. If not, follow the steps below.

Follow the link here to download Git. Once GitBash is installed, use the following command within that console: `git config --global core.autocrlf false`
If you used GitHub Desktop, open Github Desktop and open the text editor by following Repository -> Command Prompt and enter the following lines: `git config core.eol lf', 'git config core.autocrlf "input"`.

## Python
Python is the language used for scripts within the project. Install the latest Python SDK by following this [link](https://www.python.org/downloads/). During this download, ensure that the option to add Python to the PATH is enabled.

If you do not have an IDE installed, it is Pycharm is a recommended option. Follow this [link](https://www.jetbrains.com/help/pycharm/installation-guide.html#standalone) to install Pycharm.

## Clone this repository locally

To copy this repository locally, in a terminal window, enter and clone the repository using the command: `git clone git@github.com:mit-quest/necstlab-damage-segmentation.git`. 
If you do not have `git` installed, return to the previous header and follow the installation instructions.

Alternatively you can use GitHub desktop to clone the repository. You can accomplish this through clicking add -> clone repository and using the following URL: https://github.com/mit-quest/necstlab-damage-segmentation

All commands will assume to be run from the `necstlab-damage-segmentation` directory, which you can `cd` into using: `cd ~/necstlab-damage-segmentation`

## GCP

All of the workflows use Google Cloud Platform (GCP) for storage (buckets) and compute (virtual machines). To allow the code to programmatically interact with GCP, we will set up a Software Development Kit (SDK) on your local machine. To install the GCP SDK follow the instructions [here](https://cloud.google.com/sdk/docs/downloads-interactive).

To set up and destroy virtual machines, Terraform requires access to GCP. To access these credentials, open the service accounts section through IAM & Admin of your bucket. Create a new service account with your account name, "'username'-terraform". Create a key for the new account and add that json file to the "Keys" subfolder of the repository.

Edit the `terraform.tfvars` file with your `username`, `gcp_key_file_location`, `public_ssh_key_location`, `private_ssh_key_location`. For more information on how to generate a public and private SSH key pair, see [here](https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

## Testing

To ensure that the local setup worked, use the Google Cloud SDK to create a Virtual Machine using the following commands
  - `cd` into the necstlab-damage-segmentation repository.
  - `terraform init`
  - `terraform apply -lock=false`
 If you are able to get to a success message after about 10 minutes, your local setup is fully functional.


# Setting up your local machine on a MacOS Machine

## Terraform

To programmatically set up and destroy cloud resources (virtual machines, buckets, etc.), we will use a tool called Terraform. Follow the link [here](https://www.terraform.io/downloads.html) for the Terraform download. Extract the executable. You’ll need to add this to your local bin using the instructions found [here](https://learn.hashicorp.com/tutorials/terraform/install-cli): follow the directions exactly to add the Terraform executable to your `/usr/local/bin`. Terraform should now be part of your PATH. You can check by running `terraform init`, then `terraform`.

## GitHub Desktop

This software easily allows integration with GitHub version control and seamless cloning of the repository. Follow the download link [here](https://desktop.github.com/). During installation, ensure Git is installed and sign into your GitHub account (not a GitHub Enterprise account) once the installation is finished.

## Git/GitBash

Because macOSX has a Unix-based core, you should not need to utilize Git Bash. This is assuming that you utilize your computer’s built-in terminal instead.

## Python

Python is the language used for scripts within the project. Install the latest Python SDK by following this [link](https://www.python.org/downloads/). During this download, ensure that the option to add Python to the PATH is enabled.

If you do not have an IDE installed, Pycharm is a recommended option. Follow this [link](https://www.jetbrains.com/help/pycharm/installation-guide.html#standalone) to install Pycharm.

## Clone this repository locally

To copy this repository locally, in a terminal window, enter and clone the repository using the command: `git clone git@github.com:mit-quest/necstlab-damage-segmentation.git`. 

Alternatively you can use GitHub desktop to clone the repository. You can accomplish this through clicking add -> clone repository and using the following URL: https://github.com/mit-quest/necstlab-damage-segmentation

All commands are assumed to run from the `necstlab-damage-segmentation` directory, which you can `cd` into using: `cd ~/necstlab-damage-segmentation`.

## GCP

All of the workflows use Google Cloud Platform (GCP) for storage (buckets) and compute (virtual machines). To allow the code to programmatically interact with GCP, we will set up a Software Development Kit (SDK) on your local machine. To install the GCP SDK follow the instructions [here](https://cloud.google.com/sdk/docs/downloads-interactive).

To set up and destroy virtual machines, Terraform requires access to the GCP. To access these credentials, open the "Service Accounts" section of the GCP through the “IAM & Admin” category of your bucket using the left-hand dropdown window. Create a new service account with your account name, "'username'-terraform". Create a key for your new service account and add the created .json file to the "Keys" subfolder of the repository using GitHub.

Additionally, you will need to generate a public and private SSH key pair. For more information on how to generate these, see [here](https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

Locate the `terraform.tfvars` file in the necstlab-damage-segmentation directory on GitHub. Edit the .tfvars file with your account username, gcp_key_file_location (replace the name of the .json file at the end of the gcp_key_file_location), public_ssh_key_location, and private_ssh_key_location. For the public and private ssh keys, the locations correspond to your computer’s directory (i.e. if located in your home directory, you would write “~/.ssh/'ssh key id'”). 

## Testing

To ensure that the local setup worked, use the Google Cloud SDK to create a Virtual Machine using the following commands
  - `cd` into the necstlab-damage-segmentation repository.
  - `terraform init`
  - `terraform apply -lock=false`
 If you are able to get to a success message after about 10 minutes, your local setup is fully functional.
