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

To set up and destroy virtual machines, Terraform requires access to GCP. To access these credentials, open the service accounts section through IAM & Admin of your bucket. Create a new service account with your account name, "'username'terraform". Create a key for the new account and add that json file to the "Keys" subfolder of the repository.

Edit the `terraform.tfvars` file with your `username`, `gcp_key_file_location`, `public_ssh_key_location`, `private_ssh_key_location`. For more information on how to generate a public and private SSH key pair, see [here](https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

## Testing

To ensure that the local setup worked, use the Google Cloud SDK to create a Virtual Machine using the following commands
  - `cd` into the necstlab-damage-segmentation repository.
  - `terraform init`
  - `terraform apply -lock=false`
 If you are able to get to a success message after about 10 minutes, your local setup is fully functional.

