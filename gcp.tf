variable "gcp_key_file_location" {}
variable "username" {}
variable "project_name" {}
variable "number_of_cpus" {}
variable "ram_size_mb" {}
variable "hard_drive_size_gp" {}
variable "number_of_machines" {}
variable "private_ssh_key_location" {}
variable "public_ssh_key_location" {}
variable "gpu_type" {}
variable "number_of_gpus" {}
variable "repository_name" {}


provider "google" {
  credentials = "${var.gcp_key_file_location}"
  project     = "${var.project_name}"
  region      = "us-east1"
}

resource "google_compute_instance" "vm" {
  count        = "${var.number_of_machines}"
  name         = "${element(split("_", var.username), 0)}-${var.project_name}-${count.index}"
  machine_type = "custom-${var.number_of_cpus}-${var.ram_size_mb}"
  zone         = "us-east1-c"
  allow_stopping_for_update = true
  tags = ["${var.project_name}"]

 boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/ubuntu-1604-xenial-v20190430"
      size = "${var.hard_drive_size_gp}"
      type = "pd-ssd"
    }
  }

  metadata = {
    install-nvidia-driver = "True"
    ssh-keys = "${var.username}:${file(var.public_ssh_key_location)}"
  }

  service_account {
    scopes = ["cloud-platform"]
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral IP - leaving this block empty will generate a new external IP and assign it to the machine
    }
  }

  guest_accelerator{
    type = "${var.gpu_type}" // Type of GPU attahced
    count = "${var.number_of_gpus}" // Num of GPU attached
  }

  scheduling{
    on_host_maintenance = "TERMINATE" // Need to terminate GPU on maintenance
  }

  provisioner "file" {
    source      = "../${var.repository_name}"
    destination = "~/${var.repository_name}"
    connection {
      user = "${var.username}"
      type = "ssh"
      private_key = "${file(var.private_ssh_key_location)}"
      host = "${self.network_interface[0].access_config[0].nat_ip}"
    }
  }

  provisioner "remote-exec" {
    script = "./scripts/resource-creation.sh"
    connection {
      user = "${var.username}"
      type = "ssh"
      private_key = "${file(var.private_ssh_key_location)}"
      host = "${self.network_interface[0].access_config[0].nat_ip}"
    }
  }

  # final reboot because of the cuda install, will tell terraform it has "failed" since it lost connection
  provisioner "remote-exec" {
    inline = [
      "sudo reboot"
    ]
    on_failure = "continue"  # ignore the incorrect failure
    connection {
      user = "${var.username}"
      type = "ssh"
      private_key = "${file(var.private_ssh_key_location)}"
      host = "${self.network_interface[0].access_config[0].nat_ip}"
    }
  }

}
