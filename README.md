# DA4DTE-VQA

Visual Question Answering engine for DA4DTE. Temporarily uses RSVQAxBEN for training. 

## Setup

To setup with Conda:

`$ conda create --name <env> --file requirements.txt`

## Training

Run `lit4rsvqa.py` with the appropriate options. For more information: 

`$ lit4rsvqa.py --help`

## Web Server

Run `vqa_server.py` with the appropriate options (port and model checkpoint are required). For more information: 

`$ vqa_server.py --help`

## Docker setup

### Enable nvidia-container-toolkit

Configure the repository:

      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee                   /etc/apt/sources.list.d/nvidia-container-toolkit.list \
      && sudo apt-get update

Install the NVIDIA Container Toolkit packages:

      sudo apt-get install -y nvidia-container-toolkit

Configure the container runtime by using the nvidia-ctk command:

      sudo nvidia-ctk runtime configure --runtime=docker

Restart the Docker daemon:

      sudo systemctl restart docker

### Build and run through Dockerfile

Enter the `docker` directory:

      cd docker/

To build the docker image run:

      sudo docker build -t vqa .

To run the docker container image run:

      sudo docker run --gpus all --name vqa-container -p 5000:8080 vqa

After a short delay vqa_server will be online on http://localhost:5000
