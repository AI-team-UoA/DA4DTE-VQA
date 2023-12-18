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
