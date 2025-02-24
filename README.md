# Introduction to Deep Learning Homework 4

This repository contains the code and documentation for Homework 4 of CS499 Introduction to Deep Learning.

## Contents

- `chkpts/`: Directory containing the two best weights created after training.
- `datasets/`: Directory containing data files used in the project, and the CIFAR10 dataloader.
- `models/`: Directory containing model architecture files.

## Requirements

- Python 3.8-3.11. I used python 3.9.
- All other dependencies are listed in requirements.txt.
- I used windows 11, however any OS should work.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/CS499-hw4.git
    ```
2. Navigate to the project directory:
    ```
    cd CS499-hw4
    ```
3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Set up Wandb. This can be done by following the instructions here: https://docs.wandb.ai/quickstart/.

2. run ```py train.py``` to start the training. Checkpoint files will automatically be created and placed in ./chkpts.


## Output
Assuming everything runs correctly, in Wandb you will be able to see a number of output graphs, the Acc/Val graph should look close to this. 
<img width="343" alt="Untitled" src="https://github.com/user-attachments/assets/c82070d6-a140-4491-b22d-d0458d2c0ce6" />

