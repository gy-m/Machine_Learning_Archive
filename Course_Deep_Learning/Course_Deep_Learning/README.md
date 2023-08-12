# Machine_Learning_Course
## Table of contents
* [Overview](#Overview)
* [Installations](#Installations)
* [Notes](#Notes)
* [TODO](#TODO)
* [Repository status](#Repository-status)

## Overview
Thie repository is based on the Technion course "046211 - Deep Learning" [046211 - Deep Learning Course Reposiroty](https://github.com/taldatech/ee046211-deep-learning).

## Installations
* Miniconda - Installation from the official [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).
* Create the environment and make all the installations according to the given yaml in the repository: `conda env create -f environment.yml`
* Activation of the environment: `conda activate deep_learning`
* Installing PyTorch:
	* PyTorch CPU: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
	* PyTorch GPU (only if you have one): `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
	* TorchText (046211): `conda install -c pytorch torchtext`

## Usage
* Running the Jupyter:
	* Option 1 (Locally) - Run `Jupyter-Lab` (From the jupyter directory: "Notes_JupyterLab" / "Notes_Colab" )
	* Option 2 (Remotly, using Colab and it's provided GPU) - Open Colab website and upload the local jupyter notebooks.  

## Notes
The Installations instructions are based on the instructions which are part of the Technion's course (046211) mentioned above in the following [link](https://github.com/taldatech/ee046211-deep-learning/blob/main/Setting%20Up%20The%20Working%20Environment.pdf)

## TODO
Do the Course

## Repository-status
Freezed - The have not been taken yet