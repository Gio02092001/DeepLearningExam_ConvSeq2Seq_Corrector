# DeepLearningExam_ConvSeq2Seq_Corrector

About The Project

This project is an implementation of a Convolutional Sequence-to-Sequence (ConvS2S) model, based on the architecture described in "A Convolutional Encoder for Neural Machine Translation" by Gehrig et al. (2017). The model is designed to correct artificially corrupted sentences, simulating common typing errors.

This work was developed as part of the Deep Learning course exam at the University of Florence, under the supervision of Professor Paolo Frasconi.

The core task is to take a "noisy" or corrupted sentence as input and have the model output the original, correct version.
Technology Stack

    PyTorch

    Optuna (for hyperparameter optimization)

    NLTK & Hugging Face Tokenizers

    SacreBLEU & Jiwer (for evaluation metrics)

    PyYAML

Getting Started

Follow these steps to set up and run the project locally.
Prerequisites

    Python 3.10.12

    pip package manager

Installation

    Clone the repository to your local machine.

    Install all the required Python packages using the requirements.txt file:

    pip install -r requirements.txt

Usage

There are two main ways to run this project: standard training or hyperparameter optimization.
1. Standard Training

To train the model with the current settings, run main.py. The model will use the parameters defined in the configuration file.

python3 main.py

    Configuration: To change hyperparameters such as learning rate, batch size, embedding dimensions, or dropout, you can directly edit the Config/config.yaml file.

    Checkpoints: The training script will save checkpoints (best_model.pt and last_model.pt) and a copy of the config file to a new directory in models/{timestamp}/.

2. Hyperparameter Optimization

To start a hyperparameter search using Optuna, run the optimization_hyper.py script. This will automatically try different combinations of hyperparameters to find the best-performing model.

python3 optimization_hyper.py

    Search Space: The range of hyperparameters for the search is defined directly within the optimization_hyper.py script.

    Results: At the end of the optimization run, the best parameters and scores will be printed to the console and saved in a file named Results.json.
