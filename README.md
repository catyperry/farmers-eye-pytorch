# farmers-eye-pytorch

This project is a tool to train and test different deep learning models on the lucasVision dataset.

It is a full modern rewrite of [this repository](https://github.com/Momut1/LUCASvision/tree/main) with pytorch.

## Project structure

### `models.py`

This file contains all configurations for every model in use. It implements one class for each model defining its model creation, hyperparameters, optimizers and training parameters.

### `test_data_balanced.py`

This script creates a balanced test data set with 85 images per class from the lucasVision dataset. This number is capped by the class with the fewest images.

### `train_data.py`

Similar to the `test_data_balanced.py` this script creates the training dataset from the lucasVision dataset.

### `preprocess_transform.py`

This script converts the images of the train and test dataset to tensor files to speed up downstream training and testing.

### `train.py`

This is the main script of the project. It can be run seamlessly in google colab or local environments. It has function for loading hyperparameters, evaluating the model (test it), training an epoch, and a main function which stitches everything together. It provides a command line execution with an extensive list of arguments to configure your base model, optimizers, hyperparameters and more.

### `notebook.ipnyb` & `colab_example.ipynb`

These notebook files provide examples on how to run this project locally or in google colab.

## Installation

We use UV https://docs.astral.sh/uv/getting-started/installation/ instead of venv/conda/pip.

Install all dependencies and virtual python environment with:

```bash
uv sync --extra cu128
```

The exta argument installs dependencies for running libraries in a cuda environment.
Then, in Visual Studio Code, open the command palette (Ctrl+Shift+P) and select "Python: Select Interpreter". Choose the interpreter that corresponds to the virtual environment created by UV.

We use python.analysis.typeCheckingMode "standard".

## Preprocessing

The full and clean data can be found here: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/
Images are already sorted via the crop calendar and are ready to use. Additionally, several .csv files are supplied that provide meta data to the images as well as the split into training and test (balanced and imbalanced) data set with. The preprocessing script supplies two programs that read in the information from the .csv files and the images are copied into a separate inputs/training and inputs/test_balanced (balanced testing set with 85 images per category) folder.

## Model creation

The models are registered in the `model.py` file. In the `train.py` file, the model can be selected via the `--model` argument. There are default hyperparamters in the `model.py` file, which can be overwritten via the command line arguments.

## TODOS

- [x] use cases on top of file
- [x] test85 -> balanced
- [x] retrain -> train
- [x] reduce structure
- [x] unify training script
- [x] use tensorboard for all outputs: training loss, testing loss, training accuracy, testing accuracy
- [ ] save hyperparamters with model?
- [-] add other models
  - [x] vit_b_16
  - [ ] foundation model
  - [x] resonant model
  - [ ] convolutional autoencoder
  - [ ] efficientnet
- [x] hyperparameter tuning for vit
- [x] make colab integration easier
- [x] overwrite main branch
- [ ] Add early stopping based on validation loss
- [ ] automatically delete old models during training
- [ ] Support for different optimizers via arguments
- [ ] add augmented data
- [x] update notebook
