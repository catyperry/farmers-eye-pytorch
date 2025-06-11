# farmers-eye-pytorch

This project aims to do a full modern rewrite of [this repository](https://github.com/Momut1/LUCASvision/tree/main) with pytorch.

## Installation

We use UV https://docs.astral.sh/uv/getting-started/installation/ instead of venv/conda/pip.

Install all dependencies and virtual python environment with:

```bash
uv sync
```

Then, in Visual Studio Code, open the command palette (Ctrl+Shift+P) and select "Python: Select Interpreter". Choose the interpreter that corresponds to the virtual environment created by UV.

We use python.analysis.typeCheckingMode "standard".

## Preprocessing

The full and clean data can be found here: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/
Images are already sorted via the crop calendar and are ready to use. Additionally, several .csv files are supplied that provide meta data to the images as well as the split into training and test (balanced and imbalanced) data set with. The preprocessing supplies two programs that read in the information from the .csv files and the images are copied into a separate inputs/training and inputs/test_balanced (balanced testing set with 85 images per category) folder.

## Model creation

The models are registered in the `model.py` file. In the `train.py` file, the model can be selected via the `--model` argument. There are default hyperparamters in the `model.py` file, but they can be overwritten via the command line arguments.

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
  - [ ] resonant model
  - [ ] convolutional autoencoder
  - [ ] efficientnet
- [ ] make colab integration easier
- [ ] overwrite main branch
- [ ] Add early stopping based on validation loss
- [ ] Support for different optimizers via arguments
- [ ] add augmented data
- [ ] update notebook
