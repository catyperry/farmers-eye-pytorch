# farmers-eye-pytorch

This project aims to do a full modern rewrite of [this repository](https://github.com/Momut1/LUCASvision/tree/main) with pytorch. 


1 - Preprocessing
The full and clean data can be found here: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/LUCASvision/
Images are already sorted via the crop calendar and are ready to use. Additionally, several .csv files are supplied that provide meta data to the images as well as the split into training and test (balanced and imbalanced) data set with. The preprocessing supplies two programs that read in the information from the .csv files and the images are copied into a separate inputs/training and inputs/test85 (balanced testing set with 85 images per category) folder.

2 - Run model
retrain.py: Uses MobileNet v2 via pytorch to train a CNN. Testing on test data can be switched on if wanted. 

Use case: 

python retrain.py \
  --data_dir_train /content/training_data \
  --batch_size 1 \
  --num_epoch 1 \
  --test85 True \
  --data_dir_test85 /content/drive/MyDrive/farmers_eye/inputs/test85

