#!/bin/bash

#make sure kaggle api has been setup
[ -d data ] || mkdir data
cd data

{
 kaggle competitions download -c histopathologic-cancer-detection 
} ||
{
echo "Kaggle API could not download the dataset. 
Make sure Kaggle API has been setup properly and/or dataset is available on Kaggle. Refer to README for instructions."
}

unzip histopathologic-cancer-detection.zip
echo "Data downloaded succesfully"
rm histopathologic-cancer-detection.zip
du -hs * | sort -hr
echo "Open the CancerNet notebooks to start training the model"