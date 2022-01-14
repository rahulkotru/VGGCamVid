import zipfile
import tensorflow as tf

print("Tensorflow version " + tf.__version__)

# download the dataset (zipped file) using bash
#gdown --id 0B0d9ZiqAgFkiOHR1NTJhWVJMNEU -O D:/28_GitHub/VGG-16/fcnn-dataset.zip 

# extract the downloaded dataset to a local directory: /tmp/fcnn
local_zip = 'fcnn-dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/fcnn')
zip_ref.close()