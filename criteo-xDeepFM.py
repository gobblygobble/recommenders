import sys
import os
import papermill as pm

import tensorflow as tf

from reco_utils.common.constants import SEED
from reco_utils.recommender.deeprec.deeprec_utils import (
    download_deeprec_resources, prepare_hparams
)
from reco_utils.recommender.deeprec.models.xDeepFM import XDeepFMModel
from reco_utils.recommender.deeprec.IO.iterator import FFMTextIterator

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

# criteo run parameters
EPOCHS = 30
BATCH_SIZE = 4096
RANDOM_SEED = SEED

data_path = "data-for-xDeepFM/"
yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
train_file = os.path.join(data_path, r'criteo_tiny_train')
valid_file = os.path.join(data_path, r'criteo_tiny_valid')
test_file = os.path.join(data_path, r'criteo_tiny_test')

if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/deeprec/', data_path, 'xdeepfmresources.zip')

# set hyper-parameters
hparams = prepare_hparams(yaml_file, 
                          FEATURE_COUNT=2300000, 
                          FIELD_COUNT=39, 
                          cross_l2=0.01, 
                          embed_l2=0.01, 
                          layer_l2=0.01,
                          learning_rate=0.002, 
                          batch_size=BATCH_SIZE, 
                          epochs=EPOCHS, 
                          cross_layer_sizes=[20, 10], 
                          init_value=0.1, 
                          layer_sizes=[20,20],
                          use_Linear_part=True, 
                          use_CIN_part=True, 
                          use_DNN_part=True)

model = XDeepFMModel(hparams, FFMTextIterator, seed=RANDOM_SEED)

# train model
model.fit(train_file, valid_file)

# inference after training
#print(model.run_eval(test_file))