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

# synthetic run parameters
EPOCHS = 15
BATCH_SIZE = 128
RANDOM_SEED = SEED

data_path = "data-for-xDeepFM/"
yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
train_file = os.path.join(data_path, r'synthetic_part_0')
valid_file = os.path.join(data_path, r'synthetic_part_1')
test_file = os.path.join(data_path, r'synthetic_part_2')
output_file = os.path.join(data_path, r'output.txt')

if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/deeprec/', data_path, 'xdeepfmresources.zip')

print("Data gathering complete")

# 1. prepare hyper-parameters
hparams = prepare_hparams(yaml_file, 
                          FEATURE_COUNT=1000, 
                          FIELD_COUNT=10, 
                          cross_l2=0.0001, 
                          embed_l2=0.0001, 
                          learning_rate=0.001, 
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE)
print("Hyper-parameters: ")
print(hparams)
# 2. create data loader
# designate a data iterator for xDeepFM model (FFMTextIterator)
input_creator = FFMTextIterator

# 3. create model
model = XDeepFMModel(hparams, input_creator, seed=RANDOM_SEED)
# we can also load a pre-trained model with model.load_model(r'model_path')

# untrained model's performance
print("Untrained model's performance: {}".format(model.run_eval(test_file)))

# 4. train model
print("Begin model training...")
model.fit(train_file, valid_file)
print("End model training...")

# 5. evaluate model
res_syn = model.run_eval(test_file)
print("Trained model's performance: {}".format(res_syn))
pm.record("rest_syn", res_syn)
# we can also get full prediction scores rather than evaluation metrics with model.predict(test_file, output_file)

