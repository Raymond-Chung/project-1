"""
work in virtual envrionment (uv - https://docs.astral.sh/uv/pip/environments/)

git steps:
git status
git add .
git commit -m "some messge"
git push

add branch git checkout -b image-recognition
git checkout main
"""

import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential 
