from explainit.app import build

import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

TRAIN_CSV_PATH = os.getenv("TRAIN_CSV_PATH")
TEST_CSV_PATH = os.getenv("TEST_CSV_PATH")
TARGET_COLUMN_NAME = os.getenv("TARGET_COLUMN_NAME")
TARGET_COLUMN_TYPE = os.getenv("TARGET_COLUMN_TYPE")

ref_data = pd.read_csv(TRAIN_CSV_PATH)
cur_data = pd.read_csv(TEST_CSV_PATH)

ref_data = pd.read_csv()

build(
  reference_data=ref_data,
  current_data=cur_data,
  target_column_name=TARGET_COLUMN_NAME,
  target_column_type=TARGET_COLUMN_TYPE
)