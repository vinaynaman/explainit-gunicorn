import os
os.system("pip install explainit-1.0-py3-none-any.whl")

from explainit.app import build
import pandas as pd

TRAIN_CSV_PATH = os.getenv("TRAIN_CSV_PATH")
TEST_CSV_PATH = os.getenv("TEST_CSV_PATH")
TARGET_COLUMN_NAME = os.getenv("TARGET_COLUMN_NAME")
TARGET_COLUMN_TYPE = os.getenv("TARGET_COLUMN_TYPE")

ref_data = pd.read_csv(TRAIN_CSV_PATH)
cur_data = pd.read_csv(TEST_CSV_PATH)

if "Unnamed: 0" in list(ref_data.columns):
    ref_data = ref_data.drop("Unnamed: 0", axis =1)

if "Unnamed: 0" in list(cur_data.columns):
    cur_data = cur_data.drop("Unnamed: 0", axis =1)

build(
  reference_data=ref_data,
  current_data=cur_data,
  target_column_name=TARGET_COLUMN_NAME,
  target_column_type=TARGET_COLUMN_TYPE
)