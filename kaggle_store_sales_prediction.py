
import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'store-sales-time-series-forecasting:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F29781%2F2887556%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20241004%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20241004T151708Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D3c43f9f0d1f9f71734e52430f913e27996e2bb8bb7cd8d36552838f19883c08682cf93665b108cc69c6512bd5d8aacdef0d57a10c986f8e9665ab447ad905bdb7702fd66d402b1d2886c10823e14a5d55abbd4756650a97c7548b4101985596c634b1b77f5c29b13860dedd05d4b2d6a0806a253389f0de4386b49d204aaa905efc8babb55c3d099e2575c5358d569acbb398d968b052c45bce325f84a1d2c3207406da4a34f7bf201c19b11ea188a88908320306f047dfc81688bf9d40d941d5715f705402368ffb84779b7fcf3e7092edd7e326338e9f17c4e45cb9e4c5aeeb8ffecf0860a3c953416a83739d9f2e06d2fffe611fb1e5fc59236146e54d1ce'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

use_data_since_2016 = False

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')
stores = pd.read_csv('/content/stores.csv')
oil = pd.read_csv('/content/oil.csv')
holidays = pd.read_csv('/content/holidays_events.csv')
transactions = pd.read_csv('/content/transactions.csv')
sample_submission = pd.read_csv('/content/sample_submission.csv')

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
if use_data_since_2016:
    train = train[train['date'] >= '2016-01-01']
    test = test[test['date'] >= '2016-01-01']

train['holiday'] = train['date'].isin(holidays['date'])
test['holiday'] = test['date'] == pd.to_datetime('2017-08-24')

train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['weekday'] = train['date'].dt.weekday

test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['weekday'] = test['date'].dt.weekday

train = train.drop(columns=['date'])
test = test.drop(columns=['date'])

object_cols = train.select_dtypes(include=['object']).columns
train = pd.get_dummies(train, columns=object_cols, drop_first=True)
test = pd.get_dummies(test, columns=object_cols, drop_first=True)

train, test = train.align(test, join='left', axis=1, fill_value=0)

print(train.head())
print(test.head())

#training
X = train.drop(columns=['sales'])
y = train['sales']
y_log = np.log1p(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)
X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
X_val.columns = X_val.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
X_train, test = X_train.align(test, join='left', axis=1, fill_value=0)
X_val, test = X_val.align(test, join='left', axis=1, fill_value=0)
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

august_2017_train = train[(train['year'] == 2017) & (train['month'] == 8)]
august_2017_train.columns = august_2017_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

X_august = august_2017_train.drop(columns=['sales'])
y_august_log = np.log1p(august_2017_train['sales'])

august_pool = Pool(X_august, y_august_log)

#hyperparameters
iterations = 50
lr = 0.1
train_loop_count = 100
retrain_loop_count = 10

catboost_model = CatBoostRegressor(
    iterations=iterations,
    learning_rate=lr,
    depth=8,
    random_seed=42,
    loss_function='RMSE',
    verbose=100
)

catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=50, use_best_model=True, init_model = None)
for i in range(train_loop_count):
    catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=50, use_best_model=True, init_model = catboost_model)

for i in range(retrain_loop_count):
    catboost_model.fit(august_pool, verbose=50, init_model = catboost_model)

test_predictions_log = catboost_model.predict(test)
test['sales'] = np.expm1(test_predictions_log)
test['sales'] = np.where(test['sales'] < 0, 0, test['sales'])

#final submission
submission = test[['id', 'sales']]
submission

submission.to_csv('submission2.csv', index=False)
