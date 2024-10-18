import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
import numpy as np
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)

    # hyperparameters sent by client
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train-V1.csv')
    parser.add_argument('--test-file', type=str, default='test-V1.csv')

    args, _ = parser.parse_known_args()

    print(f'SKLearn Version: {sklearn.__version__}')
    print(f'Joblib Version: {joblib.__version__}')

    print('[INFO] Reading data\n')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)

    print('Building training and testing datasets\n')
    xtrain = train_df[features]
    ytrain = train_df[label]
    xtest = test_df[features]
    ytest = test_df[label]

    print('Training RandomForest Model...\n')
    model = RandomForestClassifier(n_estimators=args.n_estimators,
                                   random_state=args.random_state,
                                   verbose=1)
    model.fit(xtrain, ytrain)
    print()

    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f'Model persisted at {model_path}')
    print()

    ypred_test = model.predict(xtest)
    test_acc = accuracy_score(ytest, ypred_test)
    test_rep = classification_report(ytest, ypred_test)

    print(f'\n---- METRICS RESULTS FOR TESTING DATA ----\n')
    print(f'Total rows are: {xtest.shape[0]}')
    print(f'[TESTING] Model accuracy is: {test_acc}')
    print(test_rep)
