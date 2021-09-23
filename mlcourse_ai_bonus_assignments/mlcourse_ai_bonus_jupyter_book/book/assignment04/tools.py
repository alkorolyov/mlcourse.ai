import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Union, List

full_df, y_train, idx_split = pd.DataFrame(), pd.DataFrame(), 0
X_full, X_train, X_test = None, None, None
feat_df = pd.DataFrame()
BASELINE = 0.0


def write_to_submission_file(predicted_probs, out_file: str = 'to_submission.csv.zip',
                             target='target', index_label='session_id'):
    df = pd.DataFrame(predicted_probs,
                      index=np.arange(1, len(predicted_probs) + 1),
                      columns=[target])
    df.to_csv(out_file, index_label=index_label, compression="zip")


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """ Define all type transformations in a single function """
    sites = [s for s in df.columns if "site" in s]
    df[sites] = df[sites].fillna(0).astype('uint16')
    times = [t for t in df.columns if "time" in t]
    df[times] = df[times].apply(pd.to_datetime)
    if 'target' in df.columns:
        df['target'] = df.target.astype('uint8')
    return df


def _read_data():
    train_df = pd.read_csv('../../../../data/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv.zip')
    train_df = convert_types(train_df)
    train_df.sort_values(by='time1', inplace=True)

    test_df = pd.read_csv('../../../../data/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv.zip')
    test_df = convert_types(test_df)

    # Our target variable
    global y_train
    y_train = train_df["target"]

    # United dataframe of the initial data
    global full_df
    full_df = pd.concat([train_df.drop("target", axis=1), test_df])

    # Index to split the training and test data sets
    global idx_split
    idx_split = train_df.shape[0]


def _vectorize_data():
    sites = [s for s in full_df.columns if 'site' in s]
    # times = [t for t in full_df.columns if 'time' in t]

    sites_corpus = full_df[sites].to_string(header=False, index=False).split('\n')

    tfv = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    global X_train
    X_train = tfv.fit_transform(sites_corpus[:idx_split])
    global X_test
    X_test = tfv.transform(sites_corpus[idx_split:])

    global X_full
    X_full = vstack([X_train, X_test]).tocsr()


def get_auc_logit_score(X, y, C=1.0, seed=17, n_splits=10):
    # Split the data into the training and validation sets
    time_split = TimeSeriesSplit(n_splits=n_splits)
    logit = LogisticRegression(C=C, random_state=seed, solver='liblinear')
    cv_scores = cross_val_score(logit, X, y, cv=time_split, scoring='roc_auc', n_jobs=-1)
    return cv_scores.mean()


def _set_baseline():
    global BASELINE
    BASELINE = get_auc_logit_score(X_train, y_train)


def _set_features():
    times = [t for t in full_df.columns if 'time' in t]
    for weekday in range(7):
        feat_name = f'weekday_{weekday}'
        feat_df[feat_name] = (full_df['time1'].dt.weekday == weekday).astype(int)
    for hour in range(23):
        feat_name = f'hour_{hour}'
        feat_df[feat_name] = (full_df['time1'].dt.hour == hour).astype(int)

    feat_df['duration'] = (full_df[times].max(axis=1) - full_df[times].min(axis=1)).dt.total_seconds()

    hour = full_df['time1'].dt.hour
    feat_df['morning'] = hour.between(7, 11).astype(int)
    feat_df['noon'] = hour.between(12, 18).astype(int)
    feat_df['evening'] = hour.between(19, 23).astype(int)
    feat_df['night'] = hour.between(0, 6).astype(int)

    best_hours = ['hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_14', 'hour_16', 'hour_17', 'hour_18',
                  'hour_19', 'hour_20', 'hour_21', 'hour_22']
    best_weekdays = ['weekday_0']
    best_previous = ['morning', 'noon', 'evening']


def initialize():
    _read_data()
    _vectorize_data()
    _set_baseline()
    _set_features()


# Add the new feature to the sparse matrix
def add_feature(feat: str, f_df: pd.DataFrame = feat_df, X_sparse=X_full, standardize=True, onehot=False):
    tmp = f_df[[feat]].values
    if onehot:
        enc = OneHotEncoder(dtype=np.uint8, sparse=False)
        tmp = enc.fit_transform(tmp)
    if standardize:
        tmp = StandardScaler().fit_transform(tmp)
    return hstack([X_sparse, tmp]).tocsr()


def add_multi_feature(feat_list: list, f_df: pd.DataFrame = feat_df, X_sparse=X_full):
    X_new = X_sparse
    for feat in feat_list:
        X_new = add_feature(feat, f_df=f_df, X_sparse=X_new)
    return X_new[:idx_split, :], X_new[idx_split:, :]


def test_feature(features: Union[list, str], standardize=True, onehot=False, baseline=BASELINE, C=1):
    print(f"Testing:\t{features}")

    if isinstance(features, str):
        features = [features]
    
    X_new = X_full
    for feat in features:
        X_new = add_feature(feat, X_sparse=X_new, onehot=onehot)
    X_train_new = X_new[:idx_split, :]
    score = get_auc_logit_score(X_train_new, y_train, C=C)

    print(f"Score:\t\t{score:.4f}\t", end="")
    if score > baseline:
        print(f"+++ baseline:\t{baseline:.4f}")
    else:
        print(f"--- baseline:\t{baseline:.4f}")
    return score


def test_multi_feature(feat_list: list, baseline=BASELINE, C=1):
    """
    Deprecated
    """
    print(f"Testing:\t{feat_list}")

    X_new = X_full
    for feat in feat_list:
        X_new = add_feature(feat, X_sparse=X_new)
    X_train_new = X_new[:idx_split, :]
    score = get_auc_logit_score(X_train_new, y_train, C=C)

    print(f"Score:\t\t{score:.4f}\t", end="")
    if score > baseline:
        print(f"+++ baseline: {baseline:.4f}")
    else:
        print(f"--- baseline: {baseline:.4f}")
    return score


def predict_probs(feat_list: list, C=1):
    X_new = X_full
    for feat in feat_list:
        X_new = add_feature(feat, X_sparse=X_new)
    X_train_new = X_new[:idx_split, :]
    X_test_new = X_new[idx_split:, :]
    estimator = LogisticRegression(C=C, random_state=17, solver='liblinear')
    estimator.fit(X_train_new, y_train)
    return estimator.predict_proba(X_test_new)[:, 1]


