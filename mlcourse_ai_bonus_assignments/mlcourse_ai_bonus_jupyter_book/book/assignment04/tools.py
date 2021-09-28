import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Union, List



def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """ Define all type transformations in a single function """
    sites = [s for s in df.columns if "site" in s]
    df[sites] = df[sites].fillna(0).astype('uint16')
    times = [t for t in df.columns if "time" in t]
    df[times] = df[times].apply(pd.to_datetime)
    if 'target' in df.columns:
        df['target'] = df.target.astype('uint8')
    return df


def get_auc_scores(X, y, estimator, n_splits=10):
    time_split = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_validate(estimator, X, y, cv=time_split, scoring='roc_auc',
                               return_train_score=True, n_jobs=-1)
    return scores


def write_to_submission_file(predicted_probs, out_file: str = 'to_submission.csv.zip',
                             target='target', index_label='session_id'):
    df = pd.DataFrame(predicted_probs,
                      index=np.arange(1, len(predicted_probs) + 1),
                      columns=[target])
    df.to_csv(out_file, index_label=index_label, compression="zip")


def _plot_search_results(param_grid: list, param_name: str, scores: list):
    df = pd.DataFrame(scores, index=param_grid)
    df.index.name = param_name
    df = df.explode(list(df.columns)).reset_index()
    sns.lineplot(x=param_name, y='train_score', data=df, label='train')
    sns.lineplot(x=param_name, y='test_score', data=df, label='test')
    plt.ylabel('roc_auc_score')
    plt.show()


class AliceKaggle:
    
    def __init__(self):
        self.load_data()
        self.vectorize_text()
        self.generate_features()
        self.baseline = 0.0

    def load_data(self):
        data_folder = '../../../../data/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/'

        train_df = pd.read_csv(data_folder + 'train_sessions.csv.zip')
        train_df = convert_types(train_df)
        train_df.sort_values(by='time1', inplace=True)

        test_df = pd.read_csv(data_folder + 'test_sessions.csv.zip')
        test_df = convert_types(test_df)

        # Our target variable
        self.y_train = train_df["target"]

        # United dataframe of the initial data
        self.full_df = pd.concat([train_df.drop("target", axis=1), test_df])

        # Index to split the training and test data sets
        self.idx_split = train_df.shape[0]

        self._set_text_corprus()

    def _set_text_corprus(self):
        sites = [s for s in self.full_df.columns if 'site' in s]
        self.sites_corpus = self.full_df[sites].to_string(header=False, index=False).split('\n')

    def vectorize_text(self, vectorizer=TfidfVectorizer(ngram_range=(1, 2), max_features=20000)):
        self.X_train = vectorizer.fit_transform(self.sites_corpus[:self.idx_split])
        self.X_test = vectorizer.transform(self.sites_corpus[self.idx_split:])
        self.X_full = vstack([self.X_train, self.X_test]).tocsr()

    def generate_features(self):
        full_df = self.full_df
        times = [t for t in full_df.columns if 'time' in t]
        feat_df = pd.DataFrame(index=full_df.index)

        # week day features
        feat_df['weekday'] = full_df['time1'].dt.weekday
        for weekday in range(7):
            feat_name = f'weekday_{weekday}'
            feat_df[feat_name] = (full_df['time1'].dt.weekday == weekday).astype(int)

        feat_df['duration'] = (full_df[times].max(axis=1) - full_df[times].min(axis=1)).dt.total_seconds()

        # hour base features
        for hour in range(23):
            feat_name = f'hour_{hour}'
            feat_df[feat_name] = (full_df['time1'].dt.hour == hour).astype(int)

        feat_df['hour'] = full_df['time1'].dt.hour
        feat_df['hour_sin'] = np.sin(full_df['time1'].dt.hour / 24)
        feat_df['hour_cos'] = np.cos(full_df['time1'].dt.hour / 24)

        feat_df['morning'] = feat_df['hour'].between(7, 11).astype(int)
        feat_df['noon'] = feat_df['hour'].between(12, 18).astype(int)
        feat_df['evening'] = feat_df['hour'].between(19, 23).astype(int)
        feat_df['night'] = feat_df['hour'].between(0, 6).astype(int)

        # time between sites features
        deltas = ['delta' + str(i) for i in range(1, 10)]
        delta_df = (full_df[times] - full_df[times].shift(1, axis=1)) \
            .copy() \
            .drop(columns='time1') \
            .apply(lambda x: x.dt.total_seconds())
        delta_df.columns = deltas
        for delta in deltas:
            feat_df[delta] = delta_df[delta]
        feat_df['delta_avg'] = delta_df.mean(axis=1, skipna=True).fillna(0.0)

        self.feat_df = feat_df.fillna(0.0)

    # Add the new feature to the sparse matrix
    def add_feature(self, feat: str, X_sparse=None, standardize=True, onehot=False):
        tmp = self.feat_df[[feat]].values
        if onehot:
            enc = OneHotEncoder(dtype=np.uint8, sparse=False)
            tmp = enc.fit_transform(tmp)
        if standardize:
            tmp = StandardScaler().fit_transform(tmp)
        if X_sparse is not None:
            return hstack([X_sparse, tmp]).tocsr()
        else:
            return csr_matrix(tmp)

    def add_multi_feature(self, feat_list: list, X_sparse=None):
        X_new = X_sparse
        for feat in feat_list:
            X_new = self.add_feature(feat, X_new)
        return X_new[:self.idx_split, :], X_new[self.idx_split:, :]

    def test_features(self, features: Union[list, str], estimator, standardize=True, onehot=False):
        print(f"Testing:\t{features}")

        if isinstance(features, str):
            features = [features]

        X_new = self.X_full
        for feat in features:
            X_new = self.add_feature(feat, X_new, onehot=onehot, standardize=standardize)
        X_train_new = X_new[:self.idx_split, :]
        scores = get_auc_scores(X_train_new, self.y_train, estimator)

        print(f"Score:\t\t{scores['train_score'].mean():.4f}\t", end="")

        if scores['train_score'].mean() > BASELINE:
            print(f"+++ baseline:\t{BASELINE:.4f}")
        else:
            print(f"--- baseline:\t{BASELINE:.4f}")
        return scores

    def predict_probs(self, X, estimator):
        return estimator.predict_proba(X)[:, 1]

    def param_search(self, X, param_grid: list, param_name: str, estimator, plot=True):
        scores = []
        best_param = None
        best_test_score = 0.0
        for i, param in enumerate(param_grid):
            print(f"{i + 1} / {len(param_grid)}")
            estimator.set_params(**{param_name: param})
            scores.append(get_auc_scores(X, self.y_train, estimator))
            test_score = scores[-1]['test_score'].mean()
            if test_score > best_test_score:
                best_test_score = test_score
                best_param = param
        print("Best results:")
        print(f"{param_name}: {best_param}")
        print(f"{best_test_score:.4f}")

        if plot:
            _plot_search_results(param_grid, param_name, scores)

        return scores



