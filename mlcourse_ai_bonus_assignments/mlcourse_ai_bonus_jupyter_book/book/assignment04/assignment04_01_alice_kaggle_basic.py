import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
np.set_printoptions(linewidth=160)
os.getcwd()
#%%
"""Define all type transformations in a single function"""
def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    sites = [s for s in df.columns if "site" in s]
    df[sites] = df[sites].fillna(0).astype('uint16')
    times = [t for t in df.columns if "time" in t]
    df[times] = df[times].apply(pd.to_datetime)
    if 'target' in df.columns:
        df['target'] = df.target.astype('uint8')
    return df

#%%
def write_to_submission_file(predicted_labels, out_file: str = 'to_submit.csv',
                             target='target', index_label='session id'):
    df = pd.DataFrame(predicted_labels,
                      index = np.arange(1, len(predicted_labels)),
                      columns=[target])
    df.to_csv(out_file, index_label=index_label)

#%%
%%time
df = pd.read_csv('data/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv.zip')
#%%
df.index.name = 'sid'
#%%
df.info()
#%%
df.head().T
#%%
sites = [s for s in df.columns if "site" in s]
sites
#%%
df[sites].min().min()
#%%
df[sites].max().max()
#%%
np.iinfo('uint16')
#%%
df[sites] = df[sites].fillna(0).astype('uint16')
#%%
times = [t for t in df.columns if "time" in t]
times
#%%
df[times] = df[times].apply(pd.to_datetime)
#%%
df['target'] = df.target.astype('uint8')
#%%
df[sites].info()

#%%
""" Convert types in a single line """
df = convert_types(df)
#%%
%%time
train_sessions_str = df[sites].to_string(header=False, index=False).split("\n")
# df[sites][:5].to_string(header=False, index=False).split('\n')
#%%
print(len(train_sessions_str))

print(df[sites].shape)

#%%
%%time
cv = CountVectorizer()
X_train = cv.fit_transform(train_sessions_str)
#%%
y_train = df.target
#%%
%%time
logit = LogisticRegression(C=1, random_state=17)
scores = cross_val_score(logit, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
#%%
print(f"{scores.mean():.3f}, {scores.std():.3f}")
#%%
%%time
logit.fit(X_train, y_train)
#%%

%%time
"""Read test set """
df = pd.read_csv('data/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv.zip')
df = convert_types(df)
df.info()

#%%
%%time
test_sessions_str = df[sites].to_string(header=False, index=False).split("\n")
X_test = cv.transform(test_sessions_str)
#%%

y_test_predict = logit.predict(X_test)
#%%

# logit.coef_.shape



