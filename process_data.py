import random
import pdb
#import torchtext
import pandas as pd
from sklearn.model_selection import train_test_split

PATH_TO_TR = "../opus/OpenSubtitles.en-tr.tr"
PATH_TO_EN = "../opus/OpenSubtitles.en-tr.en"

# PATH_TO_TR = "../opus/opensubs.tr.mini"
# PATH_TO_EN = "../opus/opensubs.en.mini"

subs_en = open(PATH_TO_EN, encoding='utf-8').read().split('\n')[:-1]
subs_tr = open(PATH_TO_TR, encoding='utf-8').read().split('\n')[:-1]

raw_data = {'tr' : [line for line in subs_tr], 'en': [line for line in subs_en]}
df = pd.DataFrame(raw_data, columns=['tr', 'en'])

# Remove dashes in the beginning when they occur.
remove_dashes = lambda s: s[2:] if s.startswith('- ') else s
df = df.applymap(remove_dashes)

# Remove very long sentences
df['eng_len'] = df['en'].str.count(' ')
df['tr_len'] = df['tr'].str.count(' ')
df = df.query('tr_len < d80 & eng_len < 80')

# Drop eng_len, tr_len columns
df = df[['tr','en']]

# create train and validation set
train_size = 2E6
val_size = 10000
test_size = 10000
print(f"training size: {train_size}, validation size: {val_size}")

# Select correct number of indices
val_indices = random.sample(range(1, df.shape[0]), val_size)
train_indices = list(set(range(1, df.shape[0])) - set(val_indices))
test_indices = list(set(range(1, df.shape[0])) - set(val_indices) - set(train_indices))

def select_indices(indices, df, file_path):
    val_df = df.iloc[indices]
    val_context_df = df.iloc[[i-1 for i in indices]]
    val_context_df.columns = ["tr_context", "en_context"]
    val_df.reset_index(drop=True, inplace=True)
    val_context_df.reset_index(drop=True, inplace=True)
    val_df = pd.concat([val_df, val_context_df], axis=1)
    val_df = val_df.reindex(columns = ["tr_context", "tr", "en_context", "en"])
    val_df.to_csv(file_path, index=False, sep='\t')
    print(f"Wrote to {file_path}")
    return

select_indices(val_indices, df, "../opus/val_10k.csv")
select_indices(test_indices, df, "../opus/test_10k.csv")
select_indices(train_indices, df, "../opus/train_2m.csv")

