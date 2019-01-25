#import torchtext
import pandas as pd
from sklearn.model_selection import train_test_split

PATH_TO_TR = "../opus/OpenSubtitles.en-tr.tr"
PATH_TO_EN = "../opus/OpenSubtitles.en-tr.en"

subs_en = open(PATH_TO_EN, encoding='utf-8').read().split('\n')
subs_tr = open(PATH_TO_TR, encoding='utf-8').read().split('\n')

raw_data = {'Turkish' : [line for line in subs_tr], 'English': [line for line in subs_en]}
df = pd.DataFrame(raw_data, columns=["Turkish", "English"])

# Remove dashes in the beginning when they occur.
remove_dashes = lambda s: s[2:] if s.startswith('- ') else s
df = df.apply(remove_dashes, axis=0)

# Remove very long sentences
df['eng_len'] = df['English'].str.count(' ')
df['tr_len'] = df['Turkish'].str.count(' ')
df = df.query('tr_len < 80 & eng_len < 80')

# Drop eng_len, tr_len columns
df = df[['English', 'Turkish']]

# create train and validation set 
train, val = train_test_split(df, test_size=0.1)
train.to_csv("../opus/train.csv", index=False, sep='\t')
val.to_csv("../opus/val.csv", index=False, sep='\t')