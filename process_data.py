#import torchtext
import pandas as pd
from sklearn.model_selection import train_test_split

PATH_TO_TR = "../opus/OpenSubtitles.en-tr.tr"
PATH_TO_EN = "../opus/OpenSubtitles.en-tr.en"

subs_en = open(PATH_TO_EN, encoding='utf-8').read().split('\n')
subs_tr = open(PATH_TO_TR, encoding='utf-8').read().split('\n')

raw_data = {'Turkish' : [line for line in subs_tr], 'English': [line for line in subs_en]}
df = pd.DataFrame(raw_data, columns=["Turkish", "English"])
# remove very long sentences and sentences where translations are 
# not of roughly equal length
df['eng_len'] = df['English'].str.count(' ')
df['tr_len'] = df['Turkish'].str.count(' ')
df = df.query('tr_len < 80 & eng_len < 80')
df = df.query('tr_len < eng_len * 1.5 & tr_len * 1.5 > eng_len')

# create train and validation set 
train, val = train_test_split(df, test_size=0.1)
train.to_csv("../opus/train.csv", index=False)
val.to_csv("../opus/val.csv", index=False)