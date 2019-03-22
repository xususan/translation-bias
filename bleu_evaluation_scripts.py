import pandas as pd


def write_to_ref_file(file_path):
	input_path = "data/bleu/" + file_path
	df = pd.read_csv(input_path, sep='\t', header=None)
	# df_en = df[['en']]
	df_en = df.iloc[:, 3]
	out_path = "data/bleu/" + file_path[:-4] + "_ref.csv"
	df_en.to_csv(out_path, index=False, sep='\t', header=False)


def recover_train_200k_because_im_stupid():
	df = pd.read_csv("data/train_200k_annotated.csv", sep='\t')
	df = df[["tr_context", "tr", "en_context", "en"]]
	df.to_csv("data/train_200k.csv", index=False, sep='\t')

# recover_train_200k_because_im_stupid()

write_to_ref_file("val_10k_context_only.csv")