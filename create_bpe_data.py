import pdb
import csv
import sys
import argparse
from bpemb import BPEmb

"""
Process the data into bpe
"""

parser = argparse.ArgumentParser(description='Subsample Datasets (with special properties)')
parser.add_argument('--inpath', type=str, default="full", help='Path to infile (within data/)')
parser.add_argument('--outpath', type=str, default="full", help='Path to outfile (within data/)')
args = parser.parse_args()

def load_bpe(vocab_size):
    """ Load pre-trained byte pair embedding models.

    Return src, trg
    """
    bpemb_tr = BPEmb(lang="tr", vs=vocab_size)
    bpemb_en = BPEmb(lang="en", vs=vocab_size)
    return bpemb_tr, bpemb_en


if __name__ == "__main__":
	outfile = open("data/" + args.outpath, 'w+', newline='')
	csv_writer = csv.writer(outfile, delimiter='\t')
	nrows = 0
	print("loading bpe")
	bpemb_tr, bpemb_en = load_bpe(25000)
	print("loaded bpe")
	with open('data/' + args.inpath, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter='\t')
		csv_writer.writerow(["tr_context", "tr_src", "en_context", "en_src"])
		for row in reader:
			nrows += 1

			tr_context = ' '.join(bpemb_tr.encode(row[0]))
			tr_src = ' '.join(bpemb_tr.encode(row[1]))

			en_context = ' '.join(bpemb_en.encode(row[2]))
			en_src = ' '.join(bpemb_en.encode(row[3]))

			csv_writer.writerow([tr_context, tr_src, en_context, en_src])
			
			if (nrows % 1000) == 0:
				outfile.flush()

			if (nrows % 10000) == 0:
				print(nrows)
	outfile.close()