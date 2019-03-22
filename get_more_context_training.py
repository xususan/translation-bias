import pdb
import csv
import sys
import argparse
import random

"""
Get a subsample of the data that includes coreference between source and target.
"""

random.seed(1)

parser = argparse.ArgumentParser(description='Subsample Datasets (with wcoreference)')
parser.add_argument('--inpath', type=str, default="train_200k_annotated.csv", help='Path to infile (within data/). Must be annotated')
parser.add_argument('--outpath', type=str, default="sample_context_test.csv", help='Path to outfile (within data/)')
parser.add_argument('--line', type=int, default=0, help='Line on which to resume, if any')
args = parser.parse_args()

female_pronouns = set(['she', 'her', 'hers', "she's", "she'll"])
male_pronouns = set(['he', 'him', 'his', "he's", "he'll"])

# For individual string within a longer string.
def check_for_string(string, substr):
    return string.startswith(substr + " ") or (" " + substr + " ") in string or string.endswith(" " + substr)

def check_for_set_of_strings(string, iter_of_strings):
    for substr in iter_of_strings:
        if check_for_string(string, substr):
        	return True
    return False

assert(check_for_set_of_strings(("hello you there"), ["hello"]))
assert(check_for_string(("hello you there"), "hello"))

if __name__ == "__main__":
	outfile = open("data/" + args.outpath, 'w+', newline='')
	csv_writer = csv.writer(outfile, delimiter='\t')
	nrows = 0
	nwritten =0
	max_non_context = 1747621
	n_non_context = 0
	with open('data/' + args.inpath, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter='\t')
		for row in reader:
			nrows += 1
			if nrows < args.line:
				continue

			if row[0] == 'tr_context':
				continue

			if len(row[4]) > 2: # Indicates more than '[]'
				print(nrows - 1, row[4])
				csv_writer.writerow([row[0], row[1], row[2], row[3]])
				nwritten+=1
			else:
				if random.random() < .873 and n_non_context < max_non_context:
					csv_writer.writerow([row[0], row[1], row[2], row[3]])
					nwritten +=1
					n_non_context +=1
			
			if (nrows % 1000) == 0:
				outfile.flush()

			if nwritten == 2E6:
				break
	outfile.close()