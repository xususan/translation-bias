"""
Make a dataset balanced wrt # of examples that mention a female pronoun vs a male pronouns
"""

import pdb
import csv
import sys
import argparse
import random

"""
Create new dataset of same size as original that is balanced wrt # of male/female mentions.
"""

random.seed(1)

parser = argparse.ArgumentParser(description='Create New Datasets (with special properties)')
parser.add_argument('--inpath', type=str, help='Path to infile (within data/)')
parser.add_argument('--outpath', type=str, help='Path to outfile (within data/)')
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

def print_marginal_pronoun_counts_iter(csv_iter):
    n_male = 0; n_female = 0; n_overlap = 0
    for row in csv_iter:
    	has_male = check_for_set_of_strings(row[3], male_pronouns)
    	has_female = check_for_set_of_strings(row[3], female_pronouns)
    	if has_male:
    		n_male += 1
    	if has_female:
    		n_female +=1
    	if has_male and has_female:
    		n_overlap +=1
    print("lines with male pronoun in target:", n_male)
    print("lines with female pronoun in target:", n_female)
    print('lines with both:', n_overlap)


# We need to remove 3000 'male' mentions

if __name__ == "__main__":
	outfile = open("data/" + args.outpath, 'w+', newline='')
	csv_writer = csv.writer(outfile, delimiter='\t')
	nrows = 0
	nwritten =0
	with open('data/' + args.inpath, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter='\t')
		for row in reader:
			nrows += 1
			if nrows < args.line:
				continue

			# If it has a male pronoun. Include with probability 1/ 1.8
			# Otherwise, write it.
			male_prob = .46/.655
			if not(check_for_set_of_strings(row[3], male_pronouns)) or random.random() < male_prob:
				csv_writer.writerow([row[0], row[1], row[2], row[3]])
				nwritten +=1

			if (nwritten % 1000) == 0:
				outfile.flush()

	# Pad with female samples to get to the correct size.
	female_file_path = "data/" + args.inpath[:-4] + "_female_pronouns.csv"

	extra_rows_needed = nrows - nwritten
	with open(female_file_path, newline='') as female_csv:
		reader = csv.reader(female_csv, delimiter='\t')

		# Count number of rows and reset to beginning when done.
		n_fem_entries = sum(1 for row in reader)
		female_csv.seek(0)
		
		for row in reader:
			if random.random() < extra_rows_needed/float(n_fem_entries - 100):
				csv_writer.writerow(row)
				nwritten +=1
			if (nwritten % 1000) == 0:
				outfile.flush()
			if nwritten == nrows:
				break

	outfile.close()
	print("Wrote to: data/%s" % args.outpath)

	with open("data/" + args.inpath, 'r', newline='') as original_file:
		reader = csv.reader(original_file, delimiter='\t')
		print("Before balancing:")
		print_marginal_pronoun_counts_iter(reader)

	with open("data/" + args.outpath, 'r', newline='') as just_written_file:
		reader = csv.reader(just_written_file, delimiter='\t')
		print("After balancing:")
		print_marginal_pronoun_counts_iter(reader)
