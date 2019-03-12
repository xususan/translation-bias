import pdb
import csv
import sys
import argparse

"""
Get a sub sample of the data that includes pronouns with `she'.
"""

parser = argparse.ArgumentParser(description='Create New Datasets (with special properties)')
parser.add_argument('--inpath', type=str, default="full", help='Path to infile (within data/)')
parser.add_argument('--outpath', type=str, default="full", help='Path to outfile (within data/)')
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
	with open('data/' + args.inpath, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter='\t')
		for row in spamreader:
			nrows += 1
			if nrows < args.line:
				continue
			# Join the last two
			english_str_and_context = ' '.join(row[2:4])

			if check_for_set_of_strings(row[2], female_pronouns) or check_for_set_of_strings(row[3], female_pronouns):
				csv_writer.writerow([row[0], row[1], row[2], row[3]])
			
			if (nrows % 1000) == 0:
				outfile.flush()
	outfile.close()