
import pdb
import csv
import sys
import argparse

parser = argparse.ArgumentParser(description='Convert training csv to a corpus to use for Glove embeddings')
parser.add_argument('--inpath', type=str, default="full", help='Path to infile (within data/)')
parser.add_argument('--outpath', type=str, default="full", help='Path to outfile (within data/)')
args = parser.parse_args()


list_of_punct = [",", ".", "?", "!", "'"]
def main():
	outfile = open("data/" + args.outpath, 'w+', newline='')
	csv_writer = csv.writer(outfile, delimiter='\t')
	with open('data/' + args.inpath, newline='') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter='\t')
		for row in csv_reader:
			new_row = []
			for string in row:
				for punct in list_of_punct:
					string = string.replace(" %c" % punct, punct)
				new_row.append(string)
			csv_writer.writerow(new_row)
			
	print("wrote to %s" % args.outpath)

if __name__ == "__main__":
	main()