from stanfordnlp.server import CoreNLPClient
import pdb
import csv
import sys
import argparse

# text = "Chris Manning is a nice person. He also gives oranges to people."

# print(text)

parser = argparse.ArgumentParser(description='Coreference Annotator')
parser.add_argument('--inpath', type=str, default="full", help='Path to infile (within data/)')
parser.add_argument('--outpath', type=str, default="full", help='Path to outfile (within data/)')
parser.add_argument('--line', type=int, default=0, help='Line on which to resume, if any')
args = parser.parse_args()


# set up the client
print('---')
print('starting up Java Stanford CoreNLP Server...')

# Finds pronouns in the second sentence that have at least one antecedent in the 
# first sentence.

# Pronominal, 1 and (Not pronominal, 0)
def find_pronouns(annotated):
	if len(annotated.sentence) < 2: 
		print("Find pronouns did not get a pair of sentences.")
		return None
	pronouns = []
	for coreferents in annotated.corefChain:
		# If there is only one coreferent, there cannot be a pronoun with antecedent.
		if len(coreferents.mention) < 2: # No antecedent
			continue
		# Make sure there is a pronoun in the second sentence,
		# and at least one non-pronoun coreferent in the first sentence
		pronouns_in_chain = []
		antecedent_found = False
		for mention in coreferents.mention:
			# found a pronoun
			if mention.mentionType == "PRONOMINAL" and mention.sentenceIndex == 1: 
				pronoun_index = mention.beginIndex
				pronoun = annotated.sentence[1].token[pronoun_index].word.lower()
				pronouns_in_chain.append(pronoun)
			# found antecedent
			elif mention.mentionType != "PRONOMINAL": # in any location 
				antecedent_found = True
		if antecedent_found:
			pronouns += pronouns_in_chain
	return pronouns


if __name__ == "__main__":
	outfile = open("data/" + args.outpath, 'w+', newline='')
	csv_writer = csv.writer(outfile, delimiter='\t')
	nrows = 0
	with open('data/' + args.inpath, newline='') as csvfile:
		with CoreNLPClient(annotators=['coref'], timeout=50000, memory='8G') as client:
			spamreader = csv.reader(csvfile, delimiter='\t')
			for row in spamreader:
				nrows += 1
				if nrows < args.line:
					continue
				# Join the last two
				english_str_and_context = ' '.join(row[2:4])
				ann_1 = client.annotate(english_str_and_context)
				res = find_pronouns(ann_1)
				csv_writer.writerow([row[0], row[1], row[2], row[3], res])
				if (nrows % 1000) == 0:
					outfile.flush()
	outfile.close()

