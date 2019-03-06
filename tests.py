from coref_annotator import find_pronouns
from stanfordnlp.server import CoreNLPClient


def test_find_pronouns():
	with CoreNLPClient(annotators=['coref'], timeout=50000, memory='6G') as client:
		ann_1 = client.annotate("Chris Manning is a nice person. \
								 He also gives oranges to people.")
		assert(find_pronouns(ann_1) == ["he"])
		ann_2 = client.annotate("why is your mom sleeping that way?	\
								 it's because she wraps rice at night.")
		assert(find_pronouns(ann_2) == ["she"])
		ann_3 = client.annotate("allison is scott's everything.	she is his whole life.")
		assert(find_pronouns(ann_3) == ["she"])
		ann_4 = client.annotate("there once was a high school girl named song-yi. \
								 she fell in love with her college student tutor.")
		assert("she" in find_pronouns(ann_4))

		# Nothing returns because even though there is a pronoun, it has no antecedent 
		ann_5 = client.annotate("who is she? \
								 she fell in love with her college student tutor.")
		assert(find_pronouns(ann_5) == [])

test_find_pronouns()