import argparse
import logging
import xml.etree.ElementTree as ET
import pdb
import gzip
from xml.dom.minidom import parse, parseString
import datetime
import random
import numpy as np
import csv

'''
Rather than writing everything at the end, try writing things line by line.
'''
# Set up parser for arguments
parser = argparse.ArgumentParser(description='Data Processing')
parser.add_argument('--size', type=str, default="full", help='Size of file (full or mini)')
args = parser.parse_args()

seed = 1
np.random.seed(seed)
random.seed(seed)

print(f"Processing {args.size} size data...")
TR_DIR = "data/tr/"
EN_DIR = "data/en/"

if args.size == "full":
    # create train and validation set
    # train_size, val_size, test_size = int(2E8), 10000, 10000
    train_size, val_size, test_size = int(2E8),  int(2E8),  int(2E8)
    print(f"training size: {train_size}, validation size: {val_size}")
    out_paths = {
        'train': "data/train_2m_0306.csv", 
        'val': "data/val_10k_0306.csv", 
        'test':"data/test_10k_0306.csv"}
    xml_path = "data/en-tr.xml"
elif args.size == "mini":
    # create train and validation set
    train_size, val_size, test_size = 1000, 100, 100
    print(f"training size: {train_size}, validation size: {val_size}")
    out_paths = {
        'train': "data/train_mini_0306.csv", 
        'val': "data/val_mini_0306.csv", 
        'test':"data/test_mini_0306.csv"}
    xml_path = "data/en-tr-mini.xml"

sizes = {
    'train': train_size,
    'val': val_size,
    'test': test_size
}

def get_text(el):
    text = el.text.strip() if el.text.strip() != "" else el[0].tail.strip()
    return text

def remove_trailing_ellipses(string):
    if string.endswith('...'):
        return string[:-2]
    elif string.endswith('..'):
        return string[:-1]
    else:
        return string

remove_dashes = lambda s: s[1:].strip() if s.startswith('-') else s
def process_example(example):
    example = map(remove_dashes, example)
    example = map(str.lower, example)
    example = map(remove_trailing_ellipses, example)
    return example
    

def process_links(link_groups, size, file_path):
    """
    size: max size for this group of links
    """
    print(file_path)
    outfile = open(file_path, 'w+', newline='')
    csv_writer = csv.writer(outfile, delimiter='\t')
    csv_writer.writerow(["tr_context", "tr", "en_context", "en"])
    src_list, trg_list, src_context_list, trg_context_list = [], [], [], []
    n_written = 0
    for link_group in link_groups:
        print(link_group.attrib['fromDoc'],link_group.attrib['toDoc'])
        # open and read gzipped xml file
        src_doc = "data/" + link_group.attrib['fromDoc'][:-3]
        tgt_doc = "data/" + link_group.attrib['toDoc'][:-3]
        try:
            tgt_file = ET.parse(tgt_doc).getroot() # take off .gz
            src_file = ET.parse(src_doc).getroot() # take off .gz
        except Exception as e:
            print(str(e))
            continue
        for link in link_group:
            if 'overlap' in link.attrib and float(link.attrib['overlap']) > .9:
                # Acceptable
                src_align, trg_align = link.attrib['xtargets'].split(';')

                # Ignore things with multiple lines.
                # Also ignore the first line, because it will have no context.
                if ' ' in src_align or ' ' in trg_align or int(src_align) == 1 or int(trg_align) == 1:
                    continue

                trg_sentence = tgt_file.find('.//s[@id="%s"]' % (trg_align))
                trg_text = get_text(trg_sentence)

                src_sentence = src_file.find('.//s[@id="%s"]' % (src_align))
                src_text = get_text(src_sentence)
                if(len(trg_text) == 0 or len(src_text) == 0):
                    continue

                # Check for a timestamp.
                if len(trg_sentence) == 0 or len(src_sentence) == 0:
                    continue

                # Find context sentence
                trg_context = tgt_file.find('.//s[@id="%d"]' % (int(trg_align) - 1))
                src_context = src_file.find('.//s[@id="%d"]' % (int(src_align) - 1))
                trg_context_text = get_text(trg_context)
                src_context_text = get_text(src_context)
                if trg_context is None or len(trg_context) == 0 or src_context is None or len(src_context) == 0:
                    continue

                # Check that previous sentence is within the last 7 seconds

                current_time = trg_sentence[0].get('value')
                previous_time = trg_context[0].get('value')
                str_to_time = lambda t: datetime.timedelta(
                    hours=int(t[:2]), minutes=int(t[3:5]), seconds=int(t[6:8]))
                try:
                    time_objects = [str_to_time(x) for x in [previous_time, current_time]]
                except:
                    continue
                use_context = ((time_objects[1] - time_objects[0]) < datetime.timedelta(seconds=7))
                if use_context:
                    training_example = process_example(
                        [trg_context_text, trg_text, src_context_text, src_text])
                    csv_writer.writerow(training_example)
                    n_written +=1

                if n_written % 100 == 0:
                    print("%d / %d" %(n_written, size))

                if n_written >= size:
                    break

    outfile.close()
    return src_list, trg_list, src_context_list, trg_context_list

def process_df(df):
    # Remove dashes in the beginning when they occur.
    remove_dashes = lambda s: s[1:].strip() if s.startswith('-') else s
    df = df.applymap(remove_dashes)

    # Lowercase
    df = df.applymap(str.lower)
    return df

# Divide document IDs.
def main():
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Split list of documents
    doc_list = [link_group for link_group in root]
    np.random.shuffle(doc_list)
    num_docs = len(doc_list)

    train, validate, test = np.split(doc_list, [int(.6*num_docs), int(.8*num_docs)])
    split = {
        'val': validate,
        'test': test,
        'train': train}

    for split_name, link_groups in split.items():
        file_path = out_paths[split_name]
        size = sizes[split_name]
        process_links(link_groups, size, file_path)
        print("Wrote to %s" % (file_path))

def subsample_csv(in_path, out_path, n):
    print("Reading from %s..." % (in_path))
    df = pd.read_csv(in_path, sep='\t')
    sampled = df.sample(n, random_state = seed)
    df.to_csv(out_path, index=False, sep='\t')
    print("Wrote to %s..." % (out_path))

# subsample_csv("data/train_2m.csv", "data/train_200k.csv", 200000)

main()    




