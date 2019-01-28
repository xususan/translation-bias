import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
import spacy
import argparse
from utils_transform import *
from transformer import *
import pdb

# Set up parser for arguments
parser = argparse.ArgumentParser(description='Data Processing')
parser.add_argument('--size', type=str, default="full", help='Size of file (full or mini)')
parser.add_argument('--batch', type=int, default=512, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--save', type=int, default=10, help='Save model after every x intervals')
parser.add_argument('--out', type=str, default="save", help='Prefix for model output, eg save for save_10.pt')
args = parser.parse_args()

# Arguments and globals
print("Command line arguments: {%s}" % args)
if args.size == "mini":
	VOCAB_SIZE = 20
	train_csv, val_csv, test_csv = "train_mini.csv", "val_mini.csv", "test_mini.csv"
else:
	VOCAB_SIZE = 50000
	train_csv, val_csv, test_csv = "train_2m.csv", "val_10k.csv", "test_10k.csv"

BATCH_SIZE = args.batch

# DATA LOADING
en = spacy.load('en')
def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

SOS, EOS, PAD = "<s>", "</s>", "<pad>"
# Context and source / target fields for English + Turkish
TR = Field(init_token = SOS, eos_token =EOS, lower=True)
EN = Field(tokenize=tokenize_en, lower=True)

# Must be in order
data_fields = [
	('src_context', TR), ('src', TR),
	('trg_context', EN), ('trg', EN)]

train, val, test = TabularDataset.splits(
	path='data/', 
	train=train_csv,
	validation=val_csv,
	test=test_csv,
	format='tsv', 
	fields=data_fields)


print('Building vocab...')
MIN_FREQ = 5
TR.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
EN.build_vocab(train, min_freq=MIN_FREQ, max_size=VOCAB_SIZE)
print("TR vocab size: %d, EN vocab size: %d" % (len(TR.vocab), len(EN.vocab)))
print('Done building vocab')

devices = range(torch.cuda.device_count())
pad_idx = EN.vocab.stoi[PAD]
model = make_model(len(TR.vocab), len(EN.vocab), N=6)
criterion = LabelSmoothing(size=len(EN.vocab), padding_idx=pad_idx, smoothing=0.1)

if torch.cuda.device_count() > 0:
	print('GPUs available:', torch.cuda.device_count())
	model.cuda()
	criterion.cuda()
train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda', 0),
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda', 0),
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)
print('Iterators built.')

print('Training model...')
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(1, args.epochs):
    print("Epoch %d / %d" % (epoch, args.epochs))
    model.train()
    run_epoch((rebatch(pad_idx, b) for b in train_iter), 
              model, 
              SimpleLossCompute(model.generator, criterion, 
                                opt=model_opt))
    model.eval()
    loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                      model, 
                      SimpleLossCompute(model.generator, criterion, 
                      opt=None))
    print("Validation loss: %f" % {loss.data.item()})
    if epoch % args.save == 0: 
      # Export model
      output_path = "models/%s_%d.pt" %( args.out, epoch)
      torch.save(model.state_dict(), PATH)
      print("Saved model to %s." %  {output_path})

