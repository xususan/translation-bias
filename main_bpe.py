import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
import spacy
import argparse
from utils_transform import *
from transformer import *
import pdb
import time, datetime
import sys
import torch.nn as nn

# Set up parser for arguments
parser = argparse.ArgumentParser(description='Data Processing')
parser.add_argument('--size', type=str, default="full", help='Size of file (full, mid, mini)')
parser.add_argument('--train', type=str, default="None", help='Train path (within data/) if it differs from size default.')
parser.add_argument('--batch', type=int, default=512, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--save', type=int, default=10, help='Save model after every x intervals')
parser.add_argument('--out', type=str, default="save", help='Prefix for model output, eg save for save_10.pt')
parser.add_argument('--context', dest='context', action='store_true')
parser.add_argument('--no-context', dest='context', action='store_false')
parser.add_argument('--load', type=str, default="None", help="model to resume training if any")
parser.set_defaults(context=False)
args = parser.parse_args()

now = datetime.datetime.now()
month, day = pad_date(now.month), pad_date(now.day)

# Arguments and globals
print("Command line arguments: {%s}" % args)
# Parse args into some more parameters.
params = Params(args)
print("Train: %s, Val: %s, test: %s" % (params.train_csv, params.val_csv, params.test_csv))
print("Vocab size: %d" % (params.vocab_size))

train, val, test, TR_SRC, TR_CONTEXT, EN = load_train_val_test_datasets(params)
TR = TR_SRC
pad_idx = EN.vocab.stoi[PAD]


if args.context:
  model = make_context_model(len(TR.vocab), len(EN.vocab), N=6)
else:
  model = make_model(len(TR.vocab), len(EN.vocab), N=6)
  # if args.load != "None":
  #   model.load_state_dict(torch.load(args.load))

criterion = LabelSmoothing(size=len(EN.vocab), padding_idx=pad_idx, smoothing=0.1)

if torch.cuda.device_count() > 0:
  device = torch.device('cuda', 0)
  print('GPUs available:', torch.cuda.device_count())
  if torch.cuda.device_count() > 1:
    print("Using DataParallel")
    model = nn.DataParallel(model)
    criterion = nn.DataParallel(criterion)
    model.to(device)
    criterion.to(device)
  else:
    model.cuda()
    criterion.cuda()
else:
  device = torch.device('cpu')

# added src context sorting
train_iter = MyIterator(train, batch_size=args.batch, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg), len(x.src_context)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=args.batch, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg), len(x.src_context)),
                        batch_size_fn=batch_size_fn, train=False)
print('Iterators built.')

print('Training model...')
if torch.cuda.device_count() > 1:
  model_opt = NoamOpt(model.module.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

else:
  model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(1, args.epochs + 1):
    print("Epoch %d / %d" % (epoch, args.epochs))
    model.train()
    gen = model.module.generator if torch.cuda.device_count() > 1 else model.generator
    start_of_epoch = time.time()
    training_loss = run_epoch((rebatch(pad_idx, b) for b in train_iter), 
              model, 
              SimpleLossCompute(gen, criterion, 
                                opt=model_opt))
    epoch_time = time.time() - start_of_epoch
    print("Training loss: %f, elapsed time: %f" % (training_loss.data.item(), epoch_time))
    model.eval()
    loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                      model, 
                      SimpleLossCompute(gen, criterion, 
                      opt=None))
    print("Validation loss: %f" % loss.data.item())
    if epoch % args.save == 0: 
      # Export model
      output_path = "models/%s%s_%s_%d.pt" % (month, day, args.out, epoch)
      torch.save(model.state_dict(), output_path)
      print("Saved model to %s." %  output_path)

