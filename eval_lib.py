import torch
from utils_transform import *
from transformer import *
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
import spacy
import pdb

def rebatch_for_eval(pad_idx, batch):
  """Returns two batches: one where batch.trg, batch.trg_y match the correct
  translation; and one where they match the incorrect translation.
  """
  src = batch.src.transpose(0, 1)
  src_context = batch.src_context.transpose(0,1)
  trg_correct = batch.trg_correct.transpose(0,1)
  trg_incorrect = batch.trg_incorrect.transpose(0,1)
  return Batch(src, trg_correct, src_context, pad_idx), Batch(src, trg_incorrect, src_context, pad_idx)

def log_likelihood(model, batch, pad_idx):
    """Calculates the log likelihood of a batch, given the model.
    """
    memory = model.encode(batch) # [40 x 7 x 512] = [batch x srclen x dim]
    total_prob = torch.zeros(batch.trg_y.size(0))
    for i in range(0, batch.trg_y.size(1)): # trg_len
        y_prev = batch.trg[:, :i + 1]
        out = model.decode(memory, batch.src_mask, 
                           y_prev.clone().detach(),
                           (subsequent_mask(y_prev.size(1))
                                    .type_as(batch.src.data)))
        probs = model.generator(out[:, -1]) # batch x vocab
        trg_index = batch.trg_y[:, i]
        prob_of_trg = probs.gather(1, trg_index.view(-1,1)) # not sure about this
        pad_mask = (trg_index != pad_idx).to(dtype=torch.float32)
        total_prob += (prob_of_trg.squeeze()) * pad_mask
    return total_prob

def greedy_decode(model, batch, max_len, start_symbol):
    src = batch.src, src_mask = batch.src_mask # This is just wrong lol
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    total_prob = 0.0
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           ys.clone().detach(),
                           (subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        max_prob, next_word = torch.max(prob, dim = 1)
        total_prob += max_prob.data
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def beam_decode(model, src, src_mask, src_context, pad_idx, max_len, start_symbol, end_symbol, k=5):
    """Beam decoder.
    """
    batch = Batch(src.unsqueeze(0), src_mask.unsqueeze(0), src_context.unsqueeze(0), pad_idx)
    memory = model.encode(batch)
    ys = torch.ones(1,1).fill_(start_symbol).type_as(src.data)
    hypotheses = [(ys, 0.0)]
    for i in range(max_len):
      candidates_at_length = []
      for hypothesis, previous_prob in hypotheses:
        if hypothesis[0, -1] == end_symbol:
          candidates_at_length.append((hypothesis, previous_prob))
        else:
          # feed through model
          out = model.decode(memory, src_mask, 
                               hypothesis.clone().detach(),
                               (subsequent_mask(hypothesis.size(1))
                                        .type_as(src.data)))
          probs = model.generator(out[:, -1])
          # Keep track of top k predictions for each candidates
          top_probs, predictions_at_step = torch.topk(probs, k, dim=1)
          new_hypotheses = [torch.cat([hypothesis.clone(), pred.reshape(1,1)], dim=1) for pred in predictions_at_step.flatten()]
          new_probs = top_probs.flatten().data + previous_prob
          candidates_at_length = candidates_at_length + list(zip(new_hypotheses, new_probs))
      hypotheses = sorted(candidates_at_length, key = lambda x: x[1], reverse=True)[:k]
    return hypotheses[0][0].squeeze() # change to 1D tensor

def eval_bleu(pad_idx, eval_iter, model, max_len, start_symbol, end_symbol, rev_tokenize_trg, bpemb_en, out_path):
  """Calculates average BLEU score of a model on some validation iterator.
  """
  full_out_path = "data/bleu/" + out_path
  out_file = open(full_out_path, "w")
  bleus = []
  n_written = 0
  for old_batch in eval_iter:
    batch = rebatch(pad_idx, old_batch)
    for i in range(batch.src.size(0)): # batch_size
      max_len_for_observation = 30
      pdb.set_trace()
      hypothesis = beam_decode(model, batch.src[i], batch.src_mask[i], batch.src_context[i],
       pad_idx, max_len_for_observation, start_symbol, end_symbol, k=5)[1:-1] # cut off SOS, EOS
      targets = batch.trg_y[i, :-1] # Doesn't have SOS. Cut off EOS
      trg_str = bpemb_en.decode(rev_tokenize_trg(targets)).replace("<pad>", "")
      hypothesis_decoded = bpemb_en.decode(rev_tokenize_trg(hypothesis))
      hypothesis_clean = hypothesis_decoded.replace("<pad>", "")
      print("%d \t %s \t %s maxlen: %d" & (n_written, trg_str, hypothesis_clean, max_len_for_observation))
      out_file.write(hypothesis_clean + "\n")
      n_written +=1 
      if n_written % 50 == 0:
        out_file.flush()

  out_file.close()
  print("Wrote to ", out_path)
  return

def eval_accuracy_helper(pad_idx, eval_iter, model):
  """ Helper function that calculates how well the model chooses the correct
  pronoun on the data in an iterator.
  """
  n_correct = 0.0
  n_total = 0.0
  for b in eval_iter:
    batch_correct, batch_incorrect = rebatch_for_eval(pad_idx, b)
    probs = torch.stack([
      log_likelihood(model, batch_correct, pad_idx), 
      log_likelihood(model, batch_incorrect, pad_idx)], dim=1) # n x 2
    correct = probs[:, 0] > probs[:, 1] # should assign higher probability to the left
    n_correct += torch.sum(correct).item()
    n_total += correct.size(0)
  print("Correct: %d / %d = %f" % (n_correct, n_total, (n_correct / n_total)))
  return

def eval_accuracy(pad_idx, path_to_test_set, model, TR, EN):
  """Calculates how well the model chooses the correct pronoun, given
  a dataset in TSV form. 
  """
  full_path = "data/%s" % (path_to_test_set)
  # Must be in order
  # !!! IMPT note there are five fields
  data_fields = [
  ('src_context', TR), ('src', TR),
  ('trg_context', EN), ('trg_correct', EN), ('trg_incorrect', EN)]
  print('Evaluating discriminative dataset: %s ' % (full_path))
  test = TabularDataset(
    full_path,
    format='tsv', 
    fields=data_fields)

  test_iter = Iterator(
    test, batch_size=100, sort_key=lambda x: 1, repeat=False, train=False)

  eval_accuracy_helper(pad_idx, test_iter, model)

def load(path, tr_voc, en_voc, use_context):
    """Loads a trained model from memory for evaluation.
    """
    if use_context:
      model = make_context_model(tr_voc, en_voc, N=6)
    else:
      model = make_model(tr_voc, en_voc, N=6)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model