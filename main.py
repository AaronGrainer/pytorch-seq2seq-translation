from __future__ import unicode_literals, print_function, division
from models import EncoderRNN, DecoderRNN, AttnDecoderRNN
from helpers import indexesFromSentence, tensorFromSentence, tensorsFromPair, asMinutes, timeSince, showPlot, showAttention
import numpy as np
from io import open
import unicodedata
import string
import re
import random
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

eng_prefixes = (
  "i am ", "i m ",
  "he is", "he s ",
  "she is", "she s",
  "you are", "you re ",
  "we are", "we re ",
  "they are", "they re "
)

HIDDEN_SIZE = 256
teacher_forcing_ratio = 0.5

class Lang:
  def __init__(self, name):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1: "EOS"}
    self.n_words = 2  # Count SOS and EOS

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1


class Dataset:
  def unicodeToAscii(self, s):
    return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
    )

  def normalizeString(self, s):
    s = self.unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

  def readLangs(self, lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
      pairs = [list(reversed(p)) for p in pairs]
      input_lang = Lang(lang2)
      output_lang = Lang(lang1)
    else:
      input_lang = Lang(lang1)
      output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

  def filterPair(self, p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
      len(p[1].split(' ')) < MAX_LENGTH and \
      p[1].startswith(eng_prefixes)

  def filterPairs(self, pairs):
    return [pair for pair in pairs if self.filterPair(pair)]

  def load_dataset(self, lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = self.filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
      input_lang.addSentence(pair[0])
      output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class Model:
  def __init__(self, input_lang, output_lang, pairs, encoder, decoder):
    self.input_lang = input_lang
    self.output_lang = output_lang
    self.pairs = pairs
    self.encoder = encoder
    self.decoder = decoder

  def train(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = self.encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
      encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
      encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
      # Teacher forcing: Feed the target as the next input
      for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    else:
      # Without teacher forcing: use its own predictions as the next input
      for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
          break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

  def trainIters(self, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(EOS_token, self.input_lang, self.output_lang, random.choice(self.pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
      training_pair = training_pairs[iter - 1]
      input_tensor = training_pair[0]
      target_tensor = training_pair[1]

      loss = self.train(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion)
      print_loss_total += loss
      plot_loss_total += loss

      if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

      if iter % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

    showPlot(plot_losses)

  def evaluate(self, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
      input_tensor = tensorFromSentence(EOS_token, self.input_lang, sentence)
      input_length = input_tensor.size()[0]
      encoder_hidden = self.encoder.initHidden()

      encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)

      for ei in range(input_length):
        encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

      decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

      decoder_hidden = encoder_hidden

      decoded_words = []
      decoder_attentions = torch.zeros(max_length, max_length)

      for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
          decoded_words.append('<EOS>')
          break
        else:
          decoded_words.append(self.output_lang.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()

      return decoded_words, decoder_attentions[:di + 1]

  def evaluateRandomly(self, n=10):
    for i in range(n):
      pair = random.choice(self.pairs)
      print('>', pair[0])
      print('=', pair[1])
      output_words, attentions = self.evaluate(pair[0])
      output_sentence = ' '.join(output_words)
      print('<', output_sentence)
      print('')


  def evaluateAndShowAttention(self, input_sentence):
    output_words, attentions = self.evaluate(input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


def main():
  dataset = Dataset()
  input_lang, output_lang, pairs = dataset.load_dataset('eng', 'fra', True)
  print(random.choice(pairs))

  encoder1 = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
  attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, max_length=MAX_LENGTH, dropout_p=0.1).to(device)

  model = Model(input_lang, output_lang, pairs, encoder1, attn_decoder1)
  model.trainIters(75000, print_every=1000)
  model.evaluateRandomly()

  model.evaluateAndShowAttention("je suis trop froid .")
  model.evaluateAndShowAttention("elle a cinq ans de moins que moi .")
  model.evaluateAndShowAttention("elle est trop petit .")
  model.evaluateAndShowAttention("je ne crains pas de mourir .")
  model.evaluateAndShowAttention("c est un jeune directeur plein de talent .")

if __name__ == "__main__":
  main()
