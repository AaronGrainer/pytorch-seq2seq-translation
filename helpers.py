import torch
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromSentence(lang, sentence):
  return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(eos_token, lang, sentence):
  indexes = indexesFromSentence(lang, sentence)
  indexes.append(eos_token)
  return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(eos_token, input_lang, output_lang, pair):
  input_tensor = tensorFromSentence(eos_token, input_lang, pair[0])
  target_tensor = tensorFromSentence(eos_token, output_lang, pair[1])
  return (input_tensor, target_tensor)


def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
  plt.figure()
  fig, ax = plt.subplots()
  # this locator puts ticks at regular intervals
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)


def showAttention(input_sentence, output_words, attentions):
  # Set up figure with colorbar
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(attentions.numpy(), cmap='bone')
  fig.colorbar(cax)

  # Set up axes
  ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
  ax.set_yticklabels([''] + output_words)

  # Show label at every tick
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()






