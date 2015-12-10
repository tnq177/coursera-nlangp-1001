#! /usr/bin/python
from __future__ import print_function, division
import sys
from decimal import *
import pdb
from math import log


class HMM(object):

    def __init__(self, count_file_handle):
        self.emissions = {}
        self.word_count = {}
        self.ngrams = {1: {}, 2: {}, 3: {}}
        self._RARE_ = '_RARE_'

        for line in count_file_handle:
            arr = line.replace('\n', '')

            if not arr:
                continue

            arr = arr.split(' ')
            count = int(arr[0])
            key = tuple(arr[2:])

            if arr[1] == '1-GRAM':
                self.ngrams[1][key[0]] = count
            elif arr[1] == '2-GRAM':
                self.ngrams[2][key] = count
            elif arr[1] == '3-GRAM':
                self.ngrams[3][key] = count
            if arr[1] == 'WORDTAG':
                self.emissions[key] = count
                self.word_count.setdefault(arr[-1], 0)
                self.word_count[arr[-1]] += count

        self.tags = self.ngrams[1].keys()
        count_file_handle.close()

        for word, count in self.word_count.iteritems():
            if count < 5:
                for tag in self.tags:
                    rare_key = (tag, self._RARE_)
                    key = (tag, word)
                    self.emissions.setdefault(rare_key, 0)
                    self.emissions[rare_key] += self.emissions.get(key, 0)

    def emission_prob(self, word, tag):
        if tag in ['*', 'STOP']:
            return 0

        _word = self.replace_word(word)
        return self.emissions.get((tag, _word), 0) / self.ngrams[1][tag]

    def trigram_prob(self, trigram_tuple):
        return self.ngrams[3].get(trigram_tuple, 0) / self.ngrams[2].get(trigram_tuple[:2])

    def replace_word(self, word):
        if self.word_count.get(word, 0) < 5:
            return self._RARE_
        else:
            return word

    def S(self, k):
        if k in [-1, 0]:
            return ['*']
        else:
            return self.tags


def unigram(hmm, sentence):
    tags = []
    append = tags.append

    for word in sentence:
        _word = hmm.replace_word(word)
        tag_e_list = [(tag, hmm.emission_prob(_word, tag))
                      for tag in hmm.tags]
        max_tag = max(tag_e_list, key=lambda x: x[1])[0]
        append(max_tag)

    return tags


def viberti(hmm, sentence):
    n = len(sentence)
    sentence = [''] + sentence
    tags = [''] * (n + 1)
    pi, bp = {}, {}
    pi[0, '*', '*'] = 0

    def tag_prob(k, v, w, u):
        q = hmm.trigram_prob((w, u, v))
        e = hmm.emission_prob(sentence[k], v)
        q = log(q) if q != 0 else float('-inf')
        e = log(e) if e != 0 else float('-inf')
        return (w, pi[k-1, w, u] + q + e)

    for k in xrange(1, n + 1):
        for u in hmm.S(k - 1):
            for v in hmm.S(k):
                tag_prob_list = [tag_prob(k, v, w, u) for w in hmm.S(k - 2)]
                bp[k, u, v], pi[k, u, v] = max(
                    tag_prob_list, key=lambda x: x[1])

    # Get y(n-1), y(n)
    max_prob = float('-inf')
    for u in hmm.S(n - 1):
        for v in hmm.S(n):
            q = hmm.trigram_prob((u, v, 'STOP'))
            q = log(q) if q != 0 else float('-inf')
            prob = pi[n, u, v] + q 
            if prob > max_prob:
                max_prob = prob
                tags[n - 1], tags[n] = u, v

    for k in xrange(n - 2, 0, -1):
        tags[k] = bp[k + 2, tags[k + 1], tags[k + 2]]

    return tags[1:]


def main(partId, test_file, out_file):
    count_file_handle = open('./gene.counts', 'r')
    hmm = HMM(count_file_handle)

    test_file_handle = open(test_file, 'r')
    open(out_file, 'w').close()
    out_file_handle = open(out_file, 'w')

    sentence = [line.replace('\n', '') for line in test_file_handle]
    sentence_without_empty_word = [word for word in sentence if word]
    test_file_handle.close()

    max_tags = None
    if partId == '1':
        max_tags = unigram(hmm, sentence_without_empty_word)
    elif partId == '2':
        max_tags = viberti(hmm, sentence_without_empty_word)

    out_data = ''
    counter = 0

    for word in sentence:
        if word:
            out_data += '{0} {1}\n'.format(word, max_tags[counter])
            counter += 1
        else:
            out_data += '\n'

    out_file_handle.write(''.join(out_data))
    out_file_handle.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(1)
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
