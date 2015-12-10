#! /usr/bin/python
from __future__ import print_function, division
import sys 

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
        _word = self.replace_word(word)
        return self.emissions.get((tag, _word), 0) / self.ngrams[1][tag]

    def replace_word(self, word):
        if self.word_count.get(word, 0) < 5:
            return self._RARE_
        else:
            return word


def unigram(hmm, sentence):
    tags = []
    append = tags.append

    for word in sentence:
        if not word:
            append('')
        else:
            _word = hmm.replace_word(word)
            tag_e_list = [(tag, hmm.emission_prob(_word, tag)) for tag in hmm.tags]
            max_tag = max(tag_e_list, key=lambda x: x[1])[0]
            append(max_tag)

    return tags 

def main(test_file, out_file):
    count_file_handle = open('./gene.counts', 'r')
    hmm = HMM(count_file_handle)

    test_file_handle = open(test_file, 'r')
    open(out_file, 'w').close()
    out_file_handle = open(out_file, 'w')

    sentence = [line.replace('\n', '') for line in test_file_handle]
    test_file_handle.close()

    max_tags = unigram(hmm, sentence)
    out_data = ['{0} {1}\n'.format(word, tag) for word, tag in zip(sentence, max_tags)]
    out_file_handle.write(''.join(out_data))
    out_file_handle.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(1)
    else:
        main(sys.argv[1], sys.argv[2])



