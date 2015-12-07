#! /usr/bin/python
from __future__ import print_function, division
import subprocess
import shlex


class HMM(object):

    def __init__(self):
        self.train_file = './gene.train'
        self.count_file = './gene.counts'
        self.key_file = './gene.key'
        self.dev_file = './gene.dev'
        self.dev_out_file = './gene_dev.p{0}.out'
        self.test_file = './gene.test'
        self.test_out_file = './gene_test.p{0}.out'

        self.emission_counts = {}  # Format: [word][tag] = count
        self.word_counts = {}  # Format [word] = count
        self.is_rare = {}  # Format [word] = True/False
        self.RARE_LIMIT = 5
        self._RARE_ = '_RARE_'

        self.ngrams = [None, {}, {}, {}]

    def _generate_count(self):
        open(self.count_file, 'w').close()
        command = 'python count_freqs.py {0} > {1}'.format(
            self.train_file, self.count_file)
        subprocess.call(command, shell=True)

    def _collect_counts(self):
        self.emission_counts = {}
        self.word_counts = {}
        self.is_rare = {}
        self.ngrams = [None, {}, {}, {}]

        with open(self.count_file, 'r') as f:
            for line in f:
                arr = line.replace('\n', '')
                if not arr:
                    continue

                arr = arr.split(' ')
                if arr[1] == 'WORDTAG':
                    count, _, tag, word = arr
                    count = int(count)
                    if word not in self.emission_counts:
                        self.emission_counts[word] = {}
                    self.emission_counts[word][tag] = count

                    if word not in self.word_counts:
                        self.word_counts[word] = 0

                    self.word_counts[word] += count

                elif arr[1] == '1-GRAM':
                    count, _, tag = arr
                    self.ngrams[1][tag] = int(count)

                elif arr[1] == '2-GRAM':
                    count, _, tag1, tag2 = arr
                    if tag1 not in self.ngrams[2]:
                        self.ngrams[2][tag1] = {}
                    self.ngrams[2][tag1][tag2] = int(count)

                elif arr[1] == '3-GRAM':
                    count, _, tag1, tag2, tag3 = arr
                    if tag1 not in self.ngrams[3]:
                        self.ngrams[3][tag1] = {}
                    if tag2 not in self.ngrams[3][tag1]:
                        self.ngrams[3][tag1][tag2] = {}
                    self.ngrams[3][tag1][tag2][tag3] = int(count)

    def replace_rare_words(self):
        self._generate_count()
        self._collect_counts()

        self.emission_counts[self._RARE_] = {}
        for word, word_count in self.word_counts.iteritems():
            if word_count < self.RARE_LIMIT:
                self.is_rare[word] = True
            else:
                self.is_rare[word] = False

            for tag, tag_count in self.emission_counts[word].iteritems():
                if tag not in self.emission_counts[self._RARE_]:
                    self.emission_counts[self._RARE_][tag] = tag_count
                else:
                    self.emission_counts[self._RARE_][tag] += tag_count

    def tag_gene_simple(self, test_file=None, out_file=None):
        '''
        Part 1: Simple gene tagger
        Assign the tag y to word x, with:
        y = argmax e(x|y)
        '''
        test_file = test_file or self.dev_file
        out_file = out_file or self.dev_out_file
        out_file = out_file.format(1)

        # Empty output file if exists
        open(out_file, 'w').close()
        output_data = ''
        with open(test_file, 'r') as f:
            for line in f:
                word = line.replace('\n', '')
                if word:
                    _word = word if word in self.is_rare and not self.is_rare[
                        word] else self._RARE_
                    max_prob = -1
                    max_tag = ''
                    for tag, e_count in self.emission_counts[_word].iteritems():
                        prob = e_count / self.ngrams[1][tag]
                        if prob > max_prob:
                            max_prob = prob
                            max_tag = tag

                    output_data += '{0} {1}\n'.format(word, max_tag)
                else:
                    output_data += line

        # Write output data to file
        with open(out_file, 'w') as f:
            f.write(output_data)

    def evaluate(self, key_file=None, result_file=None, partId=1):
        key_file = key_file or self.key_file
        result_file = result_file or self.dev_out_file
        result_file = result_file.format(partId)

        command = 'python eval_gene_tagger.py {0} {1}'.format(
            key_file, result_file)
        subprocess.call(command, shell=True)

if __name__ == '__main__':
    hmm = HMM()
    hmm.replace_rare_words()
    hmm.tag_gene_simple()
    hmm.evaluate()

    # Generate result file for submission
    hmm.tag_gene_simple(test_file=hmm.test_file, out_file=hmm.test_out_file)
