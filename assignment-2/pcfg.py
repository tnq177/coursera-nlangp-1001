#! /usr/bin/python
from __future__ import print_function, division
import sys


class PCFG(object):

    def __init__(self, count_file):
        self.nonterminal_count = {}
        self.binary_count = {}
        self.unary_count = {}
        self.word_count = {}
        self.words = []

        with open(count_file, 'r') as count_file_handle:
            for line in count_file_handle:
                arr = line.replace('\n', '')
                if not arr:
                    continue

                arr = arr.split(' ')
                count = int(arr[0])
                key = tuple(arr[2:])

                if arr[1] == 'NONTERMINAL':
                    self.nonterminal_count[key[0]] = count
                elif arr[1] == 'UNARYRULE':
                    self.unary_count[key] = count
                elif arr[1] == 'BINARYRULE':
                    self.binary_count[key] = count

        count_file_handle.close()
        self.nonterminals = self.nonterminal_count.keys()
        self.replace_rare()

    def replace_rare(self):
        unary_count_keys = self.unary_count.keys()
        nonterminals_or_words = list(set([x[1] for x in unary_count_keys]))
        self.words = [
            x for x in nonterminals_or_words if x not in self.nonterminals]

        for word in self.words:
            word_count = sum(
                [self.unary_count.get((nonterminal, word), 0) for nonterminal in self.nonterminals])

            self.word_count[word] = word_count

            if word_count < 5:
                self.word_count.setdefault('_RARE_', 0)
                self.word_count['_RARE_'] += word_count

                for nonterminal in self.nonterminals:
                    self.unary_count.setdefault((nonterminal, '_RARE_'), 0)
                    self.unary_count[(nonterminal, '_RARE_')] += word_count


def write_new_count_file():
    count_file = './cfg.counts'
    out_file = './parse_train.counts.out'

    pcfg = PCFG(count_file)

    # Check if unary line with rare word has already been written down
    visited = {}
    open(out_file, 'w').close()
    out_file_handle = open(out_file, 'w')
    count_file_handle = open(count_file, 'r')
    for line in count_file_handle:
        arr = line.replace('\n', '')
        if arr:
            arr = arr.split(' ')
            if arr[1] == 'NONTERMINAL' or arr[1] == 'BINARYRULE':
                out_file_handle.write('{0}\n'.format(' '.join(arr)))
            else:
                word = arr[-1]

                if word in pcfg.nonterminals:
                    out_file_handle.write('{0}\n'.format(' '.join(arr)))
                else:
                    # This is a word, check if it's rare
                    if pcfg.word_count[word] < 5:
                        arr[-1] = '_RARE_'
                        key = tuple(arr[-2:])
                        if key not in visited:
                            visited[key] = True
                            out_file_handle.write(
                                '{0}\n'.format(' '.join(arr)))
                        else:
                            continue
                    else:
                        out_file_handle.write('{0}\n'.format(' '.join(arr)))

    out_file_handle.close()
    count_file_handle.close()


def main(partId):
    if partId == '1':
        write_new_count_file()
    else:
        print('Not implemented!')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    else:
        main(sys.argv[1])
