#! /usr/bin/python
from __future__ import print_function, division
import sys
from decimal import *
import json

class PCFG(object):

    def __init__(self, count_file):
        self.nonterminal_count = {}
        self.binary_count = {}
        self.unary_count = {}
        self.word_count = {}

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
        words = [
            x for x in nonterminals_or_words if x not in self.nonterminals]

        for word in words:
            word_count = sum(
                [self.unary_count.get((nonterminal, word), 0) for nonterminal in self.nonterminals])

            self.word_count[word] = word_count

            if word_count < 5:
                self.word_count.setdefault('_RARE_', 0)
                self.word_count['_RARE_'] += word_count

                for nonterminal in self.nonterminals:
                    self.unary_count.setdefault((nonterminal, '_RARE_'), 0)
                    self.unary_count[(nonterminal, '_RARE_')] += self.unary_count.get((nonterminal, word), 0)

    def binary_prob(self, key):
        return self.binary_count.get(key, 0) / self.nonterminal_count[key[0]]

    def unary_prob(self, key):
        _key = list(key)
        _key[1] = _key[1] if _key[1] in self.nonterminals else self.replace_word(_key[1])
        _key = tuple(_key)
        return self.unary_count.get(_key, 0) / self.nonterminal_count[_key[0]]

    def replace_word(self, word):
        if word not in self.word_count or self.word_count[word] < 5:
            return '_RARE_'
        else:
            return word

    def parse(self, sent):
        '''
        Expect sentence to be a string
        '''
        sent = sent.replace('\n', '').split(' ')
        n = len(sent)
        sent = [''] + sent

        pi, bp = {}, {}
        for i in xrange(1, n + 1):
            xi = sent[i]
            for X in self.nonterminals:
                pi[i, i, X] = Decimal(self.unary_prob((X, xi)))
                bp[i, i, X] = xi 

        for l in xrange(1, n):
            for i in xrange(1, n - l + 1):
                j = i + l 

                max_prob = {}
                max_nonterminals = {}
                for s in xrange(i, j):
                    for key, count in self.binary_count.iteritems():
                        X, Y, Z = key
                        max_prob.setdefault(X, -1)
                        max_nonterminals.setdefault(X, '')

                        prob = Decimal(self.binary_prob(key)) * pi.get((i, s, Y), Decimal(0)) * pi.get((s + 1, j, Z), Decimal(0))

                        if prob > max_prob[X]:
                            max_prob[X] = prob
                            max_nonterminals[X] = (Y, Z, s)

                for key, count in self.binary_count.iteritems():
                    X = key[0]
                    pi[i, j, X] = max_prob[X]
                    bp[i, j, X] = max_nonterminals[X]

        return bp, n

    def tree_to_array(self, bp, start_nonterminal, n):
        def _to_array(start, end, key):
            if start == end:
                return [key, bp[start, end, key]]
            else:
                Y, Z, s = bp[start, end, key]
                return [key, _to_array(start, s, Y), _to_array(s + 1, end, Z)]

        return _to_array(1, n, start_nonterminal)


def write_new_count_file(pcfg):
    count_file = './cfg.counts'
    out_file = './parse_train.counts.out'

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

def main(args):
    if len(args) < 2:
        sys.exit(1)

    pcfg = PCFG('./cfg.counts')

    if args[1] == '1':
        write_new_count_file(pcfg)
    elif args[1] == '2':
        if len(args) < 4:
            sys.exit(1)

        open(args[3], 'w').close()
        test_file_handle = open(args[2], 'r')
        out_file_handle = open(args[3], 'w')

        for line in test_file_handle:
            sent = line.replace('\n', '')
            if sent:
                bp, n = pcfg.parse(sent)
                array_json_string = json.dumps(pcfg.tree_to_array(bp, 'SBARQ', n))
                out_file_handle.write('{0}\n'.format(array_json_string))

        test_file_handle.close()
        out_file_handle.close()

    else:
        print('Not implemented!')

if __name__ == '__main__':
    main(sys.argv)
