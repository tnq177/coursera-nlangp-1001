#! /usr/bin/python
from __future__ import print_function, division
import sys
import pickle
from os.path import isfile, exists
from itertools import izip
import timeit

class IBM1(object):

    def __init__(self, source_corpus, target_corpus, pickle_file):
        # Really dumb way to determine if there is a pretrained model saved in
        # file
        if exists(pickle_file) and isfile(pickle_file):
            print('Loading pretrained model')
            with open(pickle_file, 'r') as f:
                self.t = pickle.load(f)
        else:
            print('Training')
            self.train(source_corpus, target_corpus, pickle_file)

    def train(self, source_corpus, target_corpus, pickle_file):
        self.t = {}

        f = open(source_corpus, 'r')
        source_file = f.readlines()
        f.close()
        f = open(target_corpus, 'r')
        target_file = f.readlines()
        f.close()

        # Calculate n(e)
        for src, tar in izip(source_file, target_file):
            src_sent = src.strip()
            tar_sent = tar.strip()

            if not src_sent or not tar_sent:
                continue

            src_sent = src_sent.split(' ')
            tar_sent = tar_sent.split(' ')

            src_sent_set = set(src_sent)
            tar_sent_set = set(['NULL'] + tar_sent)

            for e in tar_sent_set:
                self.t.setdefault(e, {})
                for f in src_sent_set:
                    self.t[e].setdefault(f, 0)

        for e in self.t:
            value = 1.0 / len(self.t[e].keys())
            for key in self.t[e]:
                self.t[e][key] = value

        def t(f, e):
            self.t.setdefault(e, {})
            self.t[e].setdefault(f, 0.0)

            return self.t[e][f]

        # Train with EM algorithm
        S = 5
        for s in xrange(0, S):
            print('Iteration: {0}'.format(s + 1))
            c = {}
            for src, tar in izip(source_file, target_file):
                src_sent = src.strip()
                tar_sent = tar.strip()

                if not src_sent or not tar_sent:
                    continue

                src_sent = src_sent.split(' ')
                tar_sent = tar_sent.split(' ')

                mk = len(src_sent)
                lk = len(tar_sent)

                src_sent = [''] + src_sent
                tar_sent = ['NULL'] + tar_sent

                for i in xrange(1, mk + 1):
                    for j in xrange(0, lk + 1):
                        fik = src_sent[i]
                        ejk = tar_sent[j]

                        denominator = sum([t(fik, _ejk)
                                           for _ejk in tar_sent])
                        delta = t(fik, ejk) / denominator

                        c[ejk, fik] = c.get((ejk, fik), 0.0) + delta
                        c[ejk] = c.get(ejk, 0.0) + delta
                        c[j, i, lk, mk] = c.get(
                            (j, i, lk, mk), 0.0) + delta
                        c[i, lk, mk] = c.get((i, lk, mk), 0.0) + delta

            for e in self.t:
                for f in self.t[e]:
                    self.t[e][f] = c[e, f] / c[e]

        # Save trained parameters to pickle file
        open(pickle_file, 'w').close()
        print('Save trained parameters to pickle file')
        with open(pickle_file, 'w') as f:
            pickle.dump(self.t, f)

        print('Done')

    def align(self, src_test_corpus, tar_test_corpus, out_alignment_file):
        print('Aligning')
        open(out_alignment_file, 'w').close()

        def t(f, e):
            self.t.setdefault(e, {})
            self.t[e].setdefault(f, 0)

            return self.t[e][f]

        with open(out_alignment_file, 'w') as output_file:
            with open(src_test_corpus, 'r') as source_file, open(tar_test_corpus, 'r') as target_file:
                sent_index = 0
                for src_sent, tar_sent in izip(source_file, target_file):
                    src_sent = src_sent.strip()
                    tar_sent = tar_sent.strip()

                    if not src_sent or not tar_sent:
                        continue

                    sent_index += 1
                    src_sent = [''] + src_sent.split(' ')
                    tar_sent = ['NULL'] + tar_sent.split(' ')

                    pairs = []
                    for i, fi in enumerate(src_sent):
                        if i == 0:
                            continue

                        max_prob = -1
                        max_j = -1

                        for j, ej in enumerate(tar_sent):
                            prob = t(fi, ej)
                            if prob > max_prob:
                                max_prob = prob
                                max_j = j

                        if max_j != 0:
                            pairs.append((max_j, i))

                    pairs = sorted(pairs, key=lambda element: (element[0], element[1]))

                    for pair in pairs:
                        output_file.write('{0} {1} {2}\n'.format(sent_index, pair[0], pair[1]))


def main(src_test_corpus, tar_test_corpus, out_alignment_file):
    source_corpus = './corpus.es'
    target_corpus = './corpus.en'
    pickle_file = './tfe-1.pickle'
    ibm_1 = IBM1(source_corpus, target_corpus, pickle_file)

    ibm_1.align(src_test_corpus, tar_test_corpus, out_alignment_file)

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 4:
        sys.exit(1)

    main(args[1], args[2], args[3])
