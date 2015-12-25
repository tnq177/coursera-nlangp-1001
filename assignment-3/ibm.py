from __future__ import print_function, division
import sys
import pickle
from os.path import isfile, exists
from itertools import izip
import timeit
import numpy
import pdb

# IBM Model 1
class IBM1(object):

    def __init__(self, source_corpus, target_corpus):
        self.source_corpus = source_corpus
        self.target_corpus = target_corpus

        self.pickle_file = './tfe-1.pickle'
        self.reverse_pickle_file = './tef-1.pickle'

        self.t = {'forward': {}, 'reverse': {}}

    def _init_t(self, source_sents, target_sents, mode):
        self.t[mode] = {}

        # Calculate n(e)
        for src, tar in izip(source_sents, target_sents):
            src_sent = src.strip()
            tar_sent = tar.strip()

            if not src_sent or not tar_sent:
                continue

            src_sent = src_sent.split(' ')
            tar_sent = tar_sent.split(' ')

            src_sent_set = set(src_sent)
            tar_sent_set = set(['NULL'] + tar_sent)

            for e in tar_sent_set:
                self.t[mode].setdefault(e, {})
                for f in src_sent_set:
                    self.t[mode][e].setdefault(f, 0.0)

        for e in self.t[mode]:
            value = 1.0 / len(self.t[mode][e].keys())
            for key in self.t[mode][e]:
                self.t[mode][e][key] = value

    def train(self, reverse=False):
        if reverse:
            mode = 'reverse'
            pickle_file = self.reverse_pickle_file
            source_corpus = self.target_corpus
            target_corpus = self.source_corpus
        else:
            mode = 'forward'
            pickle_file = self.pickle_file
            source_corpus = self.source_corpus
            target_corpus = self.target_corpus

        if exists(pickle_file) and isfile(pickle_file):
            print('Loading trained params from file')
            print('Model 1, mode: {0}'.format(mode))
            with open(pickle_file, 'r') as f:
                self.t[mode] = pickle.load(f)
        else:
            with open(source_corpus, 'r') as s_f, open(target_corpus, 'r') as t_f:
                source_sents = s_f.readlines()
                target_sents = t_f.readlines()

            # Init t to 1/n(e)
            self._init_t(source_sents, target_sents, mode)

            # Train
            def t(f, e):
                self.t[mode].setdefault(e, {})
                self.t[mode][e].setdefault(f, 0.0)

                return self.t[mode][e][f]

            print('Training model 1, mode: {0}'.format(mode))
            S = 5
            for s in xrange(0, S):
                print('Iteration: {0}'.format(s + 1))
                c = {}

                for src, tar in izip(source_sents, target_sents):
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

                for e in self.t[mode]:
                    for f in self.t[mode][e]:
                        self.t[mode][e][f] = c[e, f] / c[e]

            # Write trained parameters to file for future use
            print('Writing trained params to file: {0}'.format(pickle_file))
            print('Model 1, mode: {0}'.format(mode))
            open(pickle_file, 'w').close()
            with open(pickle_file, 'w') as f:
                pickle.dump(self.t[mode], f)

    def align(self, source_test_corpus, target_test_corpus, reverse=False):
        print('Aligning')

        mode = 'reverse' if reverse else 'forward'

        def t(f, e):
            self.t[mode].setdefault(e, {})
            self.t[mode][e].setdefault(f, 0.0)

            return self.t[mode][e][f]

        result = {}
        with open(source_test_corpus, 'r') as s_f, open(target_test_corpus, 'r') as t_f:
            sent_index = 0
            for src_sent, tar_sent in izip(s_f, t_f):
                src_sent = src_sent.strip()
                tar_sent = tar_sent.strip()

                if not src_sent or not tar_sent:
                    continue

                sent_index += 1
                src_sent = src_sent.split(' ')
                tar_sent = tar_sent.split(' ')

                l = len(tar_sent)
                m = len(src_sent)

                src_sent = [''] + src_sent
                tar_sent = ['NULL'] + tar_sent

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

                    pairs.append((max_j, i))

                pairs = sorted(
                    pairs, key=lambda element: (element[0], element[1]))

                result[sent_index] = {
                    'pairs': pairs, 'source_length': m, 'target_length': l}

        return result, sent_index

# IBM Model 2
class IBM2(object):

    def __init__(self, source_corpus, target_corpus):
        self.source_corpus = source_corpus
        self.target_corpus = target_corpus

        self.t_pickle_file = './tfe-2.pickle'
        self.q_pickle_file = './qfe.pickle'
        self.t_reverse_pickle_file = './tef-2.pickle'
        self.q_reverse_pickle_file = './qef.pickle'

        self.model_1_t_pickle_file = {'forward': './tfe-1.pickle', 'reverse': './tef-1.pickle'}

        if not exists(self.model_1_t_pickle_file['forward']) or not exists(self.model_1_t_pickle_file['reverse']):
            print('Please train model 1 in both forward and reverse mode first')
            sys.exit(0)

        self.t = {'forward': {}, 'reverse': {}}
        self.q = {'forward': {}, 'reverse': {}}

    def _init_q(self, source_sents, target_sents, mode):
        print('Init q, mode: {0}'.format(mode))
        self.q[mode] = {}

        # Init self.q
        for src, tar in izip(source_sents, target_sents):
            src_sent = src.strip()
            tar_sent = tar.strip()

            if not src_sent or not tar_sent:
                continue

            src_sent = src_sent.split(' ')
            tar_sent = tar_sent.split(' ')
            m = len(src_sent)
            l = len(tar_sent)

            for i in xrange(1, m + 1):
                for j in xrange(0, l + 1):
                    self.q[mode][j, i, l, m] = 1.0 / (l + 1)

    def train(self, reverse=False):
        if reverse:
            mode = 'reverse'
            t_pickle_file = self.t_reverse_pickle_file
            q_pickle_file = self.q_reverse_pickle_file
            source_corpus = self.target_corpus
            target_corpus = self.source_corpus
        else:
            mode = 'forward'
            t_pickle_file = self.t_pickle_file
            q_pickle_file = self.q_pickle_file
            source_corpus = self.source_corpus
            target_corpus = self.target_corpus

        already_trained = exists(q_pickle_file) and isfile(q_pickle_file) and exists(t_pickle_file) and isfile(t_pickle_file)

        if already_trained:
            print('Loading trained params from file')
            print('Model 2, mode: {0}'.format(mode))
            with open(t_pickle_file, 'r') as t_f, open(q_pickle_file, 'r') as q_f:
                self.t[mode] = pickle.load(t_f)
                self.q[mode] = pickle.load(q_f)
        else:
            with open(source_corpus, 'r') as s_f, open(target_corpus, 'r') as t_f:
                source_sents = s_f.readlines()
                target_sents = t_f.readlines()

            # Init t to model 1 trained params
            print('Load model 1 trained params, mode: {0}'.format(mode))
            with open(self.model_1_t_pickle_file[mode], 'r') as f:
                self.t[mode] = pickle.load(f)

            self._init_q(source_sents, target_sents, mode)

            # Train
            def t(f, e):
                self.t[mode].setdefault(e, {})
                self.t[mode][e].setdefault(f, 0.0)

                return self.t[mode][e][f]

            def q(j, i, l, m):
                self.q[mode].setdefault((j, i, l, m), 0.0)

                return self.q[mode][j, i, l, m]

            print('Training model 2, mode: {0}'.format(mode))
            S = 5
            for s in xrange(0, S):
                print('Iteration: {0}'.format(s + 1))
                c = {}
                for src, tar in izip(source_sents, target_sents):
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

                            denominator = 0.0
                            for _j in xrange(0, lk + 1):
                                _ejk = tar_sent[_j]
                                denominator += q(_j, i, lk, mk) * t(fik, _ejk)

                            delta = t(fik, ejk) * q(j, i, lk, mk) / denominator

                            c[ejk, fik] = c.get((ejk, fik), 0.0) + delta
                            c[ejk] = c.get(ejk, 0.0) + delta
                            c[j, i, lk, mk] = c.get(
                                (j, i, lk, mk), 0.0) + delta
                            c[i, lk, mk] = c.get((i, lk, mk), 0.0) + delta

                for e in self.t[mode]:
                    for f in self.t[mode][e]:
                        self.t[mode][e][f] = c[e, f] / c[e]

                for key in self.q[mode]:
                    self.q[mode][key] = c[key] / c[key[1:]]

            # Write trained parameters to file for future use
            print('Writing trained params to file')
            print('Model 2, mode: {0}'.format(mode))
            open(t_pickle_file, 'w').close()
            open(q_pickle_file, 'w').close()
            with open(t_pickle_file, 'w') as t_f, open(q_pickle_file, 'w') as q_f:
                pickle.dump(self.t[mode], t_f)
                pickle.dump(self.q[mode], q_f)

    def align(self, source_test_corpus, target_test_corpus, reverse=False):
        print('Aligning')

        mode = 'reverse' if reverse else 'forward'

        def t(f, e):
            self.t[mode].setdefault(e, {})
            self.t[mode][e].setdefault(f, 0.0)

            return self.t[mode][e][f]

        def q(j, i, l, m):
            self.q[mode].setdefault((j, i, l, m), 0.0)

            return self.q[mode][j, i, l, m]

        result = {}
        with open(source_test_corpus, 'r') as s_f, open(target_test_corpus, 'r') as t_f:
            sent_index = 0
            for src_sent, tar_sent in izip(s_f, t_f):
                src_sent = src_sent.strip()
                tar_sent = tar_sent.strip()

                if not src_sent or not tar_sent:
                    continue

                sent_index += 1
                src_sent = src_sent.split(' ')
                tar_sent = tar_sent.split(' ')

                l = len(tar_sent)
                m = len(src_sent)

                src_sent = [''] + src_sent
                tar_sent = ['NULL'] + tar_sent

                pairs = []
                for i, fi in enumerate(src_sent):
                    if i == 0:
                        continue

                    max_prob = -1
                    max_j = -1

                    for j, ej in enumerate(tar_sent):
                        prob = t(fi, ej) * q(j, i, l, m)
                        if prob > max_prob:
                            max_prob = prob
                            max_j = j

                    if max_j != 0:
                        pairs.append((max_j, i))

                pairs = sorted(
                    pairs, key=lambda element: (element[0], element[1]))

                result[sent_index] = {
                    'pairs': pairs, 'source_length': m, 'target_length': l}

        return result, sent_index
