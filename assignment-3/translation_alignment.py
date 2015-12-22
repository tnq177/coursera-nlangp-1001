#! /usr/bin/python
from __future__ import print_function, division
import sys
import pickle
from os.path import isfile, exists
from itertools import izip
import timeit
import numpy


class IBM(object):

    def __init__(self, source_corpus, target_corpus, model_number):
        self.model_number = model_number

        has_pretrained = False
        if model_number == 1:
            has_pretrained = exists(
                './tfe-1.pickle') and isfile('./tfe-1.pickle')
        else:
            t_file, q_file = self._get_pickle_files(source_corpus)
            has_pretrained = exists(t_file) and isfile(
                t_file) and exists(q_file) and isfile(q_file)

        if has_pretrained:
            print('Loading pretrained model')
            if model_number == 1:
                with open('./tfe-1.pickle', 'r') as f:
                    self.t = pickle.load(f)
            else:
                t_file, q_file = self._get_pickle_files(source_corpus)

                with open(t_file, 'r') as tfe_file, open(q_file, 'r') as q_file:
                    self.t = pickle.load(tfe_file)
                    self.q = pickle.load(q_file)
        else:
            print('Training model {0}'.format(model_number))
            if model_number == 1:
                self.train_1(source_corpus, target_corpus)
            else:
                self.train_2(source_corpus, target_corpus)

    def _get_pickle_files(self, source_corpus):
        if source_corpus.endswith('es'):
            return './tfe-2.pickle', './qfe.pickle'
        else:
            return './tef-2.pickle', './qef.pickle'

    def _init_t(self, source_file, target_file):
        self.t = {}
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

    def train_1(self, source_corpus, target_corpus):
        f = open(source_corpus, 'r')
        source_file = f.readlines()
        f.close()
        f = open(target_corpus, 'r')
        target_file = f.readlines()
        f.close()

        self._init_t(source_file, target_file)

        def t(f, e):
            self.t.setdefault(e, {})
            self.t[e].setdefault(f, 0.0)

            return self.t[e][f]

        # Train with EM algorithm
        print('Training model 1')
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
        t_file_name = ''
        if source_corpus.endswith('.es'):
            t_file_name = './tfe-1.pickle'
        else:
            t_file_name = './tef-1.pickle'

        open(t_file_name, 'w').close()
        print('Save trained parameters to pickle file')
        with open(t_file_name, 'w') as f:
            pickle.dump(self.t, f)

        print('Done')

    def train_2(self, source_corpus, target_corpus):
        f = open(source_corpus, 'r')
        source_file = f.readlines()
        f.close()
        f = open(target_corpus, 'r')
        target_file = f.readlines()
        f.close()

        # Should use the trained parameters from model 1
        # In case it does not exist, train it
        t_file = './tfe-1.pickle' if source_corpus.endswith('.es') else './tef-1.pickle'
        if exists(t_file) and isfile(t_file):
            with open(t_file) as f:
                self.t = pickle.load(f)
        else:
            self.train_1(source_corpus, target_corpus)

        # Init self.q
        self.q = {}
        for src, tar in izip(source_file, target_file):
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
                    self.q[j, i, l, m] = 1.0 / (l + 1)

        def t(f, e):
            self.t.setdefault(e, {})
            self.t[e].setdefault(f, 0.0)

            return self.t[e][f]

        def q(j, i, l, m):
            self.q.setdefault((j, i, l, m), 0.0)
            return self.q[j, i, l, m]

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

            for e in self.t:
                for f in self.t[e]:
                    self.t[e][f] = c[e, f] / c[e]

            for key in self.q:
                self.q[key] = c[key] / c[key[1:]]

        # Save trained parameters to pickle file
        open(t_file, 'w').close()
        open(q_file, 'w').close()
        print('Save trained parameters to pickle files')
        with open(t_file, 'w') as t_pickle_file, open(q_file, 'w') as q_pickle_file:
            pickle.dump(self.t, t_pickle_file)
            pickle.dump(self.q, q_pickle_file)

        print('Done')

    def align(self, src_test_corpus, tar_test_corpus):
        print('Aligning')

        def t(f, e):
            self.t.setdefault(e, {})
            self.t[e].setdefault(f, 0)

            return self.t[e][f]

        def q(j, i, l, m):
            self.q.setdefault((j, i, l, m), 0.0)

            return self.q[j, i, l, m]

        result = {}
        with open(src_test_corpus, 'r') as source_file, open(tar_test_corpus, 'r') as target_file:
            sent_index = 0
            for src_sent, tar_sent in izip(source_file, target_file):
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
                        prob = t(fi, ej) if self.model_number == 1 else t(
                            fi, ej) * q(j, i, l, m)
                        if prob > max_prob:
                            max_prob = prob
                            max_j = j

                    if max_j != 0 and i != 0:
                        pairs.append((max_j, i))

                pairs = sorted(
                    pairs, key=lambda element: (element[0], element[1]))

                result[sent_index] = {
                    'pairs': pairs, 'foreign_length': m, 'tar_length': l}

        return result


def main(model_number, src_test_corpus, tar_test_corpus, out_alignment_file):
    if model_number not in ['1', '2', '3']:
        sys.exit(1)

    model_number = int(model_number)

    source_corpus = './corpus.es'
    target_corpus = './corpus.en'

    if model_number in [1, 2]:
        imb = IBM(source_corpus, target_corpus, model_number)

        result = imb.align(src_test_corpus, tar_test_corpus)
        open(out_alignment_file, 'w').close()
        with open(out_alignment_file, 'w') as out_file:
            for sent_index, values in result.iteritems():
                pairs = values['pairs']
                for pair in pairs:
                    out_file.write(
                        '{0} {1} {2}\n'.format(sent_index, pair[0], pair[1]))
    else:
        neighboring = [
            (-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def get_neighbors(e_index, f_index, e_length, f_length):
            neighbors = []
            for neighbor in neighboring:
                _e_index = e_index + neighbor[0]
                _f_index = f_index + neighbor[1]

                if 1 <= _e_index <= e_length and 1 <= _f_index <= f_length:
                    neighbors.append((_e_index, _f_index))

            return neighbors

        print('Init ibm_fe')
        imb_fe = IBM(source_corpus, target_corpus, model_number)
        print('Init ibm_ef')
        imb_ef = IBM(target_corpus, source_corpus, model_number)
        print('Align FE')
        result_fe = imb_fe.align(src_test_corpus, tar_test_corpus)
        print('Align EF')
        result_ef = imb_ef.align(tar_test_corpus, src_test_corpus)

        open(out_alignment_file, 'w').close()
        out_file = open(out_alignment_file, 'w')

        for sent_index in result_fe:
            pairs_fe = result_fe[sent_index]['pairs']
            pairs_ef = result_ef[sent_index]['pairs']

            foreign_length = result_fe[sent_index]['foreign_length']
            target_length = result_fe[sent_index]['tar_length']

            align_fe = numpy.zeros(
                (target_length + 1, foreign_length + 1), dtype=numpy.bool)
            align_ef = numpy.zeros(
                (target_length + 1, foreign_length + 1), dtype=numpy.bool)

            for pair in pairs_fe:
                align_fe[pair] = True

            for pair in pairs_ef:
                align_ef[pair[::-1]] = True

            alignment = align_fe * align_ef
            union = align_fe + align_ef

            # http://www.statmt.org/moses/?n=FactoredTraining.AlignWords
            # Grow diag
            while True:
                has_new_point = False
                for e in xrange(1, target_length + 1):
                    for f in xrange(1, foreign_length + 1):
                        if alignment[e, f]:
                            neighbors = get_neighbors(
                                e, f, target_length, foreign_length)
                            for neighbor in neighbors:
                                if (not numpy.any(alignment[neighbor[0]]) or not numpy.any(alignment[:, neighbor[1]])) and union[neighbor]:
                                    alignment[neighbor] = True
                                    has_new_point = True

                if not has_new_point:
                    break

            # Final
            for e in xrange(1, target_length + 1):
                for f in xrange(1, foreign_length + 1):
                    if (not numpy.any(alignment[e]) or not numpy.any(alignment[:, f])) and union[e, f]:
                        alignment[e, f] = True

            for e in xrange(1, target_length + 1):
                for f in xrange(1, foreign_length + 1):
                    if alignment[e, f]:
                        out_file.write(
                            '{0} {1} {2}\n'.format(sent_index, e, f))

        out_file.close()


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 5:
        sys.exit(1)

    main(args[1], args[2], args[3], args[4])
