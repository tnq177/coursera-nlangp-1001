#! /usr/bin/python
from __future__ import print_function, division
import sys
import cPickle as pickle
from os.path import isfile, exists
from itertools import izip
import timeit
import numpy
from ibm import IBM1, IBM2


if __name__ == '__main__':
    source_corpus = './corpus.es'
    target_corpus = './corpus.en'

    # Part 1
    # Dev
    source_test_corpus = './dev.es'
    target_test_corpus = './dev.en'

    ibm1 = IBM1(source_corpus, target_corpus)
    ibm1.train()
    result, sent_count = ibm1.align(source_test_corpus, target_test_corpus)

    open('./alignment_dev.p1.out', 'w').close()
    with open('./alignment_dev.p1.out', 'w') as f:
        for sent_index in xrange(1, sent_count + 1):
            pairs = result[sent_index]['pairs']
            for pair in pairs:
                f.write('{0} {1} {2}\n'.format(sent_index, pair[0], pair[1]))

    # Test
    source_test_corpus = './test.es'
    target_test_corpus = './test.en'

    result, sent_count = ibm1.align(source_test_corpus, target_test_corpus)

    open('./alignment_test.p1.out', 'w').close()
    with open('./alignment_test.p1.out', 'w') as f:
        for sent_index in xrange(1, sent_count + 1):
            pairs = result[sent_index]['pairs']
            for pair in pairs:
                f.write('{0} {1} {2}\n'.format(sent_index, pair[0], pair[1]))

    # Part 2
    # Train reverse mode of model 1
    ibm1.train(reverse=True)

    # Train model 2
    ibm2 = IBM2(source_corpus, target_corpus)
    ibm2.train()

    # Dev
    source_test_corpus = './dev.es'
    target_test_corpus = './dev.en'
    result, sent_count = ibm2.align(source_test_corpus, target_test_corpus)

    open('./alignment_dev.p2.out', 'w').close()
    with open('./alignment_dev.p2.out', 'w') as f:
        for sent_index in xrange(1, sent_count + 1):
            pairs = result[sent_index]['pairs']
            for pair in pairs:
                f.write('{0} {1} {2}\n'.format(sent_index, pair[0], pair[1]))

    # Test
    source_test_corpus = './test.es'
    target_test_corpus = './test.en'
    result, sent_count = ibm2.align(source_test_corpus, target_test_corpus)

    open('./alignment_test.p2.out', 'w').close()
    with open('./alignment_test.p2.out', 'w') as f:
        for sent_index in xrange(1, sent_count + 1):
            pairs = result[sent_index]['pairs']
            for pair in pairs:
                f.write('{0} {1} {2}\n'.format(sent_index, pair[0], pair[1]))

    ibm2.train(reverse=True)

    # Part 3
    neighboring = [
        (-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def get_neighbors(e, f, e_length, f_length):
        neighbors = []
        for neighbor in neighboring:
            _e = e + neighbor[0]
            _f = f + neighbor[1]

            if 1 <= _e <= e_length and 1 <= _f <= f_length:
                neighbors.append((_e, _f))

        return neighbors

    def many_to_many_align(result_fe, result_ef, output_file):
        open(output_file, 'w').close()
        with open(output_file, 'w') as out_file:
            for sent_index in xrange(1, sent_count + 1):
                pairs_fe = result_fe[sent_index]['pairs']
                pairs_ef = result_ef[sent_index]['pairs']

                f_length = result_fe[sent_index]['source_length']
                e_length = result_fe[sent_index]['target_length']

                align_fe = numpy.zeros(
                    (e_length + 1, f_length + 1), dtype=numpy.bool)
                align_ef = numpy.zeros(
                    (e_length + 1, f_length + 1), dtype=numpy.bool)

                for pair in pairs_fe:
                    align_fe[pair] = True
                for pair in pairs_ef:
                    align_ef[pair[::-1]] = True

                alignment = align_fe * align_ef
                union = align_fe + align_ef

                while True:
                    has_new_point = False

                    for e in xrange(1, e_length + 1):
                        for f in xrange(1, f_length + 1):
                            if alignment[e, f]:
                                neighbors = get_neighbors(
                                    e, f, e_length, f_length)

                                for neighbor in neighbors:
                                    if (not numpy.any(alignment[neighbor[0]]) or not numpy.any(alignment[:, neighbor[1]])) and union[neighbor]:
                                        alignment[neighbor] = True
                                        has_new_point = True

                    if not has_new_point:
                        break

                # Ok, got this from http://www.statmt.org/moses/?n=FactoredTraining.AlignWords
                # But just intersection gives better result 
                # for e in xrange(1, e_length + 1):
                #     for f in xrange(1, f_length + 1):
                #         if (not numpy.any(alignment[e]) or not numpy.any(alignment[:, f])) and align_ef[e, f]:
                #             alignment[e, f] = True

                # for e in xrange(1, e_length + 1):
                #     for f in xrange(1, f_length + 1):
                #         if (not numpy.any(alignment[e]) or not numpy.any(alignment[:, f])) and align_fe[e, f]:
                #             alignment[e, f] = True

                for e in xrange(1, e_length + 1):
                    for f in xrange(1, f_length + 1):
                        if alignment[e, f]:
                            out_file.write(
                                '{0} {1} {2}\n'.format(sent_index, e, f))

    # Dev
    source_test_corpus = './dev.es'
    target_test_corpus = './dev.en'
    result_fe, sent_count = ibm2.align(source_test_corpus, target_test_corpus)
    result_ef, sent_count = ibm2.align(
        target_test_corpus, source_test_corpus, reverse=True)
    many_to_many_align(result_fe, result_ef, './alignment_dev.p3.out')

    # Test
    source_test_corpus = './test.es'
    target_test_corpus = './test.en'
    result_fe, sent_count = ibm2.align(source_test_corpus, target_test_corpus)
    result_ef, sent_count = ibm2.align(
        target_test_corpus, source_test_corpus, reverse=True)
    many_to_many_align(result_fe, result_ef, './alignment_test.p3.out')
