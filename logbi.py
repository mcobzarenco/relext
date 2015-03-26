#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import cPickle as pickle
import sys
import time
import unittest
from collections import defaultdict, namedtuple
from itertools import count, cycle, islice, izip
from random import randint, shuffle

import numpy as np
import pandas as pd
from numpy import array, concatenate, dot, eye, inf, isnan, log, log1p, outer, \
    r_, c_, pi, mean, cov, einsum, exp, zeros, zeros_like
from numpy.random import rand, randn
from numpy.linalg import cholesky, det, inv
from scipy import optimize
from scipy.optimize import check_grad, minimizee


DEFAULT_MAXNUMLINESEARCH = 150

sp_minimize = optimize.minimize
sigmoid = lambda u: 1.0 / (1.0 + exp(-u))
# sigmoid = lambda u: (1.0 + u / np.sqrt(1 + u * u)) / 2

LogBilinearParams = namedtuple('LogBilinearParams', ['e', 'R'])
SparseGradient = namedtuple('SparseGradient', ['de', 'dR'])

# Codes the assertion R_k(e_i, e_j) = r
RelationInstance = namedtuple('RelationInstance', ['i', 'j', 'k', 'r']);
Dataset = namedtuple(
    'Dataset', ['ent_dict', 'rel_dict', 'train_rels', 'valid_rels'])

TrainTestSplit = namedtuple('TrainTestSplit', ['train', 'test'])
FitResult = namedtuple('FitResult', ['ll_train', 'll_valid'])


def split_train_test(xs, test_fraction):
    assert 0 <= test_fraction <= 1
    split_point = int(len(xs) * (1 - test_fraction))
    return TrainTestSplit(xs[:split_point], xs[split_point:])


class RelationParams(object):
    def __init__(self, U, V):
        assert U.shape == V.T.shape
        self.U = U
        self.V = V

    def __getitem__(self, i):
        if i == 0:
            return self.U
        elif i == 1:
            return self.V
        else:
            raise IndexError()

    def __repr__(self):
        return "RelationParams({}, {})".format(self.U, self.V)


def params_to_vec(params):
    NE = len(params.e)
    NR = len(params.R)
    D, rho = params.R[0].U.shape

    theta = np.empty(D * NE + 2 * D * rho * NR)
    for i in xrange(NE):
        theta[D * i:D * (i + 1)] = params.e[i].reshape((-1,))

    copy_index = D*NE;
    for k in xrange(NR):
        theta[copy_index:copy_index + D * rho] = params.R[k].U.reshape((-1,))
        copy_index += D * rho;
        theta[copy_index:copy_index + D * rho] = params.R[k].V.reshape((-1,))
        copy_index += D * rho;
    assert copy_index == len(theta)
    return theta


def vec_to_params(D, NE, NR, rho, theta):
    e, R = [], []
    for i in xrange(NE):
        e.append(theta[D * i:D * (i + 1)].reshape((D, 1)))
    copy_index = D * NE;
    for k in xrange(NR):
        U = theta[copy_index:copy_index + D * rho].reshape(D, rho)
        copy_index += D * rho
        V = theta[copy_index:copy_index + D * rho].reshape(rho, D)
        copy_index += D * rho
        R.append(RelationParams(U, V))
    return LogBilinearParams(e, R)


def numerical_grad(f, x0):
    EPS = 1e-12
    grad = np.empty(len(x0))
    x_minus = x0
    x_plus = np.array(x0)
    for i in xrange(len(x0)):
        x_minus[i] -= EPS/2
        x_plus[i] += EPS/2
        grad[i] = (f(x_plus) - f(x_minus)) / EPS
        x_minus[i] += EPS/2
        x_plus[i] -= EPS/2
    return grad


class FeatureDictionary(object):
    def __init__(self):
        self._map = {}
        self._imap = {}
        self._dirty_inv = False
        self._index = 0

    def encode(self, feature):
        if feature not in self._map:
            self._dirty_inv = True
            self._map[feature] = self._index
            index = self._index
            self._index += 1
            return index
        return self._map[feature]

    def decode(self, index):
        if self._dirty_inv:
            self._imap = {v: k for k, v in self._map.iteritems()}
        return self._imap[index]


class TestFeatureDictionary(unittest.TestCase):
    def setUp(self):
        self.fdict = FeatureDictionary()

    def test_unique(self):
        fdict = self.fdict
        self.assertEqual(0, fdict.encode("a"))
        self.assertEqual(1, fdict.encode("b"))
        self.assertEqual(0, fdict.encode("a"))
        self.assertEqual(1, fdict.encode("b"))
        self.assertEqual(2, fdict.encode("c"))
        self.assertEqual(3, fdict.encode("a "))
        self.assertEqual(4, fdict.encode(" a "))


def csv_to_rels(path):
    rels = pd.read_csv(path, sep='\t', header=None,
                       names=['rel', 'lhs', 'rhs'])
    ent_dict = FeatureDictionary()
    rel_dict = FeatureDictionary()
    train_rels, valid_rels = [], []
    i = 0;
    for (ix, row) in rels.iterrows():
        if i % 2 == 0: #row.train == 'Test':
            valid_rels.append(RelationInstance(
                ent_dict.encode(row.lhs), ent_dict.encode(row.rhs),
                rel_dict.encode(row.rel), 1))
        else:
            train_rels.append(RelationInstance(
                ent_dict.encode(row.lhs), ent_dict.encode(row.rhs),
                rel_dict.encode(row.rel), 1))
        i += 1
    return Dataset(ent_dict, rel_dict, train_rels, valid_rels)


def l2_norm2(x):
    return np.sum(x.flatten() ** 2)


def compute_accuracy(predictions, ground, threshold=0.5):
    predictions = (predictions > threshold) * 1
    acc = map(lambda x: x[0].r == x[1] , zip(ground, predictions))
    return np.sum(acc) / float(len(acc))


class LogBilinear(object):
    def __init__(self, D, NE, NR, rho, sigma=1):
        e, R = [], []
        init_vals = lambda m, n: sigma * randn(m, n)
        for _ in xrange(NE):
            e.append(init_vals(D, 1))
        for _ in xrange(NR):
            U = sigma * init_vals(D, rho)
            V = sigma * init_vals(rho, D)
            R.append(RelationParams(U, V))
        self.params = LogBilinearParams(e, R)

    def fit(self, dataset, gamma=0):
        cls = self.__class__
        D, rho = self.params.R[0].U.shape
        NE, NR = len(self.params.e), len(self.params.R)
        valid_len = len(dataset.valid_rels)
        n_ones = sum(map(lambda x: x.r == 1, dataset.valid_rels)) / float(valid_len)
        print(("Training model with {} datapoints ({} held out for validation " +
              "with {}% =1)").format(len(dataset.train_rels), valid_len, n_ones))

        def obj(rels, x):
            params = vec_to_params(D, NE, NR, rho, x)
            train_nll = -cls._loglik(params.e, params.R, gamma, rels)
            train_acc = compute_accuracy(cls._p(params.e, params.R, rels), rels)
            # valid_nll = -cls._loglik(params.e, params.R, gamma, dataset.valid_rels)
            # valid_acc = compute_accuracy(cls._p(params.e, params.R, dataset.valid_rels),
            #                              dataset.valid_rels)
            # print("valid_nll = {} / valid_acc = {} / train_nll = {} / "
            #       "train_acc = {}".format(
            #           valid_nll / len(dataset.valid_rels), valid_acc,
            #           train_nll / len(rels), train_acc))
            return train_nll

        def grad(rels, x, compute_du, compute_dv):
            params = vec_to_params(D, NE, NR, rho, x)
            train_nll = -cls._loglik(params.e, params.R, gamma, rels)
            train_acc = compute_accuracy(cls._p(params.e, params.R, batch), rels)
            print("Objective: {} / Accuracy: {}".format(train_nll, train_acc))
            return -numerical_grad(lambda y: obj(rels, y), x)
            # return -params_to_vec(
            #     cls._dloglik(params.e, params.R, gamma, rels, compute_du, compute_dv))

        theta = params_to_vec(self.params)
        print("Number of model parameters: {}".format(len(theta)))

        theta_update = zeros(len(theta))
        alpha, mu = 0.001, 0.2
        N_ITER = 100
        batch_size = 5000
        train_rels = dataset.train_rels
        shuffle(train_rels)
        batches = cycle(train_rels)
        # for i in xrange(N_ITER):
        #     batch = list(islice(batches, batch_size))
        #     print("Loaded batch {} ({})".format(i, len(batch)))
        #     probe = theta + theta_update
        #     theta_update = mu * theta_update - alpha * grad(batch, probe, True, False)
        #     theta += theta_update
        #     probe = theta + theta_update
        #     theta_update = mu * theta_update - alpha * grad(batch, probe, False, True)
        #     theta += theta_update
        #     if (i + 1) % 50 == 0:
        #         print("{}".format(obj(batch, theta)))

        for i in xrange(N_ITER):
            batch = list(islice(batches, batch_size))
            o = lambda x: obj(batch, x)
            g = lambda x: grad(batch, x, True, True)
            print("Loaded batch {} ({})".format(i, len(batch)))
            result = minimize(o, theta, method='Newton-CG', jac=g,
                              options={'disp': True, 'maxiter': 10})
                                       # 'ftol': 0, 'gtol': 1e-8})
            assert (theta == result.x).all() == False
            theta = result.x
        theta_star = theta

        params = vec_to_params(D, NE, NR, rho, theta_star)
        preds = cls._p(params.e, params.R, dataset.valid_rels) > 0.5
        acc = map(lambda x: x[0].r == x[1] , zip(dataset.valid_rels, preds))
        acc = np.sum(acc) / len(acc)
        print("At exit: Validation likelihood: {} accuracy: {}".format(
            -cls._loglik(params.e, params.R, gamma, dataset.valid_rels), acc))
        self.params = params
        return FitResult(0, 0)

    def eval_one(self, entity1, entity2, relation):
        assert(0 <= entity1 < len(self.params.e) and
               0 <= entity2 < len(self.params.e) and
               0 <= relation < len(self.params.R))
        return self.__class__._p(self.params.e, self.params.R,
                                 [RelationInstance(entity1, entity2, relation, False)])

    def eval_rels(self, rels):
        return self.__class__._p(self.params.e, self.params.R, rels)

    @staticmethod
    def _p(e, R, rels):
        p_one = lambda rel: sigmoid(
            np.einsum('ij,il,lk,kj->', # e'_i * U_k * V_k * e_j
                      e[rel.i], R[rel.k].U, R[rel.k].V, e[rel.j]).flatten()[0])
        return array(map(p_one, rels))

    def loglik(self, gamma, rels):
        return self.__class__._loglik(
            self.params.e, self.params.R, gamma, rels)

    @staticmethod
    def _loglik(e, R, gamma, rels):
        D, rho = R[0].U.shape
        NE, NR =  len(e), len(R)
        loglik = 0
        for rel in rels:
            corr = np.einsum('ij,il,lk,kj->', # e'_i * U_k * V_k * e_j
                             e[rel.i], R[rel.k].U, R[rel.k].V, e[rel.j])
            # scorr = sigmoid(corr)
            # assert rel.r == 1 or rel.r == 0
            # if (scorr == 1.0 and rel.r == 1) or (scorr == 0.0 and rel.r == 0):
            #     continue
            # loglik += (1 - rel.r) * log(1 - scorr) + rel.r * log(scorr)
            loglik += (rel.r - 1) * (corr + log1p(exp(-corr))) - \
                      rel.r * log1p(exp(-corr))
            if np.isnan(loglik):
                import pdb
                pdb.set_trace()
            assert not np.isnan(loglik)
        assert loglik <= 0
        # loglik /= len(rels)
        if gamma > 0:
            reg = np.sum(map(lambda r: l2_norm2(r.U) + l2_norm2(r.V), R)) + \
                  np.sum(map(l2_norm2, e))
            loglik -= reg * gamma / 2
            #loglik -= log(2 * pi * gamma) * (D*NE + 2*D*rho*NR) / 2
        assert isinstance(loglik, np.float)
        return loglik

    @staticmethod
    def _dloglik(e, R, gamma, rels, compute_du=True, compute_dv=True):
        D, NE, NR = e[0].shape[0], len(e), len(R)
        N = len(rels)
        de, dR = [], []
        for i in xrange(NE):
            de.append(-e[i] * gamma)
        for k in xrange(NR):
            dU = -R[k].U * gamma if compute_du else zeros_like(R[k].U)
            dV = -R[k].V * gamma if compute_dv else zeros_like(R[k].V)
            dR.append(RelationParams(dU, dV))

        for rel in rels:
            ei, Uk, Vk, ej = e[rel.i], R[rel.k].U, R[rel.k].V, e[rel.j]
            scorr = sigmoid(einsum('ij,il,lk,kj->', ei, Uk, Vk, ej))
            premult = (rel.r - 1) * scorr + rel.r * (1 - scorr)
            de[rel.i] += premult * einsum('ik,kl,lj->ij', Uk, Vk, ej)
            de[rel.j] += premult * einsum('ji,kj,kl->il', Vk, Uk, ei)
            if compute_du:
                dR[rel.k].U += premult * einsum('ij,kj,lk->il', ei, ej, Vk)
            if compute_dv:
                dR[rel.k].V += premult * einsum('ji,jk,lk->il', Uk, ei, ej)
        return LogBilinearParams(de, dR)


class TestLogBilinear(unittest.TestCase):
    def test_init(self):
        lb = LogBilinear(4, 10, 11, 3, 1)
        self.assertEqual(10, len(lb.params.e))
        self.assertEqual(11, len(lb.params.R))
        self.assertTrue(all(map(lambda e: len(e) == 4, lb.params.e)))
        for (U, V) in lb.params.R:
            self.assertEqual((4, 3), U.shape)
            self.assertEqual((3, 4), V.shape)
        print("lb={}".format(lb.eval_one(1,1,1)))

    def test_params_to_vec(self):
        D, NE, NR, rho = 3, 9, 14, 2
        lb = LogBilinear(D, NE, NR, rho)

        as_vec = params_to_vec(lb.params)
        self.assertEqual(lb.params.e[0][0, 0], as_vec[0])
        self.assertEqual(lb.params.e[0][1, 0], as_vec[1])
        self.assertEqual(lb.params.e[0][2, 0], as_vec[2])
        self.assertEqual(lb.params.e[1][0, 0], as_vec[3])
        self.assertEqual(lb.params.e[1][1, 0], as_vec[4])
        self.assertEqual(lb.params.R[1].V[0, 1], as_vec[D*NE + 3 * D*rho + 1])

        to_vec_and_back = vec_to_params(D, NE, NR, rho, as_vec)
        for i in xrange(NE):
            self.assertEqual(lb.params.e[i].shape, to_vec_and_back.e[i].shape)
            self.assertTrue((lb.params.e[i] == to_vec_and_back.e[i]).all())
        for k in xrange(NR):
            self.assertEqual(lb.params.R[k].U.shape, to_vec_and_back.R[k].U.shape)
            self.assertEqual(lb.params.R[k].V.shape, to_vec_and_back.R[k].V.shape)
            self.assertTrue((lb.params.R[k].U == to_vec_and_back.R[k].U).all())
            self.assertTrue((lb.params.R[k].V == to_vec_and_back.R[k].V).all())

    def test_check_gradient(self):
        D, NE, NR, rho = 3, 3, 3, 2
        gamma = 0.2
        lb = LogBilinear(D, NE, NR, rho, 1)
        rels = [RelationInstance(0, 1, 0, 1)]# , RelationInstance(0, 2, 1, 0)]

        def obj(x):
            params = vec_to_params(D, NE, NR, rho, x)
            return lb._loglik(params.e, params.R, gamma, rels)

        def grad_both(x):
            params = vec_to_params(D, NE, NR, rho, x)
            return params_to_vec(lb._dloglik(params.e, params.R, gamma, rels))

        def grad_dU(x):
            params = vec_to_params(D, NE, NR, rho, x)
            return params_to_vec(lb._dloglik(
                params.e, params.R, gamma, rels, compute_du=True, compute_dv=False))

        def grad_dV(x):
            params = vec_to_params(D, NE, NR, rho, x)
            return params_to_vec(lb._dloglik(
                params.e, params.R, gamma, rels, compute_du=False, compute_dv=True))

        x0 = params_to_vec(lb.params)
        for i in xrange(1):
            dnum = vec_to_params(D, NE, NR, rho, numerical_grad(obj, x0))
            dexact = vec_to_params(D, NE, NR, rho, grad_both(x0))
            dexact_dU = vec_to_params(D, NE, NR, rho, grad_dU(x0))
            dexact_dV = vec_to_params(D, NE, NR, rho, grad_dV(x0))
            # print("\ndnum:\n{}".format(dnum))
            # print("\ndexact:\n{}".format(dexact))
            # print("\ndexact_dU:\n{}".format(dexact_dU))
            # print("\ndexact_dV:\n{}".format(dexact_dV))
            # print(params_to_vec(dnum) / params_to_vec(dexact))
            self.assertLess(
                max((params_to_vec(dnum) - params_to_vec(dexact)) ** 2), 1e-5)
            for r_num, r_dU, r_dV in zip(dnum.R, dexact_dU.R, dexact_dV.R):
                self.assertLess(
                    max((r_num.U.flatten() - r_dU.U.flatten()) ** 2), 1e-5)
                self.assertTrue((r_dU.V == 0).all())

                self.assertLess(
                    max((r_num.V.flatten() - r_dV.V.flatten()) ** 2), 1e-5)
                self.assertTrue((r_dV.U == 0).all())

            x0 += randn(len(x0)) / 100

    def test_split_train_test(self):
        rng = range(1, 11)
        self.assertEqual(TrainTestSplit([1,2,3,4,5,6,7], [8,9,10]),
                         split_train_test(rng, 0.3))


class BprBilinear(object):
    def __init__(self, D, NE, NR, rho, sigma=1):
        e, R = [], []
        init_vals = lambda m, n: 2 * sigma * (rand(m, n) - 0.5)
        for _ in xrange(NE):
            e.append(init_vals(D, 1))
        for _ in xrange(NR):
            U = sigma * init_vals(D, rho)
            V = sigma * init_vals(rho, D)
            R.append(RelationParams(U, V))
        self.params = LogBilinearParams(e, R)

    def fit(self, dataset, gamma=0):
        cls = self.__class__
        D, rho = self.params.R[0].U.shape
        NE, NR = len(self.params.e), len(self.params.R)
        valid_len = len(dataset.valid_rels)
        n_ones = sum(map(lambda x: x.r == 1, dataset.valid_rels)) / float(valid_len)
        print(("Training model with {} datapoints ({} held out for validation " +
              "with {}% =1)").format(len(dataset.train_rels), valid_len, n_ones))

        def obj(rel_pairs, x):
            params = vec_to_params(D, NE, NR, rho, x)
            train_obj = -cls._obj(params.e, params.R, gamma, rel_pairs)
            # train_acc = compute_accuracy(cls._p(params.e, params.R, rels), rels)
            return train_obj

        def grad(rel_pairs, x, compute_du, compute_dv):
            params = vec_to_params(D, NE, NR, rho, x)
            train_obj = -cls._obj(params.e, params.R, gamma, rel_pairs)
            # train_acc = compute_accuracy(cls._p(params.e, params.R, batch), rels)
            print("Objective: {}".format(train_obj))
            return -params_to_vec(cls._dobj(
                params.e, params.R, gamma, rel_pairs, compute_du, compute_dv))

        theta = params_to_vec(self.params)
        print("Number of model parameters: {}".format(len(theta)))

        theta_update = zeros(len(theta))
        alpha, mu = .1, 0.8
        N_ITER = 30
        batch_size = 3000
        train_rels = dataset.train_rels
        shuffle(train_rels)
        batches = cycle(train_rels)
        neg_rel = lambda k : RelationInstance(randint(0, NE - 1), randint(0, NE - 1), k, 0)
        # for i in xrange(N_ITER):
        #     batch = map(lambda x: (x, neg_rel(x.k)), islice(batches, batch_size))
        #     print("Loaded batch {} ({})".format(i, len(batch)))
        #     for j in [0, 1]:
        #         start = time.clock()
        #         probe = theta + theta_update
        #         theta_update = mu * theta_update - alpha * grad(batch, probe, True, False)
        #         theta += theta_update
        #         print("Update took {} seconds".format(time.clock() - start))

        #         start = time.clock()
        #         probe = theta + theta_update
        #         theta_update = mu * theta_update - alpha * grad(batch, probe, False, True)
        #         theta += theta_update
        #         print("Update took {} seconds".format(time.clock() - start))

        def apply_sparse_grad(alpha, p, sg):
            for i, ent in sg.de.iteritems():
                p.e[i] += alpha * ent
            for k, rel in sg.dR.iteritems():
                p.R[k].U += alpha * rel.U
                p.R[k].V += alpha * rel.V

        for i in xrange(N_ITER):
            batch = map(lambda x: (x, neg_rel(x.k)), islice(batches, batch_size))
            print("Loaded batch {} ({})".format(i, len(batch)))
            start = time.clock()
            sgrad = cls._dobj_sparse(
                self.params.e, self.params.R, gamma, batch, True, True)
            apply_sparse_grad(alpha, self.params, sgrad)

            # sgrad = cls._dobj_sparse(
            #     self.params.e, self.params.R, gamma, batch, False, True)
            # apply_sparse_grad(alpha, self.params, sgrad)

            # train_obj = -cls._obj(self.params.e, self.params.R, gamma, batch)
            # print("Objective: {}".format(train_obj))

            print("Update took {} seconds".format(time.clock() - start))

            train_obj = -cls._obj(self.params.e, self.params.R, gamma, batch)
            print("Objective: {}".format(train_obj))



        # for i in xrange(N_ITER):
        #     batch = map(lambda x: (x, neg_rel(x.k)), islice(batches, batch_size))
        #     o = lambda x: obj(batch, x)
        #     print("Loaded batch {} ({})".format(i, len(batch)))

        #     g = lambda x: grad(batch, x, True, True)
        #     result = minimize(o, theta, method='L-BFGS-B', jac=g,
        #                       options={'disp': False, 'maxfun': 3})
        #     # g = lambda x: grad(batch, x, False, True)
        #     # result = minimize(o, theta, method='L-BFGS-B', jac=g,
        #     #                   options={'disp': False, 'maxfun': 3})

        #     assert (theta == result.x).all() == False
        #     theta = result.x

        # preds = cls._p(params.e, params.R, dataset.valid_rels) > 0.5
        # acc = map(lambda x: x[0].r == x[1] , zip(dataset.valid_rels, preds))
        # acc = np.sum(acc) / len(acc)
        # print("At exit: Validation likelihood: {} accuracy: {}".format(
        #     -cls._loglik(params.e, params.R, gamma, dataset.valid_rels), acc))

        theta_star = theta
        self.params = vec_to_params(D, NE, NR, rho, theta_star)
        return FitResult(0, 0)


    def eval_one(self, entity1, entity2, relation):
        assert(0 <= entity1 < len(self.params.e) and
               0 <= entity2 < len(self.params.e) and
               0 <= relation < len(self.params.R))
        return self.__class__._p(self.params.e, self.params.R,
                                 [RelationInstance(entity1, entity2, relation, False)])

    def eval_rels(self, rels):
        return self.__class__._p(self.params.e, self.params.R, rels)

    @staticmethod
    def _p(e, R, rels):
        p_one = lambda rel: sigmoid(
            np.einsum('ij,il,lk,kj->', # e'_i * U_k * V_k * e_j
                      e[rel.i], R[rel.k].U, R[rel.k].V, e[rel.j]).flatten()[0])
        return array(map(p_one, rels))

    def obj(self, gamma, rel_pairs):
        return self.__class__._obj(
            self.params.e, self.params.R, gamma, rel_pairs)

    @staticmethod
    def _obj(e, R, gamma, rel_pairs):
        D, rho = R[0].U.shape
        NE, NR =  len(e), len(R)
        bpr_obj = 0
        for pos_rel, neg_rel in rel_pairs:
            corr = np.einsum('ij,il,lk,kj->', # e'_i * U_k * V_k * e_j
                             e[pos_rel.i], R[pos_rel.k].U, R[pos_rel.k].V, e[pos_rel.j])
            corr -= np.einsum('ij,il,lk,kj->', # e'_i * U_k * V_k * e_j
                              e[neg_rel.i], R[neg_rel.k].U, R[neg_rel.k].V, e[neg_rel.j])
            bpr_obj -= log1p(exp(-corr))
        # bpr_obj /= len(rel_pairs)
        if gamma > 0:
            reg = np.sum(map(lambda r: l2_norm2(r.U) + l2_norm2(r.V), R)) + \
                  np.sum(map(l2_norm2, e))
            bpr_obj -= reg * gamma / 2
            #bpr_obj -= log(2 * pi * gamma) * (D*NE + 2*D*rho*NR) / 2
        assert not isnan(bpr_obj)
        return bpr_obj

    @staticmethod
    def _dobj_sparse(e, R, gamma, rel_pairs,
                     compute_du=True, compute_dv=True):
        D, rho = R[0].U.shape
        NE, NR, N = len(e), len(R), len(rel_pairs)
        de = defaultdict(lambda: zeros((D, 1)))
        dR = defaultdict(lambda: RelationParams(zeros((D, rho)), zeros((rho, D))))
        for pos_rel, neg_rel in rel_pairs:
            pos_ei, pos_ej = e[pos_rel.i], e[pos_rel.j]
            pos_Uk, pos_Vk = R[pos_rel.k]
            neg_ei, neg_ej = e[neg_rel.i], e[neg_rel.j]
            neg_Uk, neg_Vk = R[neg_rel.k]
            corr = einsum('ij,il,lk,kj->', pos_ei, pos_Uk, pos_Vk, pos_ej)
            corr -= einsum('ij,il,lk,kj->', neg_ei, neg_Uk, neg_Vk, neg_ej)
            premult = ((1.0 / (1.0 + exp(-corr))) - 1.0)
            de[pos_rel.i] -= premult * einsum('ik,kl,lj->ij', pos_Uk, pos_Vk, pos_ej)
            de[pos_rel.j] -= premult * einsum('ji,kj,kl->il', pos_Vk, pos_Uk, pos_ei)
            de[neg_rel.i] += premult * einsum('ik,kl,lj->ij', neg_Uk, neg_Vk, neg_ej)
            de[neg_rel.j] += premult * einsum('ji,kj,kl->il', neg_Vk, neg_Uk, neg_ei)
            if compute_du:
                dR[pos_rel.k].U -= premult * einsum('ij,kj,lk->il', pos_ei, pos_ej, pos_Vk)
                dR[pos_rel.k].U += premult * einsum('ij,kj,lk->il', neg_ei, neg_ej, neg_Vk)
            if compute_dv:
                dR[pos_rel.k].V -= premult * einsum('ji,jk,lk->il', pos_Uk, pos_ei, pos_ej)
                dR[pos_rel.k].V += premult * einsum('ji,jk,lk->il', neg_Uk, neg_ei, neg_ej)

        for i, ent  in de.iteritems():
            ent -= e[i] * gamma
        if compute_du and compute_dv:
            for k, rel in dR.iteritems():
                rel.U -= gamma * R[k].U
                rel.V -= gamma * R[k].V
        elif compute_du:
            for k, rel in dR.iteritems():
                rel.U -= gamma * R[k].U
        elif compute_dv:
            for k, rel in dR.iteritems():
                rel.V -= gamma * R[k].V
        return SparseGradient(de, dR)

    @staticmethod
    def _dobj(e, R, gamma, rel_pairs,
              compute_du=True, compute_dv=True):
        NE, NR = len(e), len(R)
        de, dR = [], []
        for i in xrange(NE):
            de.append(-e[i] * gamma)
        for k in xrange(NR):
            dU = -R[k].U * gamma if compute_du else zeros_like(R[k].U)
            dV = -R[k].V * gamma if compute_dv else zeros_like(R[k].V)
            dR.append(RelationParams(dU, dV))
        sparse_grad = BprBilinear._dobj_sparse(
            e, R, gamma, rel_pairs, compute_du, compute_dv)
        print("sparse_grad={}".format(sparse_grad))
        for i, ent in sparse_grad.de.iteritems():
            de[i] = ent
        for k, rel in sparse_grad.dR.iteritems():
            dR[k] = rel
        return LogBilinearParams(de, dR)


class TestBprBilinear(unittest.TestCase):
    def test_init(self):
        model = BprBilinear(4, 10, 11, 3, 1)
        self.assertEqual(10, len(model.params.e))
        self.assertEqual(11, len(model.params.R))
        self.assertTrue(all(map(lambda e: len(e) == 4, model.params.e)))
        for (U, V) in model.params.R:
            self.assertEqual((4, 3), U.shape)
            self.assertEqual((3, 4), V.shape)
        print("model={}".format(model.eval_one(1,1,1)))

    def test_check_gradient(self):
        D, NE, NR, rho = 3, 3, 3, 2
        gamma = 0.2
        lb = BprBilinear(D, NE, NR, rho, 1)
        rel_pairs = [(RelationInstance(0, 1, 0, 1), RelationInstance(1, 2, 0, 0)),
                     (RelationInstance(2, 1, 1, 1), RelationInstance(1, 0, 1, 0))]

        def obj(x):
            params = vec_to_params(D, NE, NR, rho, x)
            return lb._obj(params.e, params.R, gamma, rel_pairs)

        def grad_both(x):
            params = vec_to_params(D, NE, NR, rho, x)
            return params_to_vec(lb._dobj(params.e, params.R, gamma, rel_pairs))

        def grad_dU(x):
            params = vec_to_params(D, NE, NR, rho, x)
            return params_to_vec(lb._dobj(
                params.e, params.R, gamma, rel_pairs, compute_du=True, compute_dv=False))

        def grad_dV(x):
            params = vec_to_params(D, NE, NR, rho, x)
            return params_to_vec(lb._dobj(
                params.e, params.R, gamma, rel_pairs, compute_du=False, compute_dv=True))

        x0 = params_to_vec(lb.params)
        for i in xrange(1):
            dnum = vec_to_params(D, NE, NR, rho, numerical_grad(obj, x0))
            dexact = vec_to_params(D, NE, NR, rho, grad_both(x0))
            dexact_dU = vec_to_params(D, NE, NR, rho, grad_dU(x0))
            dexact_dV = vec_to_params(D, NE, NR, rho, grad_dV(x0))
            print("\ndnum:\n{}".format(dnum))
            print("\ndexact:\n{}".format(dexact))
            print("\ndexact_dU:\n{}".format(dexact_dU))
            print("\ndexact_dV:\n{}".format(dexact_dV))
            print(params_to_vec(dnum) / params_to_vec(dexact))
            self.assertLess(
                max((params_to_vec(dnum) - params_to_vec(dexact)) ** 2), 1e-5)
            for r_num, r_dU, r_dV in zip(dnum.R, dexact_dU.R, dexact_dV.R):
                self.assertLess(
                    max((r_num.U.flatten() - r_dU.U.flatten()) ** 2), 1e-5)
                self.assertTrue((r_dU.V == 0).all())

                self.assertLess(
                    max((r_num.V.flatten() - r_dV.V.flatten()) ** 2), 1e-5)
                self.assertTrue((r_dV.U == 0).all())
            x0 += randn(len(x0)) / 100

    def test_split_train_test(self):
        rng = range(1, 11)
        self.assertEqual(TrainTestSplit([1,2,3,4,5,6,7], [8,9,10]),
                         split_train_test(rng, 0.3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Log-Bilinear model for relation extraction.')
    _arg = parser.add_argument
    _arg('--n-iter', type=int, action='store',
         metavar='NUM', help='Number of NAG iterations to perform.')
    _arg('--train', type=str, action='store',
         metavar='PATH', help='Loads pickled training set from file.')
    _arg('--train-csv', type=str, action='store',
         metavar='PATH', help='Load the training set from a CSV file.')
    _arg('--train-out', type=str, action='store',
         metavar='PATH', help='Pickles to training set to a file.')
    _arg('--save', type=str, action='store',  metavar='PATH',
         help='Saves the trained model to a file.')
    _arg('--load', type=str, action='store',  metavar='PATH',
         help='Loads a model from file (for predictions or further training).')
    _arg('--unittest', action='store_true', help='Run unittests')
    args = parser.parse_args()

    train_set = None
    if args.train:
        train_set = pickle.load(open(args.train, 'rb'))

    if args.train_csv:
        train_set = csv_to_rels(args.train_csv)

    if train_set:
        n_train_rels = len(train_set.train_rels)
        n_valid_rels = len(train_set.valid_rels)
        print("Loaded a dataset of {} relation ({} train + {} validation)."
              .format(n_train_rels + n_valid_rels, n_train_rels, n_valid_rels))

    if args.train_out:
        if train_set is None:
            print("Error: No training set available to pickle to file {}"
                  .format(args.train_out))
            sys.exit(1)
        pickle.dump(train_set, open(args.train_out, 'wb'), -1)

    if train_set:
        gamma = 1e-5
        if args.load:
            lb = pickle.load(open(args.load, 'rb'))
        else:
            lb = BprBilinear(50, len(train_set.ent_dict._map),
                             len(train_set.rel_dict._map), 10, .3)
        fr = lb.fit(train_set, gamma)
        # fout = open("valid_eval", 'w')
        # rel_decode = train_set.rel_dict.decode
        # ent_decode = train_set.ent_dict.decode
        # for rel in train_set.valid_rels:
        #     pred = lb.eval_one(rel.i, rel.j, rel.k)
        #     fout.write("{}\t{}\t{}\t{}\t{}\n".format(
        #         pred, rel.r, rel_decode(rel.k),
        #         ent_decode(rel.i), ent_decode(rel.j)))
        if args.save:
            pickle.dump(lb, open(args.save, 'wb'), -1)

    if args.unittest:
        unittest.main(verbosity=2, module='logbi', exit=False,
                      argv=[sys.argv[0]])
