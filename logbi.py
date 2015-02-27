#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import cPickle as pickle
import sys
import unittest
from collections import namedtuple
from random import shuffle

import numpy as np
import pandas as pd
from numpy import array, concatenate, dot, eye, log, log1p, outer, zeros, \
    r_, c_, pi, mean, cov, exp
from numpy.random import rand, randn
from numpy.linalg import cholesky, det, inv
from scipy import optimize
from scipy.optimize import check_grad, minimize


DEFAULT_MAXNUMLINESEARCH = 150

sp_minimize = optimize.minimize
sigmoid = lambda u: 1.0 / (1.0 + exp(-u))
# sigmoid = lambda u: (1.0 + u / np.sqrt(1 + u * u)) / 2
LogBilinearParams = namedtuple('LogBilinearParams', ['e', 'R'])

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

def params_to_vec(params):
    D = params.e[0].shape[0]
    NE = len(params.e)
    NR = len(params.R)

    theta = np.empty(D * NE + D * D * NR)
    for i in xrange(NE):
        theta[D * i:D * (i + 1)] = params.e[i].reshape((-1,))
    for k in xrange(NR):
        theta[D*NE + D*D*k:D*NE + D*D*(k + 1)] = params.R[k].reshape((-1,))
    return theta


def vec_to_params(D, NE, NR, theta):
    e, R = [], []
    for i in xrange(NE):
        e.append(theta[D * i:D * (i + 1)].reshape((D, 1)))
    for k in xrange(NR):
        R.append(theta[D*NE + D*D*k:D*NE + D*D*(k + 1)].reshape(D, D))
    return LogBilinearParams(e, R)


def numerical_grad(f, x0):
    EPS = 1e-7
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


def csv_to_rels(path):
    rels = pd.read_csv(path, sep='\t', header=None,
                       names=['rel', 'lhs', 'rhs', 'train', 'r'])
    ent_dict = FeatureDictionary()
    rel_dict = FeatureDictionary()
    train_rels, valid_rels = [], []
    i = 0;
    for (ix, row) in rels.iterrows():
        if i % 2 == 0: #row.train == 'Test':
            valid_rels.append(RelationInstance(
                ent_dict.encode(row.lhs), ent_dict.encode(row.rhs),
                rel_dict.encode(row.rel), np.float(row.r)))
        else:
            train_rels.append(RelationInstance(
                ent_dict.encode(row.lhs), ent_dict.encode(row.rhs),
                rel_dict.encode(row.rel), np.float(row.r)))
        i += 1
    return Dataset(ent_dict, rel_dict, train_rels, valid_rels)


def l2_norm2(x):
    return np.sum(x[:] ** 2)


def compute_accuracy(predictions, ground, threshold=0.5):
    acc = map(lambda x: x[0].r == x[1] , zip(ground, predictions))
    return np.sum(acc) / float(len(acc))


class LogBilinear(object):
    def __init__(self, D, NE, NR, sigma=1):
        e, R = [], []
        for _ in xrange(NR):
            R.append(sigma * randn(D, D))
        for _ in xrange(NE):
            e.append(sigma * randn(D, 1))

        self.params = LogBilinearParams(e, R)

    def fit(self, dataset, gamma=0):
        cls = self.__class__
        params = self.params
        D, NE, NR = params.e[0].shape[0], len(params.e), len(params.R)
        # train_rev = map(lambda r: RelationInstance(r.j, r.i, r.k, 1 - r.r), train_set)
        # train_set.extend(train_rev)
        # cross_rev = map(lambda r: RelationInstance(r.j, r.i, r.k, 1 - r.r), cross_set)
        # cross_set.extend(cross_rev)
        valid_len = len(dataset.valid_rels)
        n_ones = sum(map(lambda x: x.r == 1, dataset.valid_rels)) / float(valid_len)
        print("Training model with {} datapoints ({} held out for validation with {}% =1)"
              .format(len(dataset.train_rels), valid_len, n_ones))

        def obj(x):
            params = vec_to_params(D, NE, NR, x)
            preds = cls._p(params.e, params.R, dataset.valid_rels) > 0.5
            train_nll = -cls._loglik(params.e, params.R, gamma, dataset.train_rels)
            train_acc = compute_accuracy(preds, dataset.train_rels)
            valid_nll = -cls._loglik(params.e, params.R, gamma, dataset.valid_rels)
            valid_acc = compute_accuracy(preds, dataset.valid_rels)
            print("valid_nll = {} / valid_acc = {} / train_nll = {} / "
                  "train_acc = {}".format(
                      valid_nll, valid_acc, train_nll, train_acc))
            return train_nll

        def grad(x):
            params = vec_to_params(D, NE, NR, x)
            return -1.0 * params_to_vec(
                cls._dloglik(params.e, params.R, gamma, dataset.train_rels))

        theta = params_to_vec(params)
        print("Number of model parameters: {}".format(len(theta)))

        # theta_update = zeros(len(theta))
        # alpha, mu = 2, 0.8
        # N_ITER = 1000
        # for iter in xrange(N_ITER):
        #     probe = theta + theta_update
        #     theta_update = mu * theta_update - alpha * grad(probe)
        #     theta +=  theta_update
        #     print("Objective: {} (alpha={})".format(obj(theta), alpha))
        # theta_star = theta

        result = minimize(obj, theta, method='L-BFGS-B', jac=grad,
                          options={'disp': True, 'maxiter': 200,
                                   'ftol': 0, 'gtol': 1e-8})
        theta_star = result.x

        params = vec_to_params(D, NE, NR, theta_star)
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
        return array(
            map(lambda r: sigmoid(e[r.i].T.dot(R[r.k]).dot(e[r.j])), rels))

    def loglik(self, gamma, rels):
        return self.__class__._loglik(
            self.params.e, self.params.R, gamma, rels)

    @staticmethod
    def _loglik(e, R, gamma, rels):
        D, NE, NR = e[0].shape[0], len(e), len(R)
        loglik = 0
        for rel in rels:
            corr = np.einsum('ij,ik,kj->', # e'_i * R_k * e_j
                             e[rel.i], R[rel.k], e[rel.j])
            scorr = sigmoid(corr)
            assert rel.r == 1 or rel.r == 0
            if (scorr == 1.0 and rel.r == 1) or (scorr == 0.0 and rel.r == 0):
                continue
            loglik += (rel.r - 1) * (corr + log1p(exp(-corr))) + \
                      rel.r * log(scorr)
            if np.isnan(loglik):
                import pdb
                pdb.set_trace()
            assert not np.isnan(loglik)
        loglik /= len(rels)
        if gamma > 0:
            reg = np.sum(map(l2_norm2, R)) + np.sum(map(l2_norm2, e))
            loglik -= gamma / (D * NE + D * D * NR) * reg
        assert isinstance(loglik, np.float)
        return loglik

    @staticmethod
    def _dloglik(e, R, gamma, rels):
        D, NE, NR = e[0].shape[0], len(e), len(R)
        N = len(rels)
        de, dR = [], []
        gamma_hat = gamma / (D * NE + D * D * NR)
        for i in xrange(NE):
            de.append(-2 * gamma_hat * e[i])
        for k in xrange(NR):
            dR.append(-2 * gamma_hat * R[k])
        for rel in rels:
            corr = sigmoid(np.einsum('ij,ik,kj->', # e'_i * R_k * e_j
                                     e[rel.i], R[rel.k], e[rel.j]))
            premult = (rel.r - 1) * corr + rel.r * (1 - corr)
            de[rel.i] += premult * R[rel.k].dot(e[rel.j]) / N
            de[rel.j] += premult * R[rel.k].T.dot(e[rel.i]) / N
            dR[rel.k] += premult * e[rel.i].dot(e[rel.j].T) / N
        # print(de)
        # print(dR)
        return LogBilinearParams(de, dR)


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


class TestLogBilinear(unittest.TestCase):
    def test_init(self):
        lb = LogBilinear(4, 10, 11, 1)
        self.assertEqual(10, len(lb.params.e))
        self.assertEqual(11, len(lb.params.R))
        self.assertTrue(all(map(lambda e: len(e) == 4, lb.params.e)))

        print("lb={}".format(lb.eval_one(1,1,1)))

    def test_params_to_vec(self):
        D, NE, NR = 3, 9, 14
        lb = LogBilinear(D, NE, NR, 1)

        to_vec = params_to_vec(lb.params)
        self.assertEqual(lb.params.e[0][0, 0], to_vec[0])
        self.assertEqual(lb.params.e[0][1, 0], to_vec[1])
        self.assertEqual(lb.params.e[0][2, 0], to_vec[2])
        self.assertEqual(lb.params.e[1][0, 0], to_vec[3])
        self.assertEqual(lb.params.e[1][1, 0], to_vec[4])
        self.assertEqual(lb.params.R[1][0, 1], to_vec[D*NE + D*D + 1])

        to_vec_and_back = vec_to_params(D, NE, NR, to_vec)
        for i in xrange(NE):
            self.assertEqual(lb.params.e[i].shape, to_vec_and_back.e[i].shape)
            self.assertTrue((lb.params.e[i] == to_vec_and_back.e[i]).all())
        for k in xrange(NR):
            self.assertEqual(lb.params.R[k].shape, to_vec_and_back.R[k].shape)
            self.assertTrue((lb.params.R[k] == to_vec_and_back.R[k]).all())

    def test_check_gradient(self):
        D, NE, NR = 2, 3, 3
        lb = LogBilinear(D, NE, NR, 1)
        rels = [RelationInstance(0, 1, 0, 1)]

        def obj(x):
            params = vec_to_params(D, NE, NR, x)
            return lb._loglik(params.e, params.R, 1, rels)

        def grad(x):
            params = vec_to_params(D, NE, NR, x)
            return params_to_vec(lb._dloglik(params.e, params.R, 1, rels))

        x0 = params_to_vec(lb.params)
        for i in xrange(100):
            # print("\nnum grad")
            # print(vec_to_params(D, NE, NR, numerical_grad(obj, x0)))
            # print("\ngrad")
            # print(vec_to_params(D, NE, NR, grad(x0)))
            self.assertLess(check_grad(obj, grad, x0), 1e-6)
            x0 += randn(len(x0)) / 100

    def test_split_train_test(self):
        rng = range(1, 11)
        self.assertEqual(TrainTestSplit([1,2,3,4,5,6,7], [8,9,10]),
                         split_train_test(rng, 0.3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Log-Bilinear model for relation extraction.')
    _arg = parser.add_argument
    _arg('--load', type=str, action='store',
         metavar='PATH', help='Loads a pickled model from file.')
    _arg('--n-iter', type=int, action='store',
         metavar='NUM', help='Number of NAG iterations to perform.')
    _arg('--train', type=str, action='store',
         metavar='PATH', help='Loads pickled training set from file.')
    _arg('--train-csv', type=str, action='store',
         metavar='PATH', help='Load the training set from a CSV file.')
    _arg('--train-out', type=str, action='store',
         metavar='PATH', help='Pickles to training set to a file.')
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

    # if args.load:
    #     model = pickle.load(open(args.load, 'wb'))
    #     model.fit(train_set.rels, 10)
    #     pickle.dump(lb, open("model2.pickle", 'wb'), -1)

    if train_set:
        lb = LogBilinear(30, len(train_set.ent_dict._map),
                         len(train_set.rel_dict._map), .1)
        fr = lb.fit(train_set, 150)
        fout = open("valid_eval", 'w')

        rel_decode = train_set.rel_dict.decode
        ent_decode = train_set.ent_dict.decode
        for rel in train_set.valid_rels:
            pred = lb.eval_one(rel.i, rel.j, rel.k)
            fout.write("{}\t{}\t{}\t{}\t{}\n".format(
                pred, rel.r, rel_decode(rel.k),
                ent_decode(rel.i), ent_decode(rel.j)))
        pickle.dump(lb, open("model.pickle", 'wb'), -1)

    if args.unittest:
        unittest.main(verbosity=2, module='logbi', exit=False,
                      argv=[sys.argv[0]])


        # def gauss_logZ(L):
#     assert(L.shape[0] == L.shape[1])
#     assert(all(np.tril(L_noise) == L_noise))
#     return -log(det(L)) + D * log(2 * pi) / 2.

# def loglik(X, mu, L):
#     D, N = X.shape
#     LtXzero = L.T.dot(X - mu.reshape(D, 1))
#     l = N * log(det(L)) - D * N * log(2 * pi) / 2.
#     l -= np.einsum('ij,ji->', LtXzero.T, LtXzero) / 2.
#     return -l


# def loglik_vec(X, theta):
#     D = X.shape[0]
#     (mu, L, c) = vec_to_params(theta)
#     return loglik(X, mu, L)


# def params_to_vec(mu, L, c):
#     assert(mu.size == L.shape[0] == L.shape[1])
#     D = mu.size
#     c = np.array([c]) if isinstance(c, np.float) else c
#     assert(isinstance(c, np.ndarray))
#     return concatenate((mu, L[np.tril_indices(D)], c))


# def vec_to_params(theta):
#     D = (np.sqrt(8. * theta.size + 1) - 3.) / 2.
#     assert(int(D) == D)
#     D = int(D)
#     assert(theta.size == D + D * (D + 1) / 2 + 1)

#     mu = theta[0:D]
#     L_elems = theta[D:D + D * (D + 1) / 2]
#     L = zeros((D, D))
#     L[np.tril_indices(D)] = L_elems
#     c = theta[-1]
#     return GaussParams(mu, L, c)

    # def test_check_grad(self):
    #     D = 2
    #     S = c_[[2., .2], [.2, 2.]]
    #     X = mvn.rvs(randn(2), S, 100).T

    #     mu_noise, P_noise = r_[-1., 1.], .5 * c_[[1., .1], [.1, 1.]]
    #     L_noise = cholesky(P_noise)
    #     Y = mvn.rvs(mu_noise, inv(P_noise), 200).T

    #     obj = lambda u: NceGauss.J(
    #         X, Y, mu_noise, L_noise, *vec_to_params(u))
    #     grad = lambda u: params_to_vec(
    #         *NceGauss.dJ(X, Y, mu_noise, L_noise, *vec_to_params(u)))
    #     grad_diff = lambda u: check_grad(obj, grad, u)

    #     for i in xrange(100):
    #         u = r_[0,0,2,0,2,1] + randn(6) / 10
    #         self.assertLess(grad_diff(u), 1e-5)

    # def test_sanity_fit(self):
    #     mu, P = r_[0., 0.], c_[[2., .2], [.2, 2.]]
    #     L = cholesky(P)
    #     Td, k = 100, 2
    #     X = mvn.rvs(mu, inv(P), Td).T
    #     theta = GaussParams(zeros(2), eye(2), 1.)

    #     theta_star, Y = self.model.fit_nce(
    #         X, k, mu_noise=randn(2), L_noise=(rand() + 1) * eye(2),
    #         mu0=mu, L0=L, maxnumlinesearch=2000, verbose=False)
    #     noise = self.model.params_noise
    #     self.assertLess(NceGauss.J(X, Y, noise.mu, noise.L, *theta_star),
    #                     NceGauss.J(X, Y, noise.mu, noise.L, *theta))
    #     self.assertLess(np.sum(params_to_vec(
    #         *NceGauss.dJ(X, Y, noise.mu, noise.L, *theta_star)) ** 2), 1e-6)



    # def fit_nce(self, X, k=1, mu_noise=None, L_noise=None,
    #             mu0=None, L0=None, c0=None, method='minimize',
    #             maxnumlinesearch=None, maxnumfuneval=None, verbose=False):
    #     _class = self.__class__
    #     D, Td = X.shape
    #     self._init_params(D, mu_noise, L_noise, mu0, L0, c0)

    #     noise = self._params_noise
    #     Y = mvn.rvs(noise.mu, noise.L, k * Td).T

    #     maxnumlinesearch = maxnumlinesearch or DEFAULT_MAXNUMLINESEARCH
    #     obj = lambda u: _class.J(X, Y, noise.mu, noise.L, *vec_to_params(u))
    #     grad = lambda u: params_to_vec(
    #         *_class.dJ(X, Y, noise.mu, noise.L, *vec_to_params(u)))

    #     t0 = params_to_vec(*self._params_nce)
    #     if method == 'minimize':
    #         t_star = minimize(t0, obj, grad,
    #                           maxnumlinesearch=maxnumlinesearch,
    #                           maxnumfuneval=maxnumfuneval, verbose=verbose)[0]
    #     else:
    #         t_star = sp_minimize(obj, t0, method='BFGS', jac=grad,
    #                              options={'disp': verbose,
    #                                       'maxiter': maxnumlinesearch}).x
    #     self._params_nce = GaussParams(*vec_to_params(t_star))
    #     return (self._params_nce, Y)

    # def fit_ml(self, X):
    #     D = X.shape[0]
    #     mu = mean(X, 1)
    #     L = cholesky(inv(cov(X)))
    #     c = log(det(L)) - D * log(2 * pi) / 2.
    #     self._params_ml = GaussParams(mu, L, c)

    # @staticmethod
    # def _h(U, Uzero, D, k, mu_noise, L_noise, mu, L, c):
    #     assert(U.shape == Uzero.shape)
    #     Uzero_noise = U - mu_noise.reshape(D, 1)
    #     P, P_noise = L.dot(L.T), L_noise.dot(L_noise.T)
    #     log_pn = log(det(L_noise)) - D * log(2. * pi) / 2.
    #     log_pn -= np.einsum('ij,jk,ki->i',
    #                         Uzero_noise.T, P_noise, Uzero_noise) / 2.
    #     log_pm = -np.einsum('ij,jk,ki->i', Uzero.T, P, Uzero) / 2. + c
    #     return log_pm - log_pn - log(k)

    # @staticmethod
    # def J(X, Y, mu_noise, L_noise, mu, L, c):
    #     """NCE objective function with gaussian data likelihood X and
    #     gaussian noise Y."""
    #     assert(mu.size == X.shape[0] == Y.shape[0])
    #     r = sigmoid
    #     D, Td = X.shape
    #     Tn = Y.shape[1]
    #     k = Tn / Td

    #     Xzero, Yzero = X - mu.reshape(D, 1), Y - mu.reshape(D, 1)
    #     h = lambda U, Uzero: NceGauss._h(
    #         U, Uzero, D, k, mu_noise, L_noise, mu, L, c)
    #     Jm = -np.sum(log(1 + np.exp(-h(X, Xzero))))
    #     Jn = -np.sum(log(1 + np.exp(h(Y, Yzero))))

    #     print("Jm=%10.4f "
    #           "max(-h(X, Xzero))=%12.3f " %
    #           (Jm,  max(-h(X, Xzero))))
    #     print("Jn=%10.4f "
    #           "max(-h(Y, Yzero))=%12.3f " %
    #           (Jn,  max(-h(Y, Yzero))))
    #     print("mu=%s\n; L=%10.4f\n" % (mu, loglik(X, mu, L)))

    #     return -(Jm + Jn) / Td

    # @staticmethod
    # def dJ(X, Y, mu_noise, L_noise, mu, L, c):
    #     """Gradient of the NCE objective function."""
    #     assert(mu.size == X.shape[0] == Y.shape[0])
    #     r = sigmoid
    #     D, Td = X.shape
    #     Tn = Y.shape[1]
    #     k = Tn / Td
    #     P, P_noise = dot(L, L.T), dot(L_noise, L_noise.T)

    #     Xzero, Yzero = X - mu.reshape(D, 1), Y - mu.reshape(D, 1)
    #     h = lambda U, Uzero: NceGauss._h(
    #         U, Uzero, D, k, mu_noise, L_noise, mu, L, c)
    #     rhX, rhY = r(-h(X, Xzero)), r(h(Y, Yzero))

    #     dmu = np.sum(rhX * dot(P, Xzero), 1) - np.sum(rhY * dot(P, Yzero), 1)
    #     dmu /= Td

    #     dL = -np.einsum('k,ik,jk->ij', rhX, Xzero, L.T.dot(Xzero))
    #     dL += np.einsum('k,ik,jk->ij', rhY, Yzero, L.T.dot(Yzero))
    #     dL /= Td

    #     dc = (np.sum(rhX) - np.sum(rhY)) / Td
    #     return (-dmu, -dL, -array([dc]))
