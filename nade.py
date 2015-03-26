#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import pickle
import sys
import unittest
from collections import namedtuple
from itertools import cycle, islice

import numpy as np
import numexpr
from numpy import array, concatenate, dot, eye, inf, isnan, log, log1p, \
    r_, c_, pi, mean, cov, einsum, exp, zeros, zeros_like, vstack, prod
from numpy.random import rand, randn

import theano
import theano.tensor as T

# theano.config.optimizer = 'None'
# theano.config.exception_verbosity='high'

sigmoid = lambda x: 1.0 / (1.0 + T.exp(-x))
relu = lambda x: T.switch(x < 0.0, x, 0.0)


GaussNadeParams = namedtuple('GaussNadeParams', ['W', 'V', 'C', 'gamma'])
GaussNadeRawParams = namedtuple('GaussNadeRawParams', [
    'n_input', 'n_hidden', 'n_output', 'theta'])


class GaussNade(object):
    @staticmethod
    def params_size(n_input, n_hidden, n_output):
        return n_hidden * n_input  + n_output * n_hidden * n_input \
            + n_output * n_input + n_input

    @staticmethod
    def vec_to_params(n_input, n_hidden, n_output, theta):
        expected_size = GaussNade.params_size(n_input, n_hidden, n_output)
        assert expected_size == len(theta), "Expected len(theta) == {}, " \
            "found {}".format(expected_size, len(theta))

        W_shape = (n_hidden, n_input)
        V_shape = (n_output, n_hidden, n_input)
        C_shape = (n_output, n_input)
        gamma_shape = (n_input,)
        W_size, V_size, C_size = prod(W_shape), prod(V_shape), prod(C_shape)
        gamma_size = prod(gamma_shape)

        W = theta[0:W_size].reshape(*W_shape)
        V = theta[W_size:W_size + V_size].reshape(*V_shape)
        C = theta[W_size + V_size:W_size + V_size + C_size].reshape(*C_shape)
        gamma = theta[W_size + V_size + C_size:
                      W_size + V_size + C_size + gamma_size].reshape(*gamma_shape)
        return GaussNadeParams(W, V, C, gamma)

    @staticmethod
    def params_to_vec(params):
        n_hidden, n_input = params.W.shape
        n_output = params.V.shape[0]
        assert (n_output, n_hidden, n_input) == params.V.shape
        assert (n_output, n_input) == params.C.shape

        W_size, V_size = prod(params.W.shape), prod(params.V.shape)
        C_size, gamma_size = prod(params.C.shape), prod(params.gamma.shape)

        theta = np.empty(GaussNade.params_size(n_input, n_hidden, n_output))
        theta[0:W_size] = params.W.flatten()
        theta[W_size:W_size + V_size] = params.V.flatten()
        theta[W_size + V_size: W_size + V_size + C_size] = params.C.flatten()
        theta[W_size + V_size + C_size:W_size + V_size + C_size + gamma_size] \
            = params.gamma.flatten()
        return theta

    @staticmethod
    def load(param_stream):
        raw = pickle.load(param_stream)
        return GaussNade(raw.n_input, raw.n_hidden, raw.theta)

    def __init__(self, n_input, n_hidden, raw_params=None):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = 2
        if raw_params is None:
            self._init_params()
        else:
            self._raw_params = raw_params
            self.params = GaussNade.vec_to_params(
                self.n_input, self.n_hidden, self.n_output, raw_params)

        X = T.dmatrix('X')
        D, N = X.shape
        X_shift_bias = T.concatenate((T.ones((1, N)), X[:-1, :]), axis=0)
        X_shift_bias = T.shape_padleft(X_shift_bias)

        W, V, C = T.dmatrix('W'), T.dtensor3('V'), T.dmatrix('C')
        gamma = T.dvector('gamma')
        Ws = T.shape_padright(W).repeat(N, 2)

        latent = relu((Ws * X_shift_bias).cumsum(1) *
                      T.shape_padright(T.shape_padleft(gamma)))
        gauss_params = (T.shape_padright(V) * T.shape_padleft(latent)).sum(1)
        gauss_params += T.shape_padright(C)
        mus = gauss_params[0, :, :]
        log_sigmas = gauss_params[1, :, :]

        discrep = X - mus
        llik = -0.5 * log_sigmas - 0.5 * discrep * discrep / T.exp(log_sigmas)
        llik = llik.sum() / N

        opt_params = [W, V, C, gamma]
        self._loss = theano.function(
            opt_params + [X], [llik] + T.grad(llik, opt_params))

        # Predictive distribution
        X_bias = T.concatenate((T.ones((1, N)), X), axis=0)
        pred_latent = relu(W[:, :D + 1].dot(X_bias) * gamma[D])
        pred_params = V[:, :, D].dot(pred_latent) + C[:, D:D + 1]
        self._pred = theano.function(opt_params + [X], [pred_params])

    def _init_params(self, sigma=0.01):
        raw_size = GaussNade.params_size(
            self.n_input, self.n_hidden, self.n_output)
        self._raw_params = randn(raw_size) * sigma
        self.params = GaussNade.vec_to_params(
            self.n_input, self.n_hidden, self.n_output, self._raw_params)
        gamma = self.params.gamma
        gamma *= 0.0; gamma += 1.0;

    def dump(self, out_stream):
        pickle.dump(GaussNadeRawParams(self.n_input, self.n_hidden,
                                       self.n_output, self._raw_params),
                     out_stream, protocol=pickle.HIGHEST_PROTOCOL)

    def loss(self, X):
        p = self.params
        return self._loss(p.W, p.V, p.C, p.gamma, X)

    def pred(self, X):
        p = self.params
        return self._pred(p.W, p.V, p.C, p.gamma, X)

    def fit_nag(self, X, n_iter=200, batch_size=200, decay_iter=100):
        D, N = X.shape
        alpha, mu, alpha_decay = 0.001, 0.8, 0.1
        theta = self._raw_params.copy()
        theta_update = zeros_like(theta)

        for iter in xrange(n_iter):
            if iter + 1 % decay_iter == 0:
                alpha *= alpha_decay

            batch_index = (iter * batch_size) % (N - batch_size)
            probe = GaussNade.vec_to_params(
                self.n_input, self.n_hidden, self.n_output,
                theta + mu * theta_update)
            [cost, dW, dV, dC, dgamma] = self._loss(
                probe.W, probe.V, probe.C, probe.gamma,
                X[:, batch_index:batch_index + batch_size])
            grad = concatenate(
                (dW.flatten(), dV.flatten(), dC.flatten(), dgamma.flatten()))
            if max(np.abs(grad)) > 1000:
                print("dW:\n{}\ndV:\n{}dC:\n{}dgamma:\n{}\n".format(dW, dV, dC, dgamma))

            theta_update[:] = mu * theta_update + alpha * grad
            theta += theta_update

            print("alpha = {}, mean(|g|^2) = {}, llik at iter {}: {}" \
                  .format(alpha, mean(grad ** 2), iter, cost))
            assert np.isfinite(cost)
        self._raw_params[:] = theta

    def fit_ada(self, X, n_iter=300, batch_size=300, decay_iter=100):
        D, N = X.shape
        alpha, alpha_decay = 0.0001, 0.1
        theta = self._raw_params.copy()
        hist_grad2 = zeros_like(theta)
        for iter in xrange(n_iter):
            # if iter + 1 % decay_iter == 0:
            #     alpha *= alpha_decay

            batch_index = (iter * batch_size) % (N - batch_size)
            probe = GaussNade.vec_to_params(
                self.n_input, self.n_hidden, self.n_output, theta)
            [cost, dW, dV, dC, dgamma] = self._loss(
                probe.W, probe.V, probe.C, probe.gamma,
                X[:, batch_index:batch_index + batch_size])
            grad = concatenate(
                (dW.flatten(), dV.flatten(), dC.flatten(), dgamma.flatten()))

            # if cost < 0:
            #     import ipdb; ipdb.set_trace()
            #     print("dW:\n{}\ndV:\n{}dC:\n{}dgamma:\n{}\n".format(dW, dV, dC, dgamma))

            # hist_grad2 = .1 * (g ** 2) + .9 * hist_grad2
            # hist_grad2 = (g ** 2) / (itr + 1) + itr * hist_grad2 / (itr + 1)
            hist_grad2 += grad ** 2
            theta += alpha * grad / np.sqrt(1.0 + hist_grad2)

            # cost_after = self.loss(X[:, index:index + batch_size])[0]
            print("alpha = {}, mean(hist_grad2) = {}, mean(|grad|^2) = {}, "
                  "llik at iter {}: {}" \
                  .format(alpha, mean(hist_grad2), mean(grad ** 2), iter, cost))
            assert np.isfinite(cost)
        self._raw_params[:] = theta


class TestGaussNade(unittest.TestCase):
    def test_params_to_vec(self):
        dest = GaussNade(10, 5)
        self.assertEqual(180, len(dest._raw_params))

        W, V, C, gamma = dest.params
        self.assertEqual((5, 10), W.shape)
        self.assertEqual((2, 5, 10), V.shape)
        self.assertEqual((2, 10), C.shape)
        self.assertEqual((10,), gamma.shape)
        self.assertTrue(all(1 == gamma))

        vec = GaussNade.params_to_vec(dest.params)
        self.assertTrue(all(dest._raw_params == vec))

        from_vec = GaussNade.vec_to_params(
            dest.n_input, dest.n_hidden, dest.n_output, vec)
        self.assertTrue((W == from_vec.W).all())
        self.assertTrue((V == from_vec.V).all())
        self.assertTrue((C == from_vec.C).all())

    def test_serialization(self):
        import cStringIO
        sio = cStringIO.StringIO()
        dest = GaussNade(10, 5)
        dest.dump(sio)
        sio.seek(0)
        dest2 = GaussNade.load(sio)
        self.assertTrue(all(dest._raw_params == dest2._raw_params))

    def test_predictive_dist(self):
        dest = GaussNade(10, 3)
        x1 = (c_[1, 2, 3]).T;
        x2 = (c_[3, 2, -1]).T;
        W, V, C, gamma = dest.params

        def loss(x):
            xs = vstack((1, x))
            #W[]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AEVB implementation.')
    _arg = parser.add_argument
    _arg('--unittest', action='store_true', help='Run unittests')
    _arg('--load', action='store', type=str, help='Loads model from file.')
    _arg('--dump', action='store', type=str, help='Dumps model to file.')
    args = parser.parse_args()

    if args.unittest:
        unittest.main(verbosity=2, module='nade', exit=True,
                      argv=[sys.argv[0]])

    if args.load:
        dest = GaussNade.load(open(args.load, 'rb'))
    else:
        dest = GaussNade(784, 100)

    mnist = pickle.load(open('mnist/mnist.pickle', 'rb'))
    X = mnist['train'].reshape((-1, 784)).T
    dest.fit_ada(X)

    if args.dump:
        dest.dump(open(args.dump, 'wb'))
