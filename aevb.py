#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import pickle
import sys
import unittest
from collections import namedtuple
from itertools import cycle, islice

import numpy as np
import pandas as pd
import numexpr
from numpy import array, concatenate, dot, eye, inf, isnan, log, log1p, outer,\
    r_, c_, pi, mean, cov, einsum, exp, zeros, zeros_like, vstack
from numpy.random import rand, randn

import theano
import theano.tensor as T

theano.config.optimizer = 'None'
theano.config.exception_verbosity='high'

activ = lambda u: numexpr.evaluate("1.0 / (1.0 + exp(-u))")


def numerical_grad(f, x0):
    EPS = 1e-2
    grad = np.empty(len(x0))
    x_minus = x0
    x_plus = np.array(x0)

    for i in xrange(len(x0)):
        # print("{}/{}".format(i, len(x0)))
        x_minus[i] -= EPS/2
        x_plus[i] += EPS/2
        grad[i] = (f(x_plus) - f(x_minus)) / EPS
        x_minus[i] += EPS/2
        x_plus[i] -= EPS/2
    return grad


MLPParams = namedtuple('MLPParams', ['H', 'Hb', 'O', 'Ob'])
VAEParams = namedtuple('VAEParams', ['encoder', 'decoder'])

class MLP(object):
    @staticmethod
    def _params_to_vec(params):
        n_hid, n_in = params.H.shape
        n_out = params.O.shape[0]
        theta = zeros((n_in + 1) * n_hid + (n_hid + 1) * n_out)

        index = 0
        theta[index:index + n_in * n_hid] = params.H.flatten()
        index += n_in * n_hid
        theta[index:index + n_hid] = params.Hb.flatten()
        index += n_hid
        theta[index:index + n_hid * n_out] = params.O.flatten()
        index += n_hid * n_out
        theta[index:index + n_out] = params.Ob.flatten()
        return theta

    @staticmethod
    def _params_len(n_in, n_hid, n_out):
        return (n_in + 1) * n_hid + (n_hid + 1) * n_out

    @staticmethod
    def _vec_to_params(n_in, n_hid, n_out, theta):
        assert (n_in + 1) * n_hid + (n_hid + 1) * n_out == len(theta)
        index = 0

        H = theta[index:index + n_in * n_hid].reshape((n_hid, n_in))
        index += n_in * n_hid
        Hb = theta[index:index + n_hid].reshape((n_hid, 1))
        index += n_hid

        O = theta[index:index + n_hid * n_out].reshape((n_out, n_hid))
        index += n_hid * n_out
        Ob = theta[index:index + n_out].reshape((n_out, 1))
        return MLPParams(H, Hb, O, Ob)

    @staticmethod
    def _pred(params, X):
        H, Hb, O, Ob = params
        return O.dot(activ(H.dot(X) + Hb)) + Ob

    @staticmethod
    def _loss(params, X, Y):
        return ((MLP._pred(params, X) - Y).flatten() ** 2).sum()

    @staticmethod
    def _dloss(params, X, Y):
        H, Hb, O, Ob = params
        hid_out = activ(H.dot(X) + Hb)
        diff = 2 * (O.dot(hid_out) + Ob - Y)

        dO = diff.dot(hid_out.T)
        dOb = diff.sum(1)
        dHb = O.T.dot(diff)
        dHb = numexpr.evaluate("dHb * hid_out * (1.0 - hid_out)")
        dH = dHb.dot(X.T)
        dHb = dHb.sum(1).reshape((-1, 1))

        # import ipdb; ipdb.set_trace()
        return MLPParams(dH, dHb, dO, dOb)

    def __init__(self, n_in, n_hid, n_out, sigma=.01, params=None):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        if params:
            self.params = params
        else:
            init = lambda x, y: sigma * randn(x, y)
            self.params = MLPParams(
                init(n_hid, n_in), zeros((n_hid, 1)),
                init(n_out, n_hid), zeros((n_out, 1)));

    def params_as_vec(self):
        return MLP._params_to_vec(self.params)

    def set_params_from_vec(self, X):
        return MLP._pred(self.params, X)

    def pred(self, X):
        assert X.shape[0] == self.n_in
        return MLP._pred(self.params, X)

    def loss(self, X, Y):
        assert X.shape[0] == self.n_in and Y.shape[0] == self.n_out \
            and X.shape[1] == Y.shape[1]
        return MLP._loss(self.params, X, Y)

    def dloss(self, X, Y):
        return MLP._dloss(self.params, X, Y)

    def fit(self, X, Y, batch_size=200, max_iter=100):
        cls = self.__class__
        vec_to_params = lambda theta: MLP._vec_to_params(
            self.n_in, self.n_hid, self.n_out, theta)

        alpha, mu = .0001, 0.8
        theta = self.params_as_vec()
        theta_update = zeros_like(theta)
        batch_index = 0
        for cur_iter in xrange(max_iter):
            if X.shape[1] - batch_index < batch_size:
                batch_index = 0
            batch_X = X[:, batch_index:batch_index + batch_size]
            batch_Y = Y[:, batch_index:batch_index + batch_size]
            batch_index += batch_size
            # print("Batch {} [{}:{}]".format(
            #     cur_iter, batch_index, batch_index + batch_size))
            # print("X shape: {}".format(batch_X.shape))
            # print("Y shape: {}".format(batch_Y.shape))
            # print(cls._dloss(vec_to_params(theta), batch_X, batch_Y))
            grad = cls._params_to_vec(
                cls._dloss(vec_to_params(theta), batch_X, batch_Y))
            theta_update = -alpha * grad + mu * theta_update
            theta += theta_update

            print("Loss: {}".format(
                cls._loss(vec_to_params(theta), batch_X, batch_Y)))
        self.params = vec_to_params(theta)


class TestMLP(unittest.TestCase):
    def test_init(self):
        n_in, n_hid, n_out = 4, 9, 5
        mlp = MLP(n_in, n_hid, n_out)
        self.assertEqual((n_hid, n_in), mlp.params.H.shape)
        self.assertEqual((n_hid, 1), mlp.params.Hb.shape)
        self.assertEqual((n_out, n_hid), mlp.params.O.shape)
        self.assertEqual((n_out, 1), mlp.params.Ob.shape)

    def test_params_to_vec(self):
        n_in, n_hid, n_out = 4, 9, 5
        mlp = MLP(n_in, n_hid, n_out)

        as_vec = MLP._params_to_vec(mlp.params)
        self.assertEqual(mlp.params.H[0, 0], as_vec[0])
        self.assertEqual(mlp.params.Hb[0], as_vec[n_in * n_hid])
        self.assertEqual(mlp.params.O[0, 0], as_vec[(n_in + 1) * n_hid])
        self.assertEqual(mlp.params.Ob[0], as_vec[(n_in + 1) * n_hid + n_hid*n_out])
        self.assertTrue((as_vec == mlp.params_as_vec()).all())

    def test_check_gradient(self):
        n_in, n_hid, n_out = 3, 4, 3
        vec_to_params = lambda theta: MLP._vec_to_params(n_in, n_hid, n_out, theta)
        mlp = MLP(n_in, n_hid, n_out)

        X = randn(n_in, 10)
        Y = randn(n_out, 10)

        def loss(theta):
            return mlp._loss(vec_to_params(theta), X, Y)

        theta = mlp.params_as_vec()
        for i in xrange(100):
            params = vec_to_params(theta)
            dnum = vec_to_params(numerical_grad(loss, theta))
            dexact = mlp._dloss(params, X, Y)
            # print("\ndnum:\n{}".format(dnum))
            # print("\ndexact:\n{}".format(dexact))
            # print("\ndratio:\n{}".format(
            #     vec_to_params(MLP._params_to_vec(dnum) / MLP._params_to_vec(dexact))))
            self.assertLess(
                max((MLP._params_to_vec(dnum) - MLP._params_to_vec(dexact)) ** 2), 1e-14)
            theta += randn(len(theta)) / 10


class VAE(object):
    @staticmethod
    def _params_to_vec(encoder, decoder):
        return concatenate((encoder.params_as_vec(), decoder.params_as_vec()))

    @staticmethod
    def _vec_to_params(input_size, hidden_size, latent_size, theta):
        encoder_len = MLP._params_len(input_size, hidden_size, 2 * latent_size)
        encoder_params = MLP._vec_to_params(
            input_size, hidden_size, 2 * latent_size, theta[:encoder_len])
        decoder_params = MLP._vec_to_params(
            latent_size, hidden_size, 2 * input_size, theta[encoder_len:])
        encoder = MLP(input_size, hidden_size, 2 * latent_size, params=encoder_params)
        decoder = MLP(latent_size, hidden_size, 2 * input_size, params=decoder_params)
        return (encoder, decoder)

    @staticmethod
    def _loss(encoder, decoder, X):
        latent_size, input_size = encoder.n_out / 2, decoder.n_out / 2

        encoded = encoder.pred(X)
        mu_encoded = encoded[:latent_size, :]
        log_sigma_encoded = encoded[latent_size:, :]

        eps = randn(latent_size, X.shape[1])
        Z = mu_encoded + exp(log_sigma_encoded) * eps

        decoded = decoder.pred(Z)
        mu_decoded = decoded[:input_size, :]
        log_sigma_decoded = decoded[input_size:, :]

        logpxz = np.sum(-(0.5 * log(2 * np.pi) + log_sigma_decoded) -
                        0.5 * ((X - mu_decoded) / np.exp(log_sigma_decoded)) ** 2)
        kl = 0.5 * np.sum(1 + 2 * log_sigma_encoded -
                          mu_encoded ** 2 - exp(2 * log_sigma_encoded))
        # print("kl={} / logpxz={}".format(kl, logpxz))
        lowerbound = logpxz + kl
        return lowerbound

    def __init__(self, input_size, hidden_size, latent_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encoder = MLP(input_size, hidden_size, 2 * latent_size)
        self.decoder = MLP(latent_size, hidden_size, 2 * input_size)

    def params_as_vec(self):
        return VAE._params_to_vec(self.encoder, self.decoder)

    def loss(self, X):
        return VAE._loss(self.encoder, self.decoder, X)

    def fit(self, X, batch_size=200, max_iter=100):
        cls = self.__class__
        vec_to_params = lambda theta: VAE._vec_to_params(
            self.input_size, self.hidden_size, self.latent_size, theta)

        def loss(theta):
            encoder, decoder = vec_to_params(theta)
            return VAE._loss(encoder, decoder, X)

        alpha, mu = .001, 0.
        theta = self.params_as_vec()
        theta_update = zeros_like(theta)
        batch_index = 0
        for cur_iter in xrange(max_iter):
            grad = numerical_grad(loss, theta)
            # print("|g|^2 = {}".format((grad ** 2).sum()))
            theta_update = alpha * grad + mu * theta_update
            theta += theta_update

            encoder, decoder = vec_to_params(theta)
            print("Loss: {}".format(cls._loss(encoder, decoder, X)))
        self.params = vec_to_params(theta)


class TestVAE(unittest.TestCase):
    def test_params_to_vec(self):
        vae = VAE(10, 5, 2)
        as_vec = vae.params_as_vec()
        enc_as_vec = vae.encoder.params_as_vec()
        dec_as_vec = vae.decoder.params_as_vec()
        self.assertEqual(len(enc_as_vec) + len(dec_as_vec), len(as_vec))
        self.assertTrue(all(enc_as_vec == as_vec[:len(enc_as_vec)]))


class VAE2(object):
    def __init__(self, input_size, hidden_size, latent_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self._initParams()

        # Specify decoder p(v | eps):
        G, W_mu, W_cov = T.dmatrices('G', 'W_mu', 'W_cov')
        b_mu, b_cov, eps = T.dcols('b_mu', 'b_cov', 'eps')
        Geps = T.nnet.sigmoid(G.dot(eps));
        eps_decoder_mu = W_mu.dot(Geps) + b_mu
        eps_decoder_cov = T.exp(W_cov.dot(Geps) + b_cov)
        self.decoder = theano.function(
            [G, W_mu, b_mu, W_cov, b_cov, eps], [eps_decoder_mu, eps_decoder_cov])

        # Specify encoder q(eps | v):
        U, U_mu, U_cov = T.dmatrices('U', 'U_mu', 'U_cov')
        c, c_mu, c_cov, v = T.dcols('c', 'c_mu', 'c_cov', 'v')
        Uv = T.nnet.sigmoid(U.dot(v) + c);
        encoder_mu = U_mu.dot(Uv) + c_mu
        encoder_cov = T.exp(U_cov.dot(Uv) + c_cov)
        self.encoder = theano.function(
            [U, c, U_mu, c_mu, U_cov, c_cov, v], [encoder_mu, encoder_cov])

        kl = -0.5 * (T.sum(encoder_cov) - T.sum(T.log(encoder_cov)) \
                     + T.sum(encoder_mu ** 2) - self.latent_size)

        gamma = T.dcol('gamma')
        Gunit = T.nnet.sigmoid(G.dot(encoder_mu + T.sqrt(encoder_cov) * gamma));
        unit_decoder_mu = W_mu.dot(Gunit) + b_mu
        unit_decoder_cov = T.exp(W_cov.dot(Gunit) + b_cov)

        vminus = v - unit_decoder_mu
        pv = -0.5 * self.input_size * T.log(2.0 * pi) \
             -0.5 * T.sum(T.log(unit_decoder_cov)) \
             -0.5 * ((vminus) / unit_decoder_cov).T.dot(vminus)
        joint_params = [G, W_mu, b_mu, W_cov, b_cov] + \
                       [U, c, U_mu, c_mu, U_cov, c_cov]
        lb = (kl + pv).sum()
        self._lowerbound = theano.function(
            joint_params + [gamma, v], [lb] + T.grad(lb, joint_params))

    def _initParams(self, sigma=0.01):
        # Decoder parameters:
        self.G = randn(self.hidden_size, self.latent_size) * sigma
        self.W_mu = randn(self.input_size, self.hidden_size) * sigma
        self.b_mu = randn(self.input_size, 1) * sigma
        self.W_cov = randn(self.input_size, self.hidden_size) * sigma
        self.b_cov = randn(self.input_size, 1) * sigma

        # Encoder parameters:
        self.U = randn(self.hidden_size, self.input_size) * sigma
        self.c = randn(self.hidden_size, 1) * sigma
        self.U_mu = randn(self.latent_size, self.hidden_size) * sigma
        self.c_mu = randn(self.latent_size, 1) * sigma
        self.U_cov = randn(self.latent_size, self.hidden_size) * sigma
        self.c_cov = randn(self.latent_size, 1) * sigma

    def decode(self, eps):
        return self.decoder(
            self.G, self.W_mu, self.b_mu, self.W_cov, self.b_cov,  eps)

    def encode(self, v):
        return self.encoder(
            self.U, self.c, self.U_mu, self.c_mu, self.U_cov, self.c_cov, v)

    def lowerbound(self, gamma, v):
        return self._lowerbound(
            self.G, self.W_mu, self.b_mu, self.W_cov, self.b_cov,
            self.U, self.c, self.U_mu, self.c_mu, self.U_cov, self.c_cov, gamma, v)

    def fit(self, v):
        mu = 0.1
        alpha = 0.00005
        [last_dG, last_dW_mu, last_db_mu, last_dW_cov, last_db_cov,\
         last_dU, last_dc, last_dU_mu, last_dc_mu, last_dU_cov, last_dc_cov] \
            = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for i in xrange(20000):
            gamma = randn(self.latent_size, 1)
            [cost, dG, dW_mu, db_mu, dW_cov, db_cov,
             dU, dc, dU_mu, dc_mu, dU_cov, dc_cov] = self.lowerbound(gamma, v[:, i:i+1])
            print("Lowerbound at iter {}: {}".format(i, cost))

            self.G += mu * alpha * dG + (1 - mu) * last_dG
            self.W_mu += mu * alpha * dW_mu + (1 - mu) * last_dW_mu
            self.b_mu += mu * alpha * db_mu + (1 - mu) * last_db_mu
            self.W_cov += mu * alpha * dW_cov + (1 - mu) * last_dW_cov
            self.b_cov += mu * alpha * db_cov + (1 - mu) * last_db_cov
            self.U += mu * alpha * dU + (1 - mu) * last_dU
            self.c += mu * alpha * dc + (1 - mu) * last_dc
            self.U_mu += mu * alpha * dU_mu + (1 - mu) * last_dU_mu
            self.c_mu += mu * alpha * dc_mu + (1 - mu) * last_dc_mu
            self.U_cov += mu * alpha * dU_cov + (1 - mu) * last_dU_cov
            self.c_cov += mu * alpha * dc_cov + (1 - mu) * last_dc_cov

            last_dG = mu * alpha * dG + (1 - mu) * last_dG
            last_dW_mu = mu * alpha * dW_mu + (1 - mu) * last_dW_mu
            last_db_mu = mu * alpha * db_mu + (1 - mu) * last_db_mu
            last_dW_cov = mu * alpha * dW_cov + (1 - mu) * last_dW_cov
            last_db_cov = mu * alpha * db_cov + (1 - mu) * last_db_cov
            last_dU = mu * alpha * dU + (1 - mu) * last_dU
            last_dc = mu * alpha * dc + (1 - mu) * last_dc
            last_dU_mu = mu * alpha * dU_mu + (1 - mu) * last_dU_mu
            last_dc_mu = mu * alpha * dc_mu + (1 - mu) * last_dc_mu
            last_dU_cov = mu * alpha * dU_cov + (1 - mu) * last_dU_cov
            last_dc_cov = mu * alpha * dc_cov + (1 - mu) * last_dc_cov


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AEVB implementation.')
    _arg = parser.add_argument
    _arg('--unittest', action='store_true', help='Run unittests')
    args = parser.parse_args()

    if args.unittest:
        unittest.main(verbosity=2, module='aevb', exit=False,
                      argv=[sys.argv[0]])

    # mlp = MLP(784, 32, 784)
    # mnist = pickle.load(open('mnist/mnist.pickle', 'rb'))
    # X = mnist['train'].reshape((-1, 784)).T
    # mlp.fit(X, X)
    # pickle.dump(mlp, open('model', 'wb'), -1)

    vae = VAE(10, 10, 2)
    mnist = pickle.load(open('mnist/mnist.pickle', 'rb'))
    X = mnist['train'].reshape((-1, 784)).T[:, :20]
    # print(vae.loss(X))
    vae.fit(randn(10, 20))
    # pickle.dump(mlp, open('model', 'wb'), -1)
