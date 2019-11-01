#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss function for VDN
"""
import math
import torch
from scipy import special


def get_loss(x, y, Xi, p, mu, m, alpha, beta, epsilon, clamp=True):
    if clamp:
        log_max = math.log(1e4)
        log_min = math.log(1e-8)
        m.clamp_(min=log_min, max=log_max)
        alpha.clamp_(min=log_min, max=log_max)
        beta.clamp_(min=log_min, max=log_max)
    m = torch.exp(m)
    alpha = torch.exp(alpha)
    beta = torch.exp(beta)
    loglikelihood = calculate_loglikelihood(mu, m, alpha, beta)
    kl_z, term = dkl_z(mu, epsilon, x, y, m)
    kl_sigma = dkl_sigma(alpha, beta, p, Xi)
    loss = -loglikelihood + kl_z + kl_sigma
    return loss, loglikelihood, kl_z, kl_sigma


def calculate_loglikelihood(mu, m, alpha, beta):
    term1 = 0.5 * (alpha / beta * (mu ** 2 + m))
    loglikelihood = -0.5 * math.log(2 * math.pi) - torch.mean(0.5 * (torch.log(beta) - torch.digamma(alpha)) + term1)
    return loglikelihood


def dkl_z(mu, epsilon, x, y, m):
    error_ = y - x
    term1 = 0.5 * (mu - error_) ** 2 / (epsilon ** 2)
    dkl = 0.5 * ((m / (epsilon ** 2)) - torch.log(m / (epsilon ** 2)) - 1)
    return torch.mean(term1 + dkl), term1


def dkl_sigma(alpha, beta, p, Xi):
    first_term = (alpha - (p ** 2 / 2) + 1) * torch.digamma(alpha)
    second_term = (special.gammaln((p ** 2 / 2) - 1) - torch.lgamma(alpha))
    third_term = ((p ** 2 / 2) - 1) * (torch.log(beta) - torch.log((p ** 2 / 2) * Xi))
    fourth_term = alpha * ((((p ** 2) * Xi) / (2 * beta)) - 1)
    kl_sigma = first_term + second_term + third_term + fourth_term
    return torch.mean(kl_sigma)


if __name__ == '__main__':
    """ random testing of loss function """
    out_denoise = torch.ones(2, 4, 2)
    out_sigma = torch.ones(2, 4, 2)
    im_noisy = torch.ones(2, 2, 2)
    im_gt = torch.ones(2, 2, 2) * 1.5
    sigmaMap = torch.ones(2, 2, 2)
    eps2 = 1
    loss, loglikelihood, kl_z, kl_sigma = get_loss(im_gt, im_noisy, sigmaMap, 7, out_denoise[:, :2, ],
                                                   out_denoise[:, :2, ], out_sigma[:, :2, ], out_sigma[:, 2:, ], eps2)
