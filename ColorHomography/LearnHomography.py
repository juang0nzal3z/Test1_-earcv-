""" --------------------------------------------------------------------------------------------------
 This file has two category of functions:
 1) Learn the color homography matrix between two color checker images using alternating least square method
 2) Apply the learnt color homography matrix to the target image
 Reference : https://homepages.inf.ed.ac.uk/rbf/PAPERS/hgcic16.pdf
 ----------------------------------------------------------------------------------------------------"""

import numpy as np
from scipy import sparse

def generate_homography(src_img, tar_img):
    max_iter = 100
    tol = 1e-10
    (H, D) = calculate_H_using_ALS(tar_img, src_img, max_iter, tol)
    return H


def calculate_H_using_ALS(p1, p2, max_iter, tol):
    Npx = len(p1)    # Num of data
    N = p1
    D = sparse.eye(Npx, Npx)
    n_it = 0
    ind1 = np.sum((p1 > 0) & (p1 < np.Inf), 0) == Npx
    ind2 = np.sum((p2 > 0) & (p2 < np.Inf), 0) == Npx
    vind = ind1 & ind2
    # TODO: Add a size check for p1 & p2
    while(n_it < max_iter):
        n_it = n_it + 1
        D = solve_D(N, p2)
        P_D = np.dot(D, p1)
        P_X = np.linalg.pinv(P_D[:, vind])
        H = np.dot(P_X, p2[:, vind])
        N = np.dot(P_D, H)
    PD = D
    return H, PD


def solve_D(p, q):
    nPx = len(p)
    nCh = len(p[0])
    d = np.divide(np.dot(np.ones((1, nCh)), np.transpose(p*q)), np.dot(np.ones((1, nCh)), np.transpose(p*p)))
    D = sparse.spdiags(d, 0, nPx, nPx)
    D = sparse.dia_matrix.astype(D, float).toarray()
    return D


def apply_homo(tar, cor_mat, isTarImage):
    img_size = np.shape(tar)
    rgb_tar = np.reshape(tar, [img_size[0]*img_size[1],3])

    if isTarImage:
        corrected = rgb_tar
        corrected = np.dot(corrected, cor_mat)
        corrected = np.reshape(corrected, img_size)
    else:
        corrected = np.dot(rgb_tar, cor_mat)
        corrected = np.reshape(corrected, img_size)
    return corrected.astype(np.uint8)

