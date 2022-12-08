# importing all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, lsqr

def wls_decompositon(IN, Lambda=0.01, Alpha=0.9):
    
    L = np.log(IN+1e-22)        # Source image for the affinity matrix. log_e(IN)
    smallNum = 1e-6
    height, width = IN.shape
    k = height * width

    # Compute affinities between adjacent pixels based on gradients of L
    dy = np.diff(L, n=1, axis=0)   # axis=0 is vertical direction

    dy = -Lambda/(np.abs(dy)**Alpha + smallNum)
    dy = np.pad(dy, ((0,1),(0,0)), 'constant')    # add zeros row
    dy = dy.flatten(order='F')

    dx = np.diff(L, n=1, axis=1)

    dx = -Lambda/(np.abs(dx)**Alpha + smallNum)
    dx = np.pad(dx, ((0,0),(0,1)), 'constant')    # add zeros col 
    dx = dx.flatten(order='F')
    
    # Construct a five-point spatially inhomogeneous Laplacian matrix
    B = np.concatenate([[dx], [dy]], axis=0)
    d = np.array([-height,  -1])

    A = spdiags(B, d, k, k) 

    e = dx 
    w = np.pad(dx, (height, 0), 'constant'); w = w[0:-height]
    s = dy
    n = np.pad(dy, (1, 0), 'constant'); n = n[0:-1]

    D = 1.0 - (e + w + s + n)

    A = A + A.transpose() + spdiags(D, 0, k, k)

    # Solve using a least square solver
    OUT = spsolve(A, IN.flatten(order='F'))
    return np.reshape(OUT, (height, width), order='F') 

def reflectance_scaling(reflectance, illuminace):

    r_R = 0.5
    def compare_func(r, i, m):    
        return r * (i/m)**r_R if i > m else r 

    reflectance_scaling_fun = np.vectorize(compare_func)
    mean_I = np.mean(illuminace)
    result = reflectance_scaling_fun(reflectance, illuminace, mean_I)
    return result

def sigmoid_scaling_func(v_, mean_i_, max_i_):
    r = 1.0 - mean_i_/max_i_    
    fv = lambda v : r * ( 1/(1+np.exp(-1.0*(v - mean_i_))) - 0.5 )
    
    fv_k_ = [fv(vk) for vk in v_]
    return fv_k_

def generate_illuminations(illuminace, inv_illuminace):

    inv_illuminace /= np.max(inv_illuminace)
    mi = np.mean(illuminace)

    maxi = np.max(illuminace)
    v1 = 0.2;    v3 = mi;    v2 = 0.5 * (v1 + v3)
    v5 = 0.8;    v4 = 0.5 * (v3 + v5)
    v = [v1, v2, v3, v4, v5]
    fvk_list = sigmoid_scaling_func(v, mi, maxi)

    I_k = [(1 + fvk) * (illuminace + fvk * inv_illuminace) for fvk in fvk_list]  

    return I_k

def tonemapping(bgr_image, L, R_, Ik_list, FLAG):

    Lk_list = [ np.exp(R_) * Ik for Ik in Ik_list ] 
    L = L + 1e-22 

    rt = 1.0
    b, g, r = cv2.split(bgr_image)
    
    # Restore color image
    if FLAG == False:
        Sk_list = [cv2.merge((Lk*(b/L)**rt, Lk*(g/L)**rt, Lk*(r/L)**rt)) for Lk in Lk_list]
        return Sk_list[2]
    else:  # Weight maps

        Wk_list = []
        for index, Ik in enumerate(Ik_list):
            if index < 3:
                wk = Ik / np.max(Ik)
            else:
                temp = 0.5*(1 - Ik)
                wk = temp / np.max(temp)
            Wk_list.append(wk)

        A = np.zeros_like(Wk_list[0])
        B = np.zeros_like(Wk_list[0])
        for lk, wk in zip(Lk_list, Wk_list):
            A = A + lk * wk 
            B = B + wk

        L_ = (A/B)
        ratio = np.clip(L_/L, 0, 3) # Clip unreasonable values
        b_ = ratio * b
        g_ = ratio * g
        r_ = ratio * r
        out = cv2.merge( ( b_, g_, r_ ) )
        return np.clip(out, 0.0, 1.0)
