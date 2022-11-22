# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:32:49 2022

@author: deshe
"""

import numpy as np
import copy


def cones(b1,b2,r1,r2,b):
    a1 = (np.dot(b,b1-b2*np.dot(r1,r2))/(1-np.dot(r1,r2)**2))
    a2 = (np.dot(b,b2-b1*np.dot(r1,r2))/(1-np.dot(r1,r2)**2))
    a3 = (np.dot(b,np.cross(b1,b2))/(1-np.dot(r1,r2)**2))
    r = a1*r1+a2*r2+a3*np.cross(r1,r2)
    return r


def cones_opt(b1,b2,r1,r2,b,alpha1,alpha2):
    # co-planar condition needed in optimal    
    thetar = np.arccos(np.dot(r1,r2))
    thetab = np.arccos(np.dot(b1,b2))
    theta1 = np.arcsin(alpha2*np.sin(thetab-thetar))
    theta2 = thetab - thetar - theta1
    b1_opt = (b2*np.sin(theta1)+b1*np.sin(thetar+theta2))/np.sin(thetab)
    b2_opt = (b1*np.sin(theta2)+b2*np.sin(thetar+theta1))/np.sin(thetab)
    a1 = (np.dot(b,b1_opt-b2_opt*np.dot(r1,r2))/(1-np.dot(r1,r2)**2))
    a2 = (np.dot(b,b2_opt-b1_opt*np.dot(r1,r2))/(1-np.dot(r1,r2)**2))
    a3 = (np.dot(b,np.cross(b1_opt,b2_opt))/(1-np.dot(r1,r2)**2))
    r = a1*r1+a2*r2+a3*np.cross(r1,r2)
    return r

def triad(b1,b2,r1,r2):
    b3 = np.cross(b1,b2)/np.linalg.norm(np.cross(b1,b2))
    r3 = np.cross(r1,r2)/np.linalg.norm(np.cross(r1,r2))
    C = np.matmul(np.array([b1,b3,np.cross(b1,b3)]).T,np.array([r1,r3,np.cross(r1,r3)]))
    return C

def triad_opt(b1,b2,r1,r2,alpha1,alpha2):
    # co-planar condition needed in optimal along with matrix inverse, unlike regular triad with transpose
    thetab = np.arccos(np.dot(b1,b2))
    thetar = np.arccos(np.dot(r1,r2))
    theta1 = np.arcsin(alpha2*np.sin(thetab-thetar))
    theta2 = thetab - thetar - theta1
    b1_opt = (b2*np.sin(theta1) + b1*np.sin(thetar+theta2))/np.sin(thetab)
    b2_opt = (b1*np.sin(theta2) + b2*np.sin(thetar+theta1))/np.sin(thetab)
    b1_opt = b1_opt / np.linalg.norm(b1_opt)
    b2_opt = b2_opt / np.linalg.norm(b2_opt)
    r_mat = np.array([r1,r2,np.cross(r1,r2)/np.linalg.norm(np.cross(r1,r2))]).T
    b_mat = np.array([b1_opt,b2_opt,np.cross(b1_opt,b2_opt)/np.linalg.norm(np.cross(b1_opt,b2_opt))]).T
    C = np.matmul(b_mat,np.linalg.inv(r_mat))
    return C

def q_method(b,r,alpha):
    #create B
    B = createB(b,r,alpha)
    # create K
    K = createK(b,r,alpha)
    # compute eigenvectors and eigenvalues of K
    lamda,lambda_vec = np.linalg.eig(K)
    # get max value of eigenvalue and associated eigenvector
    max_lamda_i= np.argmax(lamda)
    q_opt = lambda_vec[:,max_lamda_i]
    q_opt = q_opt / np.linalg.norm(q_opt)
    return q_opt

def svd_method(b,r,alpha):
    #create B
    B = createB(b,r,alpha)
    #get U V and C
    U,S,VT = np.linalg.svd(B)
    C = np.matmul(U,np.matmul(np.diag((1,1,np.linalg.det(U)*np.linalg.det(VT.T))),VT))
    return C

def esoq_method(b,r,alpha):
    #create B
    B = createB(b,r,alpha)
    #create K
    K = createK(b,r,alpha)

    # compute eiganvalues of K
    lamda, eigvec = np.linalg.eig(K)

    # create H matrix
    H = K - max(lamda)*np.identity(4)

    qkk = np.zeros(4)
    # getting q_11, q_22, q_33, q_44
    for k in range(4):
        H_temp = copy.deepcopy(H)
        H_temp = np.delete(H_temp,k,0)
        H_temp = np.delete(H_temp,k,1)
        qkk[k] = ((-1)**((2*k)+2))*np.linalg.det(H_temp)

    # finding max q[k][k] = q[m][m], q[m][m] is q_opt
    m = np.argmax(qkk)

    #creating qm
    qm = np.zeros(4)
    for i in range(4):
        H_temp = copy.deepcopy(H)
        # delete row m
        H_temp = np.delete(H_temp,m,0)
        # delete column i
        H_temp = np.delete(H_temp,i,1)
        #creating q[m][i], +2 is added to exponent because python indexes starting at 0
        qm[i] = ((-1)**(m+1+i+1))*np.linalg.det(H_temp)

    # normalizing q[m][m] to get q_optimal
    q_opt = qm/np.linalg.norm(qm)
    return q_opt


def esoq2_method(b,r,alpha):
    #create B
    B = createB(b,r,alpha)

    #create z
    z = np.zeros(3)
    for i in range(len(alpha)):
        z += alpha[i]*np.cross(b[i],r[i])

    #create K
    K = createK(b,r,alpha)

    # compute eiganvalues of K
    lamda, eigvec = np.linalg.eig(K)

    # calculating M matrix
    S = B + B.T - (np.trace(B)+max(abs(lamda)))*np.identity(3)
    M = (np.trace(B)-max(abs(lamda)))*S - np.outer(z,z)

    # creating ek
    e1 = np.cross(M[1],M[2])
    e2 = np.cross(M[2],M[0])
    e3 = np.cross(M[0],M[1])
    e_arr = [np.linalg.norm(e1),np.linalg.norm(e2),np.linalg.norm(e3)]
    e_vec = [e1, e2, e3]
    # picking max e, for ek
    k = np.argmax(e_arr)
    ek = copy.deepcopy(e_vec[k])

    #calculating q_bar after choosing furthest from 0 ek
    q_bar = np.concatenate(((max(abs(lamda))-np.trace(B))*ek,np.array([np.dot(z,ek)])))
    
    # normalizing q_bar to get q_optimal
    q_opt = q_bar/np.linalg.norm(q_bar)
    return q_opt


def olae_method(b,r,alpha):
    v = np.zeros(3)
    M = np.zeros((3,3))
    B = createB(b, r, alpha)

    # adding in Mw to M
    M = (1/2)*(B+B.T)-(np.trace(B)+1)*np.identity(3)
    for k in range(len(b)):
        s = b[k]+r[k]
        d = b[k]-r[k]
        v += alpha[k]*np.cross(r[k],b[k])
        M += (1/2)*alpha[k]*(np.outer(b[k],b[k])+np.outer(r[k],r[k]))

    # creating rho
    rho = np.matmul(np.linalg.inv(M),v)
    q = np.concatenate((rho,np.array([1])))
    q_opt = q/np.linalg.norm(q)
    return q_opt


def createB(b,r,alpha):
    B = np.zeros((3,3))
    for i in range(len(b)):
        B += alpha[i]*np.outer(b[i],r[i])
    return B

def createK(b,r,alpha):
    B = createB(b, r, alpha)
    K1 = np.subtract(np.add(B, B.T), np.multiply(np.trace(B), np.identity(3)))
    K2 = np.zeros(3)
    for i in range(len(b)):
        K2 = np.add(K2,np.multiply(np.cross(b[i],r[i]),alpha[i]))
    K3 = np.trace(B)
    K_temp = copy.deepcopy(K1)
    K_temp = np.concatenate((K_temp,np.array([K2])))
    K = np.zeros((4,4))
    K[0] = np.concatenate((K_temp[0],np.array([K2[0]])))
    K[1] = np.concatenate((K_temp[1],np.array([K2[1]])))
    K[2] = np.concatenate((K_temp[2],np.array([K2[2]])))
    K[3] = np.concatenate((K_temp[3],np.array([K3])))
    return K
