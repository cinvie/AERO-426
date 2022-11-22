# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:32:16 2022

@author: deshe
"""

import numpy as np
import AttitudeDeterminationMethods as ADM
import matplotlib.pyplot as plt


def CorrectiveAttMatrix(c,ch):
    dc = np.matmul(c,ch.T)
    e_dc,phi_dc = DCM_to_AxisAngle(dc)
    return phi_dc


def CreateObservation(theta_c,n,RotMat,OA):
    r = []
    b = []
    alpha = []
    beta = []
    for i in range(n):
        axis = rotate_axis_rand(OA)
        axis = axis/np.linalg.norm(axis)
        bi = np.matmul(AxisAngle_to_RotationMatrix(axis,np.random.uniform(0,theta_c)),OA)
        bi = bi/np.linalg.norm(bi)
        axis2 = rotate_axis_rand(bi)
        axis2 = axis2/np.linalg.norm(axis2)
        # random sigmak from 3 arcseconds to 0.5 degrees
        sigmak = np.random.uniform(0.000833333,0.5)
        beta.append(np.deg2rad(sigmak) ** 2)
        phi = np.random.normal(0,sigmak)
        b_noisy = np.matmul(AxisAngle_to_RotationMatrix(axis2,phi),bi)
        b.append(b_noisy / np.linalg.norm(b_noisy))
        r.append(np.matmul(RotMat,bi) / np.linalg.norm(np.matmul(RotMat,bi)))
    for i in range(n):
        betas=0
        for j in range(len(beta)):
            betas += 1/beta[j]
        alpha.append(1/(beta[i]*betas))
    return b,r,alpha


def AxisAngle_to_RotationMatrix(e, phi):
    # e is principal axis (normalized)
    # phi is an angle in deg
    phi_rad = np.radians(phi)
    R = np.identity(3)*np.cos(phi_rad)+np.array([[e[0]*e[0],e[0]*e[1],e[0]*e[2]], [e[0]*e[1],e[1]*e[1],e[1]*e[2]],[e[0]*e[2],e[1]*e[2],e[2]*e[2]]])*(1-np.cos(phi_rad))+np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]])*(np.sin(phi_rad))
    return R


def rotate_axis_rand(a):
    b = np.array([-a[1],a[0],0])
    b = b/np.linalg.norm(b)
    c = np.matmul(AxisAngle_to_RotationMatrix(a,np.random.uniform(0,360)),b)
    return c


def DCM_to_AxisAngle(C):
    # "C" is a proper orthonormal transformation matrix
    if np.array_equal(C, np.identity(3)):
        return np.array([1, 0, 0]), 0.0
    else:
        phi_r = np.arccos(0.5 * (np.trace(C)- 1))
        e1 = (C[1][2] - C[2][1]) / (2 * np.sin(phi_r))
        e2 = (C[2][0] - C[0][2]) / (2 * np.sin(phi_r))
        e3 = (C[0][1] - C[1][0]) / (2 * np.sin(phi_r))
        e = np.array([e1, e2, e3])
        phi = np.degrees(phi_r)
        return e, phi


def Quat_to_AxisAngle(q):
    # "q" is a normalized 4-vector (quaternion)
    phi_r = np.arccos(2 * q[3] * q[3] - 1)
    e1 = q[0] / np.sin(phi_r / 2)
    e2 = q[1] / np.sin(phi_r / 2)
    e3 = q[2] / np.sin(phi_r / 2)
    e = np.array([e1, e2, e3]) / np.linalg.norm(np.array([e1, e2, e3]))
    phi = np.degrees(phi_r)
    return e, phi


def Quat_to_DCM(q):
    #q is a normalized 4-vector (quaternion)
    p1 = np.identity(3)*(q[3]*q[3]-q[0]*q[0]-q[1]*q[1]-q[2]*q[2])
    p2 = 2*np.array([[q[0]*q[0],q[0]*q[1],q[0]*q[2]],[q[0]*q[1],q[1]*q[1],q[1]*q[2]],[q[0]*q[2],q[1]*q[2],q[2]*q[2]]])
    p3 = 2*q[3]*np.array([[0,-q[2],q[1]],[q[2],0,-q[0]],[-q[1],q[0],0]])
    C = p1+p2-p3
    return C


#Field of view in degrees
FOV_deg = 15

#Cone angle 
Cone_ang_deg = FOV_deg/2

# angular velocity given in rad/s
w = 0.2
# time step
dt = 2*np.pi/(10*abs(w))

# quaternion initial attitude
qo = np.array([0.2, -0.4, -0.9, 0.1])
qo = qo/np.linalg.norm(qo)

OA = np.array([0.5, -0.5, 0.5])
e,phi = Quat_to_AxisAngle(qo)
OA = np.matmul(AxisAngle_to_RotationMatrix(e,phi),OA/np.linalg.norm(OA))

rot_axis = np.matmul(AxisAngle_to_RotationMatrix(e,phi),np.array([0,0,1]))
R = AxisAngle_to_RotationMatrix(e,phi)
t_vec_n2 = []
t_vec_nn2 = []
phi_triad_all = []
phi_triad_opt_all = []
phi_cones_all = []
phi_cones_opt_all = []
phi_q_method_all = []
phi_svd_method_all = []
phi_esoq_method_all = []
phi_esoq2_method_all = []
phi_olae_method_all = []

for i in range(1000):
    n = np.random.randint(2,10)
    b,r,alpha = CreateObservation(Cone_ang_deg,n,R,OA)
    if n == 2:
        C_triad = ADM.triad(b[0],b[1],r[0],r[1])
        C_triad_opt = ADM.triad_opt(b[0],b[1],r[0],r[1],alpha[0],alpha[1])
        Z = ADM.cones(b[0],b[1],r[0],r[1],np.array([0,0,1]))
        Z_opt = ADM.cones_opt(b[0],b[1],r[0],r[1],np.array([0,0,1]),alpha[0],alpha[1])
        X = ADM.cones(b[0],b[1],r[0],r[1],np.array([1,0,0]))
        X_opt = ADM.cones_opt(b[0],b[1],r[0],r[1],np.array([1,0,0]),alpha[0],alpha[1])
        X = X/np.linalg.norm(X)
        Z = Z/np.linalg.norm(Z)
        X_opt = X_opt/np.linalg.norm(X_opt)
        Z_opt = Z_opt/np.linalg.norm(Z_opt)
        C_cones = np.array([X,np.cross(Z,X),Z])
        C_cones_opt = np.array([X_opt,np.cross(Z_opt,X_opt),Z_opt])
        phi_triad = CorrectiveAttMatrix(R.T,C_triad)
        phi_triad_opt = CorrectiveAttMatrix(R.T,C_triad_opt)
        phi_cones = CorrectiveAttMatrix(R.T,C_cones)
        phi_cones_opt = CorrectiveAttMatrix(R.T,C_cones_opt)
        phi_triad_all.append(phi_triad)
        phi_triad_opt_all.append(phi_triad_opt)
        phi_cones_all.append(phi_cones)
        phi_cones_opt_all.append(phi_cones_opt)
        t_vec_n2.append(dt * w * i)
    else: 
        q_opt_q_method = ADM.q_method(b,r,alpha)
        C_svd_method = ADM.svd_method(b,r,alpha)
        q_opt_esoq_method = ADM.esoq_method(b,r,alpha)
        q_opt_esoq2_method = ADM.esoq2_method(b,r,alpha)
        q_opt_olae_method = ADM.olae_method(b,r,alpha)

        # getting all phi values
        phi_q_method = CorrectiveAttMatrix(R.T,Quat_to_DCM(q_opt_q_method))
        phi_svd_method = CorrectiveAttMatrix(R.T,C_svd_method)
        phi_esoq_method = CorrectiveAttMatrix(R.T,Quat_to_DCM(q_opt_esoq_method))
        phi_esoq2_method = CorrectiveAttMatrix(R.T,Quat_to_DCM(q_opt_esoq2_method))
        phi_olae_method = CorrectiveAttMatrix(R.T,Quat_to_DCM(q_opt_olae_method))

        # appending phi values 
        phi_q_method_all.append(phi_q_method)
        phi_svd_method_all.append(phi_svd_method)
        phi_esoq_method_all.append(phi_esoq_method)
        phi_esoq2_method_all.append(phi_esoq2_method)
        phi_olae_method_all.append(phi_olae_method)
        t_vec_nn2.append(dt * w * i)
    R = np.matmul(AxisAngle_to_RotationMatrix(rot_axis,w*dt*(180/np.pi)),R,AxisAngle_to_RotationMatrix(rot_axis,w*dt*(180/np.pi)).T)
    OA = np.matmul(AxisAngle_to_RotationMatrix(rot_axis,w*dt*(180/np.pi)),OA)

fig = plt.figure()
plt.plot(t_vec_n2,phi_triad_all,label='TRIAD')
plt.plot(t_vec_n2,phi_triad_opt_all,label='Optimal TRIAD')
plt.plot(t_vec_n2,phi_cones_all,label='CONES')
plt.plot(t_vec_n2,phi_cones_opt_all,label='Optimal CONES')
plt.plot(t_vec_nn2,phi_q_method_all,label='Q')
plt.plot(t_vec_nn2,phi_svd_method_all,label='SVD')
plt.plot(t_vec_nn2,phi_esoq_method_all,label='ESOQ')
plt.plot(t_vec_nn2,phi_esoq2_method_all,label='ESOQ2')
plt.plot(t_vec_nn2,phi_olae_method_all,label='OLAE')
plt.title("Single point attitude determination methods errors")
plt.xlabel("time [s]")
plt.ylabel("Error [deg]")
plt.grid()
plt.legend()
plt.show()

fig1,ax2= plt.subplots(2,5)
ax2[0][0].hist(phi_triad_all,label='TRIAD')
ax2[0][0].set_title('Ideal TRIAD')
ax2[0][1].hist(phi_triad_opt_all,label='Optimal TRIAD')
ax2[0][1].set_title('Optimal TRIAD')
ax2[0][2].hist(phi_cones_all,label='CONES')
ax2[0][2].set_title('Ideal CONES')
ax2[0][3].hist(phi_cones_opt_all,label='Optimal CONES')
ax2[0][3].set_title('Optimal CONES')
ax2[0][4].hist(phi_q_method_all,label='Q')
ax2[0][4].set_title('Q method')
ax2[1][0].hist(phi_svd_method_all,label='SVD')
ax2[1][0].set_title('SVD')
ax2[1][1].hist(phi_esoq_method_all,label='ESOQ')
ax2[1][1].set_title('ESOQ')
ax2[1][2].hist(phi_esoq2_method_all,label='ESOQ2')
ax2[1][2].set_title('ESOQ2')
ax2[1][3].hist(phi_olae_method_all,label='OLAE')
ax2[1][3].set_title('OLAE')
ax2[-1][-1].axis('off')
plt.suptitle("Single point attitude determination methods errors histogram")
for i in range(2):
    for j in range(5):
        ax2[i][j].set_ylabel('Count')
        ax2[i][j].set_xlabel('Error (deg)')
        ax2[i][j].grid()
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.show()