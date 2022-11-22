# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:32:51 2022

@author: deshe
"""

import numpy as np
import copy


def CayleyKlein_to_Quat(K):
    # "K" is a complex 2x2 matrix
    q1 = np.imag(K[0][1])
    q2 = np.real(K[0][1])
    q3 = np.imag(K[0][0])
    q4 = np.real(K[0][0])
    q = np.array([q1, q2, q3, q4])
    return q

def Quat_to_CayleyKlein(q):
    # "q" is a normalized 4-vector (quaternion)
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    K = np.array([[complex(q4, q3), complex(q2, q1)], [complex(-q2, q1), complex(q4, -q3)]])
    return K


def DCM_to_Quat(C):
    # "C" is a proper orthonormal transformation matrix
    tr_C = C[0][0] + C[1][1] + C[2][2]
    q1 = 0.5 * np.sqrt(1 - tr_C + 2 * C[0][0])
    q2 = 0.5 * np.sqrt(1 - tr_C + 2 * C[1][1])
    q3 = 0.5 * np.sqrt(1 - tr_C + 2 * C[2][2])
    q4 = 0.5 * np.sqrt(1 + tr_C)
    if max(q1, q2, q3, q4) == q1:
        q2 = (C[0][1] + C[1][0]) / (4 * q1)
        q3 = (C[2][0] + C[0][2]) / (4 * q1)
        q4 = (C[1][2] - C[2][1]) / (4 * q1)
    elif max(q1, q2, q3, q4) == q2:
        q1 = (C[1][0] + C[0][1]) / (4 * q2)
        q3 = (C[1][2] + C[2][1]) / (4 * q2)
        q4 = (C[2][0] - C[0][2]) / (4 * q2)
    elif max(q1, q2, q3, q4) == q3:
        q1 = (C[2][0] + C[0][2]) / (4 * q3)
        q2 = (C[2][1] + C[1][2]) / (4 * q3)
        q4 = (C[0][1] - C[1][0]) / (4 * q3)
    elif max(q1, q2, q3, q4) == q4:
        q1 = (C[1][2] - C[2][1]) / (4 * q4)
        q2 = (C[2][0] - C[0][2]) / (4 * q4)
        q3 = (C[0][1] - C[1][0]) / (4 * q4)
    q = np.array([q1, q2, q3, q4])
    return q


def Quat_to_DCM(q):
    # "q" is a normalized 4-vector (quaternion)
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    part1 = I * (q4 * q4 - q1 * q1 - q2 * q2 - q3 * q3)
    part2 = 2 * np.array([[q1 * q1, q1 * q2, q1 * q3], [q1 * q2, q2 * q2, q2 * q3], [q1 * q3, q2 * q3, q3 * q3]])
    part3 = 2 * q4 * np.array([[0, -q3, q2], [q3, 0, -q1], [-q2, q1, 0]])
    C = part1 + part2 - part3
    return C


def AxisAngle_to_DCM(e, phi):
    # "e" is a normalized 3-vector
    # "phi" is an angle in degrees
    e1 = e[0]
    e2 = e[1]
    e3 = e[2]
    phi_r = np.radians(phi)
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    part1 = I * np.cos(phi_r)
    part2 = np.array([[e1 * e1, e1 * e2, e1 * e3], [e1 * e2, e2 * e2, e2 * e3], [e1 * e3, e2 * e3, e3 * e3]]) * (
                1 - np.cos(phi_r))
    part3 = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]]) * (np.sin(phi_r))
    C = part1 + part2 - part3
    return C


def DCM_to_AxisAngle(C):
    # "C" is a proper orthonormal transformation matrix
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if np.array_equal(C, I):
        return np.array([1, 0, 0]), 0.0
    else:
        phi_r = np.arccos(0.5 * (C[0][0] + C[1][1] + C[2][2] - 1))
        e1 = (C[1][2] - C[2][1]) / (2 * np.sin(phi_r))
        e2 = (C[2][0] - C[0][2]) / (2 * np.sin(phi_r))
        e3 = (C[0][1] - C[1][0]) / (2 * np.sin(phi_r))
        e = np.array([e1, e2, e3])
        phi = np.degrees(phi_r)
        return [e, phi]


def Euler_to_DCM(sequence, theta_1, theta_2, theta_3):
    # "sequence" is a string of three numbers indicating the Euler sequence ('321', etc.); '111', '222', etc. are not valid.
    # "theta_1", "theta_2", and "theta_3" are three angles (in degrees) applied about the axes given by the sequence, in that order.
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    null_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    if not (sequence == '321' or sequence == '123' or sequence == '231' or sequence == '213' or sequence == '312' or
            sequence == '132' or sequence == '121' or sequence == '131' or sequence == '212' or sequence == '232' or
            sequence == '313' or sequence == '323'):
        print('Please input a valid Euler sequence!')
        return null_mat
    else:
        i = 0
        thetas = [np.radians(theta_1), np.radians(theta_2), np.radians(theta_3)]
        C = copy.deepcopy(I)
        for character in sequence:
            Q = copy.deepcopy(I)
            if int(character) == 1:
                Q = np.array(
                    [[1, 0, 0], [0, np.cos(thetas[i]), np.sin(thetas[i])], [0, -np.sin(thetas[i]), np.cos(thetas[i])]])
            elif int(character) == 2:
                Q = np.array(
                    [[np.cos(thetas[i]), 0, -np.sin(thetas[i])], [0, 1, 0], [np.sin(thetas[i]), 0, np.cos(thetas[i])]])
            elif int(character) == 3:
                Q = np.array(
                    [[np.cos(thetas[i]), np.sin(thetas[i]), 0], [-np.sin(thetas[i]), np.cos(thetas[i]), 0], [0, 0, 1]])
            C = np.matmul(Q, C)
            i += 1
        return C


def DCM_to_Euler(sequence, C):
    # "sequence" is a string of three numbers indicating the Euler sequence ('321', etc.); '111', '222', etc. are not valid.
    # "C" is a proper orthonormal transformation matrix
    if not (sequence == '321' or sequence == '123' or sequence == '231' or sequence == '213' or sequence == '312' or
            sequence == '132' or sequence == '121' or sequence == '131' or sequence == '212' or sequence == '232' or
            sequence == '313' or sequence == '323'):
        print('Please input a valid Euler sequence!')
        return [0, 0, 0]
    else:
        theta_1 = 0
        theta_2 = 0
        theta_3 = 0
        if sequence == '121':
            theta_1 = np.arctan(-C[0][1] / C[0][2])
            theta_2 = np.arccos(C[0][0])
            theta_3 = np.arctan(C[1][0] / C[2][0])
        elif sequence == '131':
            theta_1 = np.arctan(C[0][2] / C[0][1])
            theta_2 = np.arccos(C[0][0])
            theta_3 = np.arctan(-C[2][0] / C[1][0])
        elif sequence == '212':
            theta_1 = np.arctan(C[1][0] / C[1][2])
            theta_2 = np.arccos(C[1][1])
            theta_3 = np.arctan(-C[0][1] / C[2][1])
        elif sequence == '232':
            theta_1 = np.arctan(-C[1][2] / C[1][0])
            theta_2 = np.arccos(C[1][1])
            theta_3 = np.arctan(C[2][1] / C[0][1])
        elif sequence == '313':
            theta_1 = np.arctan(-C[2][0] / C[2][1])
            theta_2 = np.arccos(C[2][2])
            theta_3 = np.arctan(C[0][2] / C[1][2])
        elif sequence == '323':
            theta_1 = np.arctan(C[2][1] / C[2][0])
            theta_2 = np.arccos(C[2][2])
            theta_3 = np.arctan(-C[1][2] / C[0][2])
        elif sequence == '123':
            theta_1 = np.arctan(-C[2][1] / C[2][2])
            theta_2 = np.arcsin(C[2][0])
            theta_3 = np.arctan(-C[1][0] / C[0][0])
        elif sequence == '132':
            theta_1 = np.arctan(C[1][2] / C[1][1])
            theta_2 = np.arcsin(-C[1][0])
            theta_3 = np.arctan(C[2][0] / C[0][0])
        elif sequence == '213':
            theta_1 = np.arctan(C[2][0] / C[2][2])
            theta_2 = np.arcsin(-C[2][1])
            theta_3 = np.arctan(C[0][1] / C[1][1])
        elif sequence == '231':
            theta_1 = np.arctan(-C[0][2] / C[0][0])
            theta_2 = np.arcsin(C[0][1])
            theta_3 = np.arctan(-C[2][1] / C[1][1])
        elif sequence == '312':
            theta_1 = np.arctan(-C[1][0] / C[1][1])
            theta_2 = np.arcsin(C[1][2])
            theta_3 = np.arctan(-C[0][2] / C[2][2])
        elif sequence == '321':
            theta_1 = np.arctan(C[0][1] / C[0][0])
            theta_2 = np.arcsin(-C[0][2])
            theta_3 = np.arctan(C[1][2] / C[2][2])
        return [np.degrees(theta_1), np.degrees(theta_2), np.degrees(theta_3)]


def AxisAngle_to_Quat(e, phi):
    # "e" is a normalized 3-vector
    # "phi" is an angle in degrees
    e1 = e[0]
    e2 = e[1]
    e3 = e[2]
    phi_r = np.radians(phi)
    q1 = e1 * np.sin(phi_r / 2)
    q2 = e2 * np.sin(phi_r / 2)
    q3 = e3 * np.sin(phi_r / 2)
    q4 = np.cos(phi_r / 2)
    q = np.array([q1, q2, q3, q4])
    return q


def Quat_to_AxisAngle(q):
    # "q" is a normalized 4-vector (quaternion)
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    phi_r = np.arccos(2 * q4 * q4 - 1)
    e1 = q1 / np.sin(phi_r / 2)
    e2 = q2 / np.sin(phi_r / 2)
    e3 = q3 / np.sin(phi_r / 2)
    e = np.array([e1, e2, e3])
    phi = np.degrees(phi_r)
    return [e, phi]


def AxisAngle_to_Gibbs(e, phi):
    # "e" is a normalized 3-vector
    # "phi" is an angle in degrees
    e1 = e[0]
    e2 = e[1]
    e3 = e[2]
    phi_r = np.radians(phi)
    rho1 = e1 * np.tan(phi_r / 2)
    rho2 = e2 * np.tan(phi_r / 2)
    rho3 = e3 * np.tan(phi_r / 2)
    rho = np.array([rho1, rho2, rho3])
    return rho


def Gibbs_to_AxisAngle(rho):
    # "rho" is a 3-vector
    rho1 = rho[0]
    rho2 = rho[1]
    rho3 = rho[2]
    rho_squared = rho1 * rho1 + rho2 * rho2 + rho3 * rho3
    phi_r = np.arccos((1 - rho_squared) / (1 + rho_squared))
    e1 = rho1 / np.tan(phi_r / 2)
    e2 = rho2 / np.tan(phi_r / 2)
    e3 = rho3 / np.tan(phi_r / 2)
    e = np.array([e1, e2, e3])
    phi = np.degrees(phi_r)
    return [e, phi]


def Quat_to_Gibbs(q):
    # "q" is a normalized 4-vector (quaternion)
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    rho1 = q1 / q4
    rho2 = q2 / q4
    rho3 = q3 / q4
    rho = np.array([rho1, rho2, rho3])
    return rho


def Gibbs_to_Quat(rho):
    # "rho" is a 3-vector
    rho1 = rho[0]
    rho2 = rho[1]
    rho3 = rho[2]
    rho_squared = rho1 * rho1 + rho2 * rho2 + rho3 * rho3
    q = (1 / np.sqrt(1 + rho_squared)) * np.array([rho1, rho2, rho3, 1])
    return q


def Gibbs_to_DCM(rho):
    # "rho" is a 3-vector
    rho1 = rho[0]
    rho2 = rho[1]
    rho3 = rho[2]
    rho_squared = rho1 * rho1 + rho2 * rho2 + rho3 * rho3
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    part1 = (1 - rho_squared) * I
    part2 = 2 * np.array([[rho1 * rho1, rho1 * rho2, rho1 * rho3], [rho1 * rho2, rho2 * rho2, rho2 * rho3],
                          [rho1 * rho3, rho2 * rho3, rho3 * rho3]])
    part3 = 2 * np.array([[0, -rho3, rho2], [rho3, 0, -rho1], [-rho2, rho1, 0]])
    C = (1 / (1 + rho_squared)) * (part1 + part2 - part3)
    return C


def Quat_to_MRP(q):
    # "q" is a normalized 4-vector (quaternion)
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    sigma = (1 / (1 + q4)) * np.array([q1, q2, q3])
    return sigma


def MRP_to_Quat(sigma):
    # "sigma" is a 3-vector
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]
    sigma_squared = sigma1 * sigma1 + sigma2 * sigma2 + sigma3 * sigma3
    q = (1 / (1 + sigma_squared)) * np.array([2 * sigma1, 2 * sigma2, 2 * sigma3, 1 - sigma_squared])
    return q


def MRP_to_Gibbs(sigma):
    # "sigma" is a 3-vector
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]
    sigma_squared = sigma1 * sigma1 + sigma2 * sigma2 + sigma3 * sigma3
    rho = (2 / (1 - sigma_squared)) * sigma
    return rho


def AxisAngle_to_MRP(e, phi):
    # "e" is a normalized 3-vector
    # "phi" is an angle in degrees
    phi_r = np.radians(phi)
    sigma = e * np.tan(phi_r / 4)
    return sigma


def MRP_to_AxisAngle(sigma):
    # "sigma" is a 3-vector
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]
    sigma_squared = sigma1 * sigma1 + sigma2 * sigma2 + sigma3 * sigma3
    sigma_fourth = sigma_squared * sigma_squared
    phi_r = np.arccos((sigma_fourth - 6 * sigma_squared + 1) / (sigma_fourth + 2 * sigma_squared + 1))
    e = (1 / np.tan(phi_r / 4)) * sigma
    phi = np.degrees(phi_r)
    return [e, phi]


def MRP_to_DCM(sigma):
    # "sigma" is a 3-vector
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]
    sigma_squared = sigma1 * sigma1 + sigma2 * sigma2 + sigma3 * sigma3
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sigma_cross = np.array([[0, -sigma3, sigma2], [sigma3, 0, -sigma1], [-sigma2, sigma1, 0]])
    sigma_cross_squared = np.matmul(sigma_cross, sigma_cross)
    C = I + (1 + sigma_squared) * (8 * sigma_cross_squared - 4 * (1 - sigma_squared) * sigma_cross)
    return C


def ConvertAttitude(type1, attitude1, type2):
    # typedef:
    # type 'DCM' = 3x3 proper orthonormal transformation matrix
    # input np.array([[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]])
    # type 'Euler_ijk' = Euler angles in sequence i-j-k
    # input [theta_1, theta_2, theta_3] in degrees
    # type 'Axis-Angle' = rotate around normalized 3-vector axis e by angle phi
    # input [np.array([e1, e2, e3]), phi] with e normalized and phi in degrees
    # type 'Quaternion' = normalized quaternion with imaginary parts q1, q2, and q3, and real part q4
    # input np.array([q1, q2, q3, q4]), normalized
    # type 'Gibbs vector' = 3-vector with singularity at phi = 180 degrees
    # input np.array([rho1, rho2, rho3])
    # type 'Modified Rodrigues Parameters' = 3-vector with singularity at phi = 360 degrees
    # input np.array([sigma1, sigma2, sigma3])
    # type 'Cayley-Klein parameters' = 2x2 complex matrix with determinant +1
    # input np.array([[complex(q4, q3), complex(q2, q1)], [complex(-q2, q1), complex(q4, -q3)]])
    if type1 == 'DCM':
        if type2[:5] == 'Euler':
            return DCM_to_Euler(type2[-3:], attitude1)
        elif type2 == 'Axis-Angle':
            return DCM_to_AxisAngle(attitude1)
        elif type2 == 'Quaternion':
            return DCM_to_Quat(attitude1)
        elif type2 == 'Gibbs vector':
            return Quat_to_Gibbs(DCM_to_Quat(attitude1))
        elif type2 == 'Modified Rodrigues parameters':
            return Quat_to_MRP(DCM_to_Quat(attitude1))
        elif type2 == 'Cayley-Klein parameters':
            return Quat_to_CayleyKlein(DCM_to_Quat(attitude1))
    elif type1[:5] == 'Euler':
        if type2 == 'DCM':
            return Euler_to_DCM(type1[-3:], attitude1[0], attitude1[1], attitude1[2])
        elif type2 == 'Axis-Angle':
            return DCM_to_AxisAngle(Euler_to_DCM(type1[-3:], attitude1[0], attitude1[1], attitude1[2]))
        elif type2 == 'Quaternion':
            return DCM_to_Quat(Euler_to_DCM(type1[-3:], attitude1[0], attitude1[1], attitude1[2]))
        elif type2 == 'Gibbs vector':
            return Quat_to_Gibbs(DCM_to_Quat(Euler_to_DCM(type1[-3:], attitude1[0], attitude1[1], attitude1[2])))
        elif type2 == 'Modified Rodrigues parameters':
            return Quat_to_MRP(DCM_to_Quat(Euler_to_DCM(type1[-3:], attitude1[0], attitude1[1], attitude1[2])))
        elif type2 == 'Cayley-Klein parameters':
            return Quat_to_CayleyKlein(DCM_to_Quat(Euler_to_DCM(type1[-3:], attitude1[0], attitude1[1], attitude1[2])))
    elif type1 == 'Axis-Angle':
        if type2 == 'DCM':
            return AxisAngle_to_DCM(attitude1[0], attitude1[1])
        elif type2[:5] == 'Euler':
            return DCM_to_Euler(type2[-3:], AxisAngle_to_DCM(attitude1[0], attitude1[1]))
        elif type2 == 'Quaternion':
            return AxisAngle_to_Quat(attitude1[0], attitude1[1])
        elif type2 == 'Gibbs vector':
            return AxisAngle_to_Gibbs(attitude1[0], attitude1[1])
        elif type2 == 'Modified Rodrigues parameters':
            return AxisAngle_to_MRP(attitude1[0], attitude1[1])
        elif type2 == 'Cayley-Klein parameters':
            return Quat_to_CayleyKlein(AxisAngle_to_Quat(attitude1[0], attitude1[1]))
    elif type1 == 'Quaternion':
        if type2 == 'DCM':
            return Quat_to_DCM(attitude1)
        elif type2[:5] == 'Euler':
            return DCM_to_Euler(type2[-3:], Quat_to_DCM(attitude1))
        elif type2 == 'Axis-Angle':
            return Quat_to_AxisAngle(attitude1)
        elif type2 == 'Gibbs vector':
            return Quat_to_Gibbs(attitude1)
        elif type2 == 'Modified Rodrigues parameters':
            return Quat_to_MRP(attitude1)
        elif type2 == 'Cayley-Klein parameters':
            return Quat_to_CayleyKlein(attitude1)
    elif type1 == 'Gibbs vector':
        if type2 == 'DCM':
            return Gibbs_to_DCM(attitude1)
        elif type2[:5] == 'Euler':
            return DCM_to_Euler(type2[-3:], Gibbs_to_DCM(attitude1))
        elif type2 == 'Axis-Angle':
            return Gibbs_to_AxisAngle(attitude1)
        elif type2 == 'Quaternion':
            return Gibbs_to_Quat(attitude1)
        elif type2 == 'Modified Rodrigues parameters':
            return Quat_to_MRP(Gibbs_to_Quat(attitude1))
        elif type2 == 'Cayley-Klein parameters':
            return Quat_to_CayleyKlein(Gibbs_to_Quat(attitude1))
    elif type1 == 'Modified Rodrigues parameters':
        if type2 == 'DCM':
            return MRP_to_DCM(attitude1)
        elif type2[:5] == 'Euler':
            return DCM_to_Euler(type2[-3:], MRP_to_DCM(attitude1))
        elif type2 == 'Axis-Angle':
            return MRP_to_AxisAngle(attitude1)
        elif type2 == 'Quaternion':
            return MRP_to_Quat(attitude1)
        elif type2 == 'Gibbs vector':
            return MRP_to_Gibbs(attitude1)
        elif type2 == 'Cayley-Klein parameters':
            return Quat_to_CayleyKlein(MRP_to_Quat(attitude1))
    elif type1 == 'Cayley-Klein parameters':
        if type2 == 'DCM':
            return Quat_to_DCM(CayleyKlein_to_Quat(attitude1))
        elif type2[:5] == 'Euler':
            return DCM_to_Euler(type2[-3:], Quat_to_DCM(CayleyKlein_to_Quat(attitude1)))
        elif type2 == 'Axis-Angle':
            return Quat_to_AxisAngle(CayleyKlein_to_Quat(attitude1))
        elif type2 == 'Quaternion':
            return CayleyKlein_to_Quat(attitude1)
        elif type2 == 'Gibbs vector':
            return Quat_to_Gibbs(CayleyKlein_to_Quat(attitude1))
        elif type2 == 'Modified Rodrigues parameters':
            return Quat_to_MRP(CayleyKlein_to_Quat(attitude1))
