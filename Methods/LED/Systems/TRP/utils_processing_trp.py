#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np
import os
import time

# UTILITIES
from ... import Utils as utils


def addResultsSystemTRP(model, results, statistics, testing_mode):
    print("# addResultsSystemTRP() #")
    if "autoencoder" in testing_mode:
        targets_all = results["input_sequence_all"]
        predictions_all = results["input_decoded_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    # Estimate free energy projection on the latent space
    # Estimate free energy projection on the latent space
    # covariance_factor_scale = 10.0
    covariance_factor_scale = 10.0
    gridpoints = 25
    results = utils.calculateFreeEnergyProjection(results,
                                                  covariance_factor_scale,
                                                  gridpoints)
    # latent_range_state_percent = 0.05
    # results = utils.caclulateFreeEnergyProjectionLatentClusters(results, latent_range_state_percent)

    # print("# Estimating Mean Passage Times (MPT) between Latent space clusters.")

    # ## The timescales cannot be captured in the TRP due to the limited data
    # dt                 = results["dt"]
    # dt_msm             = 10*1e-12
    # latent_cluster_mean_passage_times, latent_clustered_trajectories = utils.estimateLatentClusterMFPT(results, dt, dt_msm=dt_msm, dt_save=dt_save)
    # results["latent_cluster_mean_passage_times"] = latent_cluster_mean_passage_times
    return results


def generateXYZfileFromABDnoRotTr(badfile, ref_conf, conffile, lammps_file):
    import numpy as np
    import math, itertools
    from math import cos, sin, sqrt, acos, atan2, fabs, pi

    atomNames = ["N","C","C","C","O","N","C","O","N","C","C","C","C","C","C","O","N","C","C","C","C","C","C","O","C","C","C","O","N","C","C","C","C","C","C","O","N",\
    "C","C","C","C","O","N","C","O","N","C","C","C","C","N","C","C","C","C","C","C","C","O","N","C","C","C","C","C","C","O","N","C","C","C","C","C","N","C","O","N", \
    "C","C","C","O","O","C","O","N","C","C","O","N","C","C","O","N","C","C","C","C","C","O","N","C","C","O","C","O","N","C","C","O","C","O","N","C","C","O","N","C", \
    "C","C","C","N","C","N","N","C","O","N","C","C","C","C","C","O","N","C","C","C","C","C","O","N","C","C","C","C","C","O","N","C","C","O","C","O","O"]

    atomtypes, mass = [], []
    for kk in range(len(atomNames)):
        if atomNames[kk] == "N":
            atomtypes.append(3)
            mass.append(14.01)
        elif atomNames[kk] == "C":
            atomtypes.append(1)
            mass.append(12.01)
        elif atomNames[kk] == "O":
            atomtypes.append(2)
            mass.append(16.0)
    masstotal = np.sum(mass)

    dih = ((0,1,2,3), (1,2,3,4), (1,2,3,5), (3,2,1,6), (0,1,6,7), (0,1,6,8), (1,6,8,9), (6,8,9,10), (8,9,10,11), (9,10,11,12), (9,10,11,13), (6,8,9,14), \
    (8,9,14,15), (8,9,14,16), (9,14,16,17), (14,16,17,18), (16,17,18,19), (17,18,19,20), (18,19,20,21), (19,20,21,22), (20,21,22,23), (20,21,22,24), (17,18,19,25),\
     (14,16,17,26), (16,17,26,27), (16,17,26,28), (17,26,28,29), (26,28,29,30), (28,29,30,31), (28,29,30,32), (29,30,32,33), (26,28,29,34), (28,29,34,35), (28,29,34,36),\
      (29,34,36,37), (34,36,37,38), (36,37,38,39), (37,38,39,40), (38,39,40,41), (38,39,40,42), (34,36,37,43), (36,37,43,44), (36,37,43,45), (37,43,45,46),\
       (43,45,46,47), (45,46,47,48), (46,47,48,49), (47,48,49,50), (48,49,50,51), (49,50,51,52), (50,51,52,53), (51,52,53,54), (52,53,54,55), (46,47,48,56), \
       (43,45,46,57), (45,46,57,58), (45,46,57,59), (46,57,59,60), (57,59,60,61), (59,60,61,62), (60,61,62,63), (60,61,62,64), (57,59,60,65), (59,60,65,66), \
       (59,60,65,67), (60,65,67,68), (65,67,68,69), (67,68,69,70), (68,69,70,71), (69,70,71,72), (70,71,72,73), (65,67,68,74), (67,68,74,75), (67,68,74,76), \
       (68,74,76,77), (74,76,77,78), (76,77,78,79), (77,78,79,80), (77,78,79,81), (74,76,77,82), (76,77,82,83), (76,77,82,84), (77,82,84,85), (82,84,85,86), \
       (84,85,86,87), (84,85,86,88), (85,86,88,89), (86,88,89,90), (88,89,90,91), (88,89,90,92), (89,90,92,93), (90,92,93,94), (92,93,94,95), (89,90,92,96), \
       (90,92,96,97), (92,96,97,98), (92,96,97,99), (96,97,99,100), (97,99,100,101), (99,100,101,102), (97,99,100,103), (99,100,103,104), (99,100,103,105), \
       (100,103,105,106), (103,105,106,107), (105,106,107,108), (103,105,106,109), (105,106,109,110), (105,106,109,111), (106,109,111,112), (109,111,112,113),\
        (111,112,113,114), (111,112,113,115), (112,113,115,116), (113,115,116,117), (115,116,117,118), (116,117,118,119), (117,118,119,120), (118,119,120,121),\
         (119,120,121,122), (119,120,121,123), (113,115,116,124), (115,116,124,125), (115,116,124,126), (116,124,126,127), (124,126,127,128), (126,127,128,129),\
          (116,124,126,130), (124,126,130,131), (126,130,131,132), (126,130,131,133), (130,131,133,134), (131,133,134,135), (133,134,135,136), (130,131,133,137), \
          (131,133,137,138), (133,137,138,139), (133,137,138,140), (137,138,140,141), (138,140,141,142), (140,141,142,143), (137,138,140,144), (138,140,144,145),\
           (140,144,145,146), (140,144,145,147), (144,145,147,148), (145,147,148,149), (147,148,149,150), (145,147,148,151), (147,148,151,152), (147,148,151,153))
    angles = ((0,1,2), (1,2,3), (2,3,4), (2,3,5), (2,1,6), (1,6,7), (1,6,8), (6,8,9), (8,9,10), (9,10,11), (10,11,12), (10,11,13), (8,9,14), (9,14,15), \
    (9,14,16), (14,16,17), (16,17,18), (17,18,19), (18,19,20), (19,20,21), (20,21,22), (21,22,23), (21,22,24), (18,19,25), (16,17,26), (17,26,27), (17,26,28), \
    (26,28,29), (28,29,30), (29,30,31), (29,30,32), (30,32,33), (28,29,34), (29,34,35), (29,34,36), (34,36,37), (36,37,38), (37,38,39), (38,39,40), (39,40,41),\
    (39,40,42), (36,37,43), (37,43,44), (37,43,45), (43,45,46), (45,46,47), (46,47,48), (47,48,49), (48,49,50), (49,50,51), (50,51,52), (51,52,53), (52,53,54), \
    (53,54,55), (47,48,56), (45,46,57), (46,57,58), (46,57,59), (57,59,60), (59,60,61), (60,61,62), (61,62,63), (61,62,64), (59,60,65), (60,65,66), (60,65,67), \
    (65,67,68), (67,68,69), (68,69,70), (69,70,71), (70,71,72), (71,72,73), (67,68,74), (68,74,75), (68,74,76), (74,76,77), (76,77,78), (77,78,79), (78,79,80),\
     (78,79,81), (76,77,82), (77,82,83), (77,82,84), (82,84,85), (84,85,86), (85,86,87), (85,86,88), (86,88,89), (88,89,90), (89,90,91), (89,90,92), (90,92,93), \
     (92,93,94), (93,94,95), (90,92,96), (92,96,97), (96,97,98), (96,97,99), (97,99,100), (99,100,101), (100,101,102), (99,100,103), (100,103,104), (100,103,105), \
     (103,105,106), (105,106,107), (106,107,108), (105,106,109), (106,109,110), (106,109,111), (109,111,112), (111,112,113), (112,113,114), (112,113,115), (113,115,116), \
     (115,116,117), (116,117,118), (117,118,119), (118,119,120), (119,120,121), (120,121,122), (120,121,123), (115,116,124), (116,124,125), (116,124,126), (124,126,127),\
      (126,127,128), (127,128,129), (124,126,130), (126,130,131), (130,131,132), (130,131,133), (131,133,134), (133,134,135), (134,135,136), (131,133,137), (133,137,138),\
       (137,138,139), (137,138,140), (138,140,141), (140,141,142), (141,142,143), (138,140,144), (140,144,145), (144,145,146), (144,145,147), (145,147,148), (147,148,149), \
       (148,149,150), (147,148,151), (148,151,152), (148,151,153))
    bonds = ((0,1), (1,2), (2,3), (3,4), (3,5), (1,6), (6,7), (6,8), (8,9), (9,10), (10,11), (11,12), (11,13), (9,14), (14,15), (14,16), (16,17), (17,18), \
    (18,19), (19,20), (20,21), (21,22), (22,23), (22,24), (19,25), (17,26), (26,27), (26,28), (28,29), (29,30), (30,31), (30,32), (32,33), (29,34), (34,35), \
    (34,36), (36,37), (37,38), (38,39), (39,40), (40,41), (40,42), (37,43), (43,44), (43,45), (45,46), (46,47), (47,48), (48,49), (49,50), (50,51), (51,52), \
    (52,53), (53,54), (54,55), (48,56), (46,57), (57,58), (57,59), (59,60), (60,61), (61,62), (62,63), (62,64), (60,65), (65,66), (65,67), (67,68), (68,69), \
    (69,70), (70,71), (71,72), (72,73), (68,74), (74,75), (74,76), (76,77), (77,78), (78,79), (79,80), (79,81), (77,82), (82,83), (82,84), (84,85), (85,86), \
    (86,87), (86,88), (88,89), (89,90), (90,91), (90,92), (92,93), (93,94), (94,95), (92,96), (96,97), (97,98), (97,99), (99,100), (100,101), (101,102), \
    (100,103), (103,104), (103,105), (105,106), (106,107), (107,108), (106,109), (109,110), (109,111), (111,112), (112,113), (113,114), (113,115), (115,116), \
    (116,117), (117,118), (118,119), (119,120), (120,121), (121,122), (121,123), (116,124), (124,125), (124,126), (126,127), (127,128), (128,129), (126,130), \
    (130,131), (131,132), (131,133), (133,134), (134,135), (135,136), (133,137), (137,138), (138,139), (138,140), (140,141), (141,142), (142,143), (140,144), \
    (144,145), (145,146), (145,147), (147,148), (148,149), (149,150), (148,151), (151,152), (151,153))

    ########################################################
    def find_BA(dd1, dd2, dd3, dd4):

        angleID = -1
        for aa in range(len(angles)):
            if (dd2 == angles[aa][0] and dd3 == angles[aa][1]
                    and dd4 == angles[aa][2]):
                angleID = aa
                break
        if (angleID == -1):
            print("angle not found", dd2, dd3, dd4)
            exit()
        #find bond
        bondID = -1
        for bb in range(len(bonds)):
            if (dd3 == bonds[bb][0] and dd4 == bonds[bb][1]):
                bondID = bb
                break
        if (bondID == -1):
            print("bond not found")
            exit()

        return bondID, angleID

    ########################################################
    def place_atom(atom_a, atom_b, atom_c, angle, torsion, bond):

        #print atom_a, atom_b, atom_c, angle, torsion, bond
        R = bond
        ab = np.subtract(atom_b, atom_a)
        bc = np.subtract(atom_c, atom_b)
        bcn = bc / np.linalg.norm(bc)

        case = 1
        okinsert = False
        while (okinsert == False):
            #case 1
            if (case == 1):
                d = np.array([
                    -R * cos(angle), R * cos(torsion) * sin(angle),
                    R * sin(torsion) * sin(angle)
                ])
                n = np.cross(bcn, ab)
                n = n / np.linalg.norm(n)
                nbc = np.cross(bcn, n)
            #case 2
            elif (case == 2):
                d = np.array([
                    -R * cos(angle), R * cos(torsion) * sin(angle),
                    R * sin(torsion) * sin(angle)
                ])
                n = np.cross(ab, bcn)
                n = n / np.linalg.norm(n)
                nbc = np.cross(bcn, n)
            #case 3
            elif (case == 3):
                d = np.array([
                    -R * cos(angle), R * cos(torsion) * sin(angle),
                    -R * sin(torsion) * sin(angle)
                ])
                n = np.cross(ab, bcn)
                n = n / np.linalg.norm(n)
                nbc = np.cross(bcn, n)
            #case 4
            elif (case == 4):
                d = np.array([
                    -R * cos(angle), R * cos(torsion) * sin(angle),
                    R * sin(torsion) * sin(angle)
                ])
                n = np.cross(ab, bcn)
                n = n / np.linalg.norm(n)
                nbc = np.cross(n, bcn)

            m = np.array([[bcn[0], nbc[0], n[0]], [bcn[1], nbc[1], n[1]],
                          [bcn[2], nbc[2], n[2]]])
            d = m.dot(d)
            atom_d = d + atom_c

            #test dihedral
            r21 = np.subtract(atom_b, atom_a)
            r23 = np.subtract(atom_b, atom_c)
            r43 = np.subtract(atom_d, atom_c)
            n1 = np.cross(r21, r23)
            n2 = np.cross(r23, r43)
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)
            r23 = r23 / np.linalg.norm(r23)
            m = np.cross(n1, r23)
            x = np.dot(n1, n2)
            y = np.dot(m, n2)
            phi = atan2(y, x)

            #test angle
            d12 = np.subtract(atom_b, atom_c)
            d32 = np.subtract(atom_d, atom_c)
            d12 = d12 / np.linalg.norm(d12)
            d32 = d32 / np.linalg.norm(d32)
            cos_theta = np.dot(d12, d32)
            m = np.linalg.norm(np.cross(d12, d32))
            theta = atan2(m, cos_theta)

            if (fabs(theta - angle) < 0.001 and fabs(phi - torsion) < 0.001):
                okinsert = True
            else:
                if (case < 4): case += 1
                else:
                    print("no case found", theta, angle, phi, torsion, atom_d)
                    break
        return atom_d

    ########################################################
    def test_angle(atoms, angleID):
        ii, jj, kk = angles[angleID]
        d12 = np.subtract(atoms[ii], atoms[jj])
        d32 = np.subtract(atoms[kk], atoms[jj])
        d12 = d12 / np.linalg.norm(d12)
        d32 = d32 / np.linalg.norm(d32)
        cos_theta = np.dot(d12, d32)
        m = np.linalg.norm(np.cross(d12, d32))
        theta = atan2(m, cos_theta)

        return theta

    ########################################################
    def test_dihedral(atoms, dihedralID):

        ii, jj, kk, ll = dih[dihedralID]
        r21 = np.subtract(atoms[jj], atoms[ii])
        r23 = np.subtract(atoms[jj], atoms[kk])
        r43 = np.subtract(atoms[ll], atoms[kk])

        n1 = np.cross(r21, r23)
        n2 = np.cross(r23, r43)

        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        r23 = r23 / np.linalg.norm(r23)

        m = np.cross(n1, r23)
        x = np.dot(n1, n2)
        y = np.dot(m, n2)

        phi = atan2(y, x)

        return phi

    ########################################################
    def new_config(CVsB, CVsA, CVsD):

        ang = CVsA[0]

        an = -1.0 * ang
        R1 = np.array([[cos(an), -sin(an), 0.0], [sin(an),
                                                  cos(an), 0.0],
                       [0.0, 0.0, 1.0]])
        R2 = np.array([[1.0, 0.0, 0.0],
                       [0.0, cos(-math.pi / 4), -sin(-math.pi / 4)],
                       [0.0, sin(-math.pi / 4),
                        cos(-math.pi / 4)]])
        R3 = np.array([[cos(-math.pi / 4), 0.0,
                        sin(-math.pi / 4)], [0.0, 1.0, 0.0],
                       [-sin(-math.pi / 4), 0.0,
                        cos(-math.pi / 4)]])

        atoms = np.zeros((154, 3), float)

        ### first 3 atoms ###
        vec01 = [1.0 / sqrt(2), 1.0 / sqrt(2), 0.0]
        vec31 = np.dot(R1, vec01)
        vec01 = np.dot(R2, vec01)
        vec31 = np.dot(R2, vec31)
        vec01 = np.dot(R3, vec01)
        vec31 = np.dot(R3, vec31)

        atoms[0] = [CVsB[0] * vec01[0], CVsB[0] * vec01[1], CVsB[0] * vec01[2]]
        atoms[1] = [0.0, 0.0, 0.0]
        atoms[2] = [CVsB[1] * vec31[0], CVsB[1] * vec31[1], CVsB[1] * vec31[2]]

        ### iteratively all other atoms ###
        for dd in range(len(dih)):
            dd1, dd2, dd3, dd4 = dih[dd]
            bondID, angleID = find_BA(dd1, dd2, dd3, dd4)
            coord = place_atom(atoms[dd1], atoms[dd2], atoms[dd3],
                               CVsA[angleID], CVsD[dd], CVsB[bondID])
            atoms[dd4] = coord

        testBAD = True
        if (testBAD):
            #bonds
            for mm in range(len(bonds)):
                ii, jj = bonds[mm]
                dist = pow(atoms[ii][0] - atoms[jj][0], 2) + pow(
                    atoms[ii][1] - atoms[jj][1], 2) + pow(
                        atoms[ii][2] - atoms[jj][2], 2)
                if (fabs(sqrt(dist) - CVsB[mm]) > 0.0001):
                    print("bond", bonds[mm], CVsB[mm], sqrt(dist), atoms[ii],
                          atoms[jj], "Reading snapshot ", Nfile)
            #angles
            for mm in range(len(angles)):
                acos_theta = test_angle(atoms, mm)
                #print "angle",angles[mm],CVsA[mm]*180/pi,acos_theta*180/pi
                if (fabs(acos_theta - CVsA[mm]) > 0.0001):
                    print("angle", angles[mm], CVsA[mm], acos_theta,
                          "Reading snapshot ", Nfile)
            #dihedrals
            for mm in range(len(dih)):
                acos_theta = test_dihedral(atoms, mm)
                #print "dihedral",dih[mm],CVsD[mm]*180/pi,acos_theta*180/pi
                if (fabs(acos_theta - CVsD[mm]) > 0.0001):
                    print("dihedral", dih[mm], CVsD[mm], acos_theta,
                          "Reading snapshot ", Nfile)

        return atoms

    ########################################################
    def remove_com(conf):
        # calculate center of mass165
        comp = [0.0, 0.0, 0.0]
        for i in range(len(conf)):
            for dim in range(3):
                comp[dim] += mass[i] * conf[i][dim]
        for dim in range(3):
            comp[dim] /= masstotal

        # substract center of mass
        conf_com = np.zeros((len(conf), 3), float)
        for i in range(len(conf)):
            for dim in range(3):
                conf_com[i, dim] = conf[i][dim] - comp[dim]

        return conf_com

    #######################################################################
    def rotationmatrix(coordref, coord):

        assert (coordref.shape[1] == 3)
        assert (coordref.shape == coord.shape)
        correlation_matrix = np.dot(np.transpose(coordref), coord)
        vv, ss, ww = np.linalg.svd(correlation_matrix)
        is_reflection = (np.linalg.det(vv) * np.linalg.det(ww)) < 0.0
        #if is_reflection:
        #print "is_reflection"
        #vv[-1,:] = -vv[-1,:]
        #ss[-1] = -ss[-1]
        #vv[:, -1] = -vv[:, -1]
        rotation = np.dot(vv, ww)

        confnew = []
        for i in range(len(coord)):
            xx = rotation[0][0] * coord[i][0] + rotation[0][1] * coord[i][
                1] + rotation[0][2] * coord[i][2]
            yy = rotation[1][0] * coord[i][0] + rotation[1][1] * coord[i][
                1] + rotation[1][2] * coord[i][2]
            zz = rotation[2][0] * coord[i][0] + rotation[2][1] * coord[i][
                1] + rotation[2][2] * coord[i][2]
            confnew.append((xx, yy, zz))

        return confnew

    ########################################################
    #read reference file
    f = open(ref_conf)
    coord = []
    for line in f:
        if len(line) < 5:
            continue
        elif "Lattice" in line:
            L = [
                float(line[9:13]) / 10.0,
                float(line[9:13]) / 10.0,
                float(line[9:13]) / 10.0
            ]
        else:
            s = line.split()
            tmp = [float(s[1]) / 10.0, float(s[2]) / 10.0, float(s[3]) / 10.0]
            if (len(coord) == 0): ref = tmp
            else:
                for dim in range(3):
                    dif = tmp[dim] - ref[dim]
                    if (dif > L[dim] / 2): tmp[dim] -= L[dim]
                    elif (dif < -L[dim] / 2): tmp[dim] += L[dim]
            coord.append(tmp)

    coordREF = remove_com(coord)

    ########################################################
    outfileC = open(conffile, "w")
    outfileC.close()
    ########################################################
    ##### read CVs #####
    f = open(badfile)
    line = ''
    Nfile = 1
    for line in f:
        if (Nfile % 100000 == 0): print("Reading line ... ", Nfile)

        L = [float(x) for x in line.split()]
        CVsB, CVsA, CVsD = [], [], []
        for l in range(len(L)):
            if l < 153: CVsB.append(L[l])
            elif l < 305: CVsA.append(L[l])
            else: CVsD.append(L[l])

        #print len(CVsB),len(CVsA),len(CVsD)
        conf = new_config(CVsB, CVsA, CVsD)
        conf_com = remove_com(conf)
        confnew = rotationmatrix(coordREF, conf_com)
        #confnew = conf

        #write to file
        outfileC = open(conffile, "a")
        outfileC.write("%d\n" % (len(confnew)))
        outfileC.write(
            'Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3 \n'
        )
        for i in range(len(confnew)):
            #print(atomNames[i],confnew[i][0]*10+10.0,confnew[i][1]*10+10.0,confnew[i][2]*10+10.0)
            outfileC.write(
                "%s%19.9f%17.9f%17.9f\n" %
                (atomNames[i], confnew[i][0] * 10 + 10.0,
                 confnew[i][1] * 10 + 10.0, confnew[i][2] * 10 + 10.0))
        outfileC.close()

        Nfile += 1
    f.close()

    ########################################################
    #write data file for Lammps so bonds are correct in Ovito
    # "dataLammps.txt"
    pdb = open(lammps_file, 'w')

    pdb.write("LAMMPS 'data.' description \n")
    pdb.write("\n")
    pdb.write("      %d atoms\n" % (len(atomNames)))
    pdb.write("      %d bonds\n" % (len(bonds)))
    pdb.write("\n")
    pdb.write("       3 atom types\n")
    pdb.write("       1 bond types\n")
    pdb.write("\n")
    pdb.write("    0.0 %1.2f      xlo xhi\n" % (20.0))
    pdb.write("    0.0 %1.2f      ylo yhi\n" % (20.0))
    pdb.write("    0.0 %1.2f      zlo zhi\n" % (20.0))
    pdb.write("\n\n")
    pdb.write("Atoms\n")
    pdb.write("\n")

    for i in range(len(coordREF)):
        pdb.write("     %d   %d  %d  %1.4f    %1.4f    %1.4f\n" %
                  (i + 1, 1, atomtypes[i], coordREF[i][0] * 10 + 10.0,
                   coordREF[i][1] * 10 + 10.0, coordREF[i][2] * 10 + 10.0))

    pdb.write("\n")
    pdb.write("Bonds\n")
    pdb.write("\n")
    for n in range(len(bonds)):
        pdb.write("     %d   1     %d     %d\n" %
                  (n + 1, bonds[n][0] + 1, bonds[n][1] + 1))
    pdb.close()
