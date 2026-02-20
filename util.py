import numpy as np
import random

def XRD_New(ListCascades):
    groups_list = []
    group1 = []
    group1.append(ListCascades[1])
    group1.append(ListCascades[2])
    group1.append(ListCascades[3])
    groups_list.append(group1)

    group2 = []
    group2.append(ListCascades[1])
    group2.append(ListCascades[4])
    group2.append(ListCascades[5])
    groups_list.append(group2)

    group3 = []
    group3.append(ListCascades[2])
    group3.append(ListCascades[4])
    group3.append(ListCascades[6])
    groups_list.append(group3)

    group4 = []
    group4.append(ListCascades[3])
    group4.append(ListCascades[5])
    group4.append(ListCascades[6])
    groups_list.append(group4)
    return groups_list

def Weights (Number_layer, Number_Mix_Per_Layer):
    probability = []
    for i in range(Number_layer):
        probabTem = []
        for j in range(Number_Mix_Per_Layer):
            probabTem.append((1) / (Number_Mix_Per_Layer))
        probability.append(probabTem)
    return probability




