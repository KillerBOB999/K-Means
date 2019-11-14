#!/usr/bin/env python3

# INFORMATION--------------------------------------------------------------------------
# DEVELOPER:        Anthony Harris
# SLATE:            Anthony999
# DATE:             14 November 2019
# PURPOSE:          Use the K-Means clustering algorithm to classify entities in a
#                   simple data set.
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# IMPORTS------------------------------------------------------------------------------
import sys
import random as rng
import statistics
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# GLOBALS------------------------------------------------------------------------------
rng.seed(999)           # Seed the random number generator to produce reliable results

# DATASET breakdown:
#   Numerical definition of entity == (float(), float())
#   Entity defined in data file with classification == (NumericalDefinition, int())
#   Entity and determined cluster == (EntityDefinition, int())
DATASET = [((float(),float()),int(),int())]
K_Means = [(float(),float()), (float(),float()), (float(),float())]
differenceMatrix = [int(sys.maxsize), int(sys.maxsize), int(sys.maxsize)]
K_NumberOfClusters = len(K_Means)
meanInfo = {0 : [float(0), float(0), int(0)], 
            1 : [float(0), float(0), int(0)],
            2 : [float(0), float(0), int(0)]}

def printData():
    global DATASET
    totalMissed = 0
    for i in range(0, K_NumberOfClusters):
        classOccurences = [0, 0, 0]
        mostFrequent = -1
        numMissed = 0
        print("================================")
        print("================================")
        print("Cluster ", i)
        print("Size of Cluster ", i, " is ", meanInfo[i][2])
        for entity in DATASET:
            if entity[2] == i:
                classOccurences[entity[1]] += 1
        if classOccurences[0] >= classOccurences[1] and classOccurences[0] >= classOccurences[2]:
                mostFrequent = 0
        elif classOccurences[1] >= classOccurences[2]:
            mostFrequent = 1
        else:
            mostFrequent = 2
        for entity in DATASET:
            if entity[2] == i:
                if not mostFrequent == entity[1]:
                    numMissed += 1
                    totalMissed += 1
        print("Cluster Label: ", mostFrequent)
        print("Number of misclustered objects in this cluster: ", numMissed)
        for entity in DATASET:
            if entity[2] == i:
                print(entity[0], " ", entity[1])
    print("================================")
    print("================================")
    print("Accuracy Rate: ", (len(DATASET) - totalMissed) / len(DATASET) * 100, "%")
    return

def updateKMeans(entityIndex, entityLow, entityHigh):
    global DATASET, meanInfo
    meanInfo[DATASET[entityIndex][2]][0] += entityLow
    meanInfo[DATASET[entityIndex][2]][1] += entityHigh
    meanInfo[DATASET[entityIndex][2]][2] += 1
    return

def checkFor0s():
    if meanInfo[0][2] == 0: meanInfo[0][2] = 1
    if meanInfo[1][2] == 0: meanInfo[1][2] = 1
    if meanInfo[2][2] == 0: meanInfo[2][2] = 1
    return

def findMeans():
    global DATASET, K_Means, differenceMatrix, meanInfo
    meanInfo = {0 : [0, 0, 0], 1: [0, 0, 0], 2 : [0, 0, 0]}
    for entityIndex in range(0, len(DATASET)):
        entityLow = DATASET[entityIndex][0][0]
        entityHigh = DATASET[entityIndex][0][1]
        differenceMatrix = [pow(pow(entityLow - K_Means[0][0], 2) + 
                                pow(entityHigh - K_Means[0][1], 2), 0.5),
                            pow(pow(entityLow - K_Means[1][0], 2) + 
                                pow(entityHigh - K_Means[1][1], 2), 0.5),
                            pow(pow(entityLow - K_Means[2][0], 2) + 
                                pow(entityHigh - K_Means[2][1], 2), 0.5)]
        if differenceMatrix[0] <= differenceMatrix[1] and differenceMatrix[0] <= differenceMatrix[2]:
            DATASET[entityIndex] = (DATASET[entityIndex][0], DATASET[entityIndex][1], 0)
        elif differenceMatrix[1] <= differenceMatrix[2]:
            DATASET[entityIndex] = (DATASET[entityIndex][0], DATASET[entityIndex][1], 1)
        else:
            DATASET[entityIndex] = (DATASET[entityIndex][0], DATASET[entityIndex][1], 2)
        updateKMeans(entityIndex, entityLow, entityHigh)
    checkFor0s()
    K_Means = [(meanInfo[0][0] / meanInfo[0][2], meanInfo[0][1] / meanInfo[0][2]),
                (meanInfo[1][0] / meanInfo[1][2], meanInfo[1][1] / meanInfo[1][2]),
                (meanInfo[2][0] / meanInfo[2][2], meanInfo[2][1] / meanInfo[2][2])]
    return

def cluster():
    global K_Means
    previousMeans = K_Means
    progressMade = True
    while progressMade:
        findMeans()
        progressMade = not (previousMeans == K_Means)
        previousMeans = K_Means
    return

def initialize():
    global K_Means
    findMeans()
    print("Initial K means are:")
    for i in range(0, len(K_Means)):
        print("K_Means[", i, "] = ", K_Means[i])
    return

def loadDataSet(fileName):
    global DATASET
    DATASET.pop()
    file = open(fileName, "r")
    for line in file:
        entity = line.split()
        DATASET.append(((float(entity[0]), float(entity[1])), int(entity[2]), rng.randint(0, 2)))
    return

def main():
    fileName = "synthetic_2D.txt"
    loadDataSet(fileName)
    initialize()
    cluster()
    printData()
    return

main()