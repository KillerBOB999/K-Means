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

# Cluster means
K_Means = [(float(),float()), (float(),float()), (float(),float())]

# Distance matrix that represents the distance between an entity and the centroid of
# a cluster
differenceMatrix = [int(sys.maxsize), int(sys.maxsize), int(sys.maxsize)]

# The number of clusters to be defined
K_NumberOfClusters = len(K_Means)

# Dictionary representation of the cluster, as well as the number of entities assigned
# to the cluster
meanInfo = {0 : [float(0), float(0), int(0)], 
            1 : [float(0), float(0), int(0)],
            2 : [float(0), float(0), int(0)]}
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# FUNCTION DESCRIPTION-----------------------------------------------------------------
# Function Name:    printData
# Parameters:       N/A
# Returns:          N/A
# Description:      Prints information about the result of the implementation of the
#                   K-Means algorithm. Some calculations required to determine useful
#                   information
#--------------------------------------------------------------------------------------
def printData():
    # Give access to global variables
    global DATASET

    # Initialize local variables
    totalMissed = 0     # Total number of misclassified entities

    # Iteratively step through each cluster
    for i in range(0, K_NumberOfClusters):
        classOccurences = [0, 0, 0]     # Frequency matrix of actual class values for
                                        # the entities in the cluster

        mostFrequent = -1               # Most frequent class in the cluster

        numMissed = 0                   # Number of misclassified entities in the
                                        # cluster

        # Format the display and do some helpful calculations
        print("================================")
        print("================================")
        print("Cluster ", i)
        print("Size of Cluster ", i, " is ", meanInfo[i][2])

        # Find the entities in the dataset that match the current cluster and increment
        # the actual class occurences
        for entity in DATASET:
            if entity[2] == i:
                classOccurences[entity[1]] += 1

        # Determine the most frequently occurring class in the cluster and assign the
        # value to the mostFrequent variable, thus labeling the cluster with a class
        if classOccurences[0] >= classOccurences[1] and classOccurences[0] >= classOccurences[2]:
                mostFrequent = 0
        elif classOccurences[1] >= classOccurences[2]:
            mostFrequent = 1
        else:
            mostFrequent = 2

        # Now that we know the class of the cluster, determine how many entities in
        # the cluster have been misclassified and reflect that information in the
        # cluster-scope variable numMissed and the method-scope variable
        # totalMissed
        for entity in DATASET:
            if entity[2] == i:
                if not mostFrequent == entity[1]:
                    numMissed += 1
                    totalMissed += 1

        # Print useful information to the screen
        print("Cluster Label: ", mostFrequent)
        print("Number of misclustered objects in this cluster: ", numMissed)
        for entity in DATASET:
            if entity[2] == i:
                print(entity[0], " ", entity[1])
    print("================================")
    print("================================")
    print("Accuracy Rate: ", (len(DATASET) - totalMissed) / len(DATASET) * 100, "%")
    return
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# FUNCTION DESCRIPTION-----------------------------------------------------------------
# Function Name:    updateKMeans
# Parameters:       entityIndex(int)
#                       Use:    Access the current entity by its index from the
#                               global DATASET variable
#                   entityLow(float)
#                       Use:    Represent the low value of the entity conscisely
#                   entityHigh(float)
#                       Use:    Represent the high value of the entity conscisely
# Returns:          N/A
# Description:      Updates the global variable meanInfo on an entity-by-entity basis
#--------------------------------------------------------------------------------------
def updateKMeans(entityIndex, entityLow, entityHigh):
    # Give access to global variables
    global DATASET, meanInfo

    # Add the entity values to the existing sum of values
    meanInfo[DATASET[entityIndex][2]][0] += entityLow
    meanInfo[DATASET[entityIndex][2]][1] += entityHigh

    # Increment the number of entities in the corresponding cluster
    meanInfo[DATASET[entityIndex][2]][2] += 1
    return
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# FUNCTION DESCRIPTION-----------------------------------------------------------------
# Function Name:    checkFor0s
# Parameters:       N/A
# Returns:          N/A
# Description:      Makes sure that you don't try to divide by 0
#--------------------------------------------------------------------------------------
def checkFor0s():
    # Check denominators of average calculations to ensure no division by 0
    if meanInfo[0][2] == 0: meanInfo[0][2] = 1
    if meanInfo[1][2] == 0: meanInfo[1][2] = 1
    if meanInfo[2][2] == 0: meanInfo[2][2] = 1
    return
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# FUNCTION DESCRIPTION-----------------------------------------------------------------
# Function Name:    findMeans
# Parameters:       N/A
# Returns:          N/A
# Description:      Calculates and defines the values for the global K_Means variable
#--------------------------------------------------------------------------------------
def findMeans():
    # Give access to global variables
    global DATASET, K_Means, differenceMatrix, meanInfo

    # Initialize meanInfo to reflect a new iteration
    meanInfo = {0 : [0, 0, 0], 1: [0, 0, 0], 2 : [0, 0, 0]}

    # Iteratively interact with each entity in the dataset
    for entityIndex in range(0, len(DATASET)):
        entityLow = DATASET[entityIndex][0][0] # conscisely access the entity lowValue
        entityHigh = DATASET[entityIndex][0][1]# conscisely access the entity HighValue
        # Define the distance matrix by determining the euclidean distance between the
        # values of the entity and the centroid of the cluster, defined by the low and
        # high values in the global K_Means variable
        differenceMatrix = [pow(pow(entityLow - K_Means[0][0], 2) + 
                                pow(entityHigh - K_Means[0][1], 2), 0.5),
                            pow(pow(entityLow - K_Means[1][0], 2) + 
                                pow(entityHigh - K_Means[1][1], 2), 0.5),
                            pow(pow(entityLow - K_Means[2][0], 2) + 
                                pow(entityHigh - K_Means[2][1], 2), 0.5)]
        # Determine which cluster is more similar to the entity and assign the entity
        # to it
        if differenceMatrix[0] <= differenceMatrix[1] and differenceMatrix[0] <= differenceMatrix[2]:
            DATASET[entityIndex] = (DATASET[entityIndex][0], DATASET[entityIndex][1], 0)
        elif differenceMatrix[1] <= differenceMatrix[2]:
            DATASET[entityIndex] = (DATASET[entityIndex][0], DATASET[entityIndex][1], 1)
        else:
            DATASET[entityIndex] = (DATASET[entityIndex][0], DATASET[entityIndex][1], 2)
        # Update the information needed to redefine K_Means at the end of the iteration
        # of the dataset
        updateKMeans(entityIndex, entityLow, entityHigh)
    # Make sure math rules are abided by
    checkFor0s()
    # Update the global K_Means variable to reflect the new changes in the clusters
    K_Means = [(meanInfo[0][0] / meanInfo[0][2], meanInfo[0][1] / meanInfo[0][2]),
                (meanInfo[1][0] / meanInfo[1][2], meanInfo[1][1] / meanInfo[1][2]),
                (meanInfo[2][0] / meanInfo[2][2], meanInfo[2][1] / meanInfo[2][2])]
    return
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# FUNCTION DESCRIPTION-----------------------------------------------------------------
# Function Name:    cluster
# Parameters:       N/A
# Returns:          N/A
# Description:      High level representation of the K-Means algorithm. Acts as the
#                   primary driver of the program
#--------------------------------------------------------------------------------------
def cluster():
    # Give access to global variables
    global K_Means
    previousMeans = K_Means

    # Start K-Means algorithm
    progressMade = True     # Boolean representation of a change in the clusters
    # While the clusters are changing, continue the algorithm
    while progressMade:
        findMeans()     # Compute the cluster associated with each entity in the
                        # dataset and update the K_Means variable accordingly
        # Progress has been made if the previous K_Means value is NOT equal to the
        # newly calculated K_Means value
        progressMade = not (previousMeans == K_Means)
        # Assign the previousMeans value to the current K_means value for use in the
        # next iteration (if applicable)
        previousMeans = K_Means
    # End K-Means algorithm
    return
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# FUNCTION DESCRIPTION-----------------------------------------------------------------
# Function Name:    initialize
# Parameters:       N/A
# Returns:          N/A
# Description:      Calculates, defines, and displays the initial values for the global
#                   K_Means variable
#--------------------------------------------------------------------------------------
def initialize():
    # Give access to global variables
    global K_Means

    findMeans() # Compute the cluster associated with each entity in the
                # dataset and update the K_Means variable accordingly

    # Display the initial information
    print("Initial K means are:")
    for i in range(0, len(K_Means)):
        print("K_Means[", i, "] = ", K_Means[i])
    return
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# FUNCTION DESCRIPTION-----------------------------------------------------------------
# Function Name:    loadDataSet
# Parameters:       fileName(string)
#                       Use:    Acts as the relative filepath to the file containing
#                               the dataset
# Returns:          N/A
# Description:      Initializes the global DATASET variable based upon the values found
#                   in the file
#--------------------------------------------------------------------------------------
def loadDataSet(fileName):
    # Give access to global variables
    global DATASET

    # Remove unnecessary empty first value of DATASET
    DATASET.pop()

    # Handle file operations
    file = open(fileName, "r")

    # Tokenize each line anf update the DATASET
    for line in file:
        entity = line.split()
        DATASET.append(((float(entity[0]), float(entity[1])), int(entity[2]), rng.randint(0, 2)))
    return
#--------------------------------------------------------------------------------------

#/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

# FUNCTION DESCRIPTION-----------------------------------------------------------------
# Function Name:    main
# Parameters:       N/A
# Returns:          N/A
# Description:      Entry point of the program
#--------------------------------------------------------------------------------------
def main():
    # The following string represents the filename of the file used
    fileName = "synthetic_2D.txt"

    # Handle file operations
    loadDataSet(fileName)

    # Initialize important data
    initialize()

    # Run the K-Means algorithm
    cluster()

    # Display information about the results
    printData()
    return
#--------------------------------------------------------------------------------------
main()