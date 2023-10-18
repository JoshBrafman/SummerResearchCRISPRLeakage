import csv
from pandas import *
import numpy as np
from itertools import combinations
import shutil
import os

#This method was taken from piCRISPR
def enforceLength(sequence, requireLength): #Make sequence a certain length. Required for Hamming distance to make sure all grna sequences are the same length 
    if (len(sequence) < requireLength): sequence = '0'*(requireLength-len(sequence))+sequence # in case sequence is too short, fill in zeros from the beginning 


def myHamming(first, second): #myHamming is Hamming distance, but 0 characters that were filled in from enforceLength() are considered to be matches
    assert len(first) == len(second), "lengths are not equal"
    diffs = 0
    for i in range(len(first)):
        if ((first[i] != second[i]) and (first[i] != '0') and second[i] != '0'):
            diffs+=1
    return diffs

#The following editDistance method was taken from online
def editDistance2(str1, str2, m, n) -> int:
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
 
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
 
            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j    # Min. operations = j
 
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
 
            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
 
    return dp[m][n]




def getEditScores(testRnaList, trainingRnaList): 
    trainingSize = len(trainingRnaList)
    testSize = len(testRnaList)
    editScores = np.zeros((trainingSize,testSize)) #init double array to store edit distances between each pair of sequences
    for i in range(trainingSize):
        for j in range(testSize):
            editScores[i][j] = editDistance2(trainingRnaList[i], testRnaList[j], len(trainingRnaList[i]), len(testRnaList[j]))
    return editScores #editScores[i][j] contains the edit distance between trainingRnaList[i] and testRnaList[j]

def getHammingScores(testRnaList, trainingRnaList):
    trainingSize = len(trainingRnaList)
    testSize = len(testRnaList)
    hammingScores = np.zeros((trainingSize,testSize)) #init double array to store hamming distances between each pair of sequences
    for i in range(trainingSize):
        for j in range(testSize):
            hammingScores[i][j] = myHamming(trainingRnaList[i], testRnaList[j])
    return hammingScores #hammingScores[i][j] contains the Hamming distance between trainingRnaList[i] and testRnaList[j]


def makeFile(filename):
    if os.path.exists(filename):
        os.remove(filename)   #BE CAREFUL ABOUT THIS. CAN CHANGE IT TO THROW EXCEPTION INSTEAD
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['splitNumber', 'testIDs', 'trainingWithSimilarTestEdit', 'trainingWithSimilarTestHamming', 'testWithSimilarTrainingHamming', 'testWithSimilarTrainingEdit', 'averageClosestHammingPerTest', 'averageClosestEditPerTest'])
        writer.writeheader()
        print(f"CSV file '{filename}' has been created.")
        destination_path = './' + filename
        shutil.move(filename, destination_path)

        data = read_csv('offtarget_260520_nuc.csv.zip')
        data = data.copy() #probably not necessary but wanted to make sure I didn't mess up the data

        expIDs = np.unique(data['experiment_id']) #list of all unique experiment ids
        counter = 0
        for triplet in combinations(expIDs, 3): #This loop will go through all 17 choose 3 ways of choosing 3 experiments
            fillIn = [] #array which will store the information that will be added to the csv file
            counter+=1

            exp1 = triplet[0]
            exp2 = triplet[1]
            exp3 = triplet[2]

            testData = data[(data['experiment_id'] == exp1) | (data['experiment_id'] == exp2) | (data['experiment_id'] == exp3)]
            trainingData = data[(data['experiment_id'] != exp1) & (data['experiment_id'] != exp2) & (data['experiment_id'] != exp3)]
            testRnaList = np.unique(testData['grna_target_sequence'].values.tolist()) #all unique grna sequences that are in the testData
            trainingRnaList = np.unique(trainingData['grna_target_sequence'].values.tolist()) #all unique grna sequences that are in the training data
            
            #Enforce a length of 23 for every grna sequence
            testRnaList = [enforceLength(rna, 23) for rna in testRnaList] 
            trainingRnaList = [enforceLength(rna, 23) for rna in trainingRnaList]

            #initialize counters that keep track of how many similar sequences there are for different ways of measuring leakage
            numTrainingWithSimilarTestEditDistance = 0
            numTestWithSimilarTrainingEditDistance = 0
            numTestWithSimilarTrainingHamming = 0
            numTrainingWithSimilarTestHamming = 0

            editScores = getEditScores(testRnaList, trainingRnaList)

            #important: for editScores[i][j] i = training and j = test
            for i in range(len(trainingRnaList)):
                    for j in range(len(testRnaList)):
                        if editScores[i][j] <= 5: #this pair of sequences is within an edit distance of 5
                            numTrainingWithSimilarTestEditDistance+=1
                            break #as long as we just find one similar sequence we add one; we do not look for more sequences that are similar to this one 
            ratioOfTrainingWithASimilarTestEditDistance = numTrainingWithSimilarTestEditDistance/len(trainingRnaList)
            #We just calculated the number of grna sequences in the training set that had at least one similar grna sequence in the test set divided by the total number of grna sequences in the training set

            for j in range(len(testRnaList)):
                for i in range(len(trainingRnaList)):
                    if editScores[i][j] <= 5: #this pair of sequences is within an edit distance of 5
                        numTestWithSimilarTrainingEditDistance+=1
                        break #as long as we just find one similar sequence we add one; we do not look for more sequences that are similar to this one 
            ratioOfTestWithASimlarTrainingEditDistance = numTestWithSimilarTrainingEditDistance/len(testRnaList)
            #We just calculated the number of grna sequences in the test set that had at least one similar grna sequence in the training set divided by the total number of grna sequences in the test set

            hammingScores = getHammingScores(testRnaList, trainingRnaList)
            #for hammingScores[i][j] i = training and j = test
            for i in range(len(trainingRnaList)):
                    for j in range(len(testRnaList)):
                        if hammingScores[i][j] <= 5:
                            numTrainingWithSimilarTestHamming+=1
                            break
            ratioOfTrainingWithAsimilarTestHamming = numTrainingWithSimilarTestHamming/len(trainingRnaList)
            #We just calculated the number of grna sequences in the training set that had at least one similar grna sequence in the test set divided by the total number of grna sequences in the training set (hamming this time instead of edit)

            for j in range(len(testRnaList)):
                for i in range(len(trainingRnaList)):
                    if hammingScores[i][j] <= 5:
                        numTestWithSimilarTrainingHamming+=1
                        break
            ratioOfTestWithASimlarTrainingHammingDistance = numTestWithSimilarTrainingEditDistance/len(testRnaList)
            #We just calculated the number of grna sequences in the test set that had at least one similar grna sequence in the training set divided by the total number of grna sequences in the test set (hamming this time instead of edit)


            #Calculate two more leakage metrics

            #For each grna sequence in the test set, calculate the closest hamming distance to a grna sequence in the training set
            closestHammingForEachTestGuide = []
            for j in range(len(testRnaList)):
                closestHamming = 24 #the sequences were length 23 so this is one more than is possible
                for i in range(len(trainingRnaList)):
                    closestHamming = min(closestHamming, hammingScores[i][j])
                closestHammingForEachTestGuide.append(closestHamming)
            assert len(closestHammingForEachTestGuide) == len(testRnaList) #sanity check: should be one integer for each test grna sequence

            #For each grna sequence in the test set, calculate the closest edit distance to a grna sequence in the training set
            closestEditForEachTestGuide = []
            for j in range(len(testRnaList)):
                closestEdit = 24
                for i in range(len(trainingRnaList)):
                    closestEdit = min(closestEdit, editScores[i][j])
                closestEditForEachTestGuide.append(closestEdit)
            assert len(closestEditForEachTestGuide) == len(testRnaList)

            averageClosestHamming = sum(closestHammingForEachTestGuide)/len(closestHammingForEachTestGuide)
            averageClosestEdit = sum(closestEditForEachTestGuide)/len(closestEditForEachTestGuide)

            new_row  = {}
            new_row['splitNumber'] = counter
            new_row['testIDs'] = triplet
            new_row['trainingWithSimilarTestEdit'] = ratioOfTrainingWithASimilarTestEditDistance
            new_row['trainingWithSimilarTestHamming'] = ratioOfTrainingWithAsimilarTestHamming
            new_row['testWithSimilarTrainingHamming'] = ratioOfTestWithASimlarTrainingHammingDistance
            new_row['testWithSimilarTrainingEdit'] = ratioOfTestWithASimlarTrainingEditDistance
            new_row['averageClosestHammingPerTest'] = averageClosestHamming
            new_row['averageClosestEditPerTest'] = averageClosestEdit
            fillIn.append(new_row)

            writer.writerows(fillIn)




#Note: there are definitely ways to improve the speed of the following method which calculates the leakage in terms of overlap of coordinates. Because this is just a one time calculation, however, I did not work on making a more efficient algorithm
#This currently is used as an entirely different method than the one above and is saved to a different file, but they can be combined so it's all done in one place


def overlapLeakage(): #calculate leakage in terms of overlaps. What is the percentage of targets in the test set that have overlap with a target in the training set
    data = read_csv('offtarget_260520_nuc.csv.zip')
    data = data.copy() #not necessary just was afraid to mess up the data
    filename = 'overlap.csv'
    if os.path.exists(filename):
        os.remove(filename) #WARNING
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['splitNumber', 'testIDs', 'overlap'])
        writer.writeheader()
        print(f"CSV file '{filename}' has been created.")
        destination_path = './' + filename
        shutil.move(filename, destination_path)


        expIDs = np.unique(data['experiment_id'])
        counter = 0
        
        for triplet in combinations(expIDs, 3):
            fillIn = []
            counter+=1
            exp1 = triplet[0]
            exp2 = triplet[1]
            exp3 = triplet[2]

            testData = data[(data['experiment_id'] == exp1) | (data['experiment_id'] == exp2) | (data['experiment_id'] == exp3)]
            trainingData = data[(data['experiment_id'] != exp1) & (data['experiment_id'] != exp2) & (data['experiment_id'] != exp3)]

            trainChromosomes = np.unique(trainingData['target_chr'])
            testChromosomes = np.unique(testData['target_chr'])

            trainingStarts = trainingData['target_start'].values
            testStarts = testData['target_start'].values

            trainingEnds = trainingData['target_end'].values
            testEnds = testData['target_end'].values

            chrToTrainingPoints = {}

            for chr in trainChromosomes:
                chrToTrainingPoints[chr] = set() 


            #change these variables so they are no longer unique lists and convert to numpy arrays
            trainChromosomes = trainingData['target_chr'].values
            testChromosomes = testData['target_chr'].values

            #sanity checks
            assert len(trainChromosomes) == len(trainingStarts)
            assert len(trainingStarts) == len (trainingEnds)

            assert len(testChromosomes) == len(testStarts)
            assert len(testStarts) == len(testEnds)

            #keep track of the points on each chromosome that existed in the training data
            for i in range(len((trainChromosomes))):
                chr = trainChromosomes[i]
                for j in range(trainingStarts[i], trainingEnds[i] + 1):
                    chrToTrainingPoints[chr].add(j)

            rowsWhereTestSequenceHadOverlapWithTrainingSequence = 0

            #add 1 for each target that has a point on a chromosome that was seen in the training data
            for i in range(len((testChromosomes))):
                chr = testChromosomes[i]
                pointsInTraining = chrToTrainingPoints[chr]
                for j in range(testStarts[i], testEnds[i] + 1):
                    #chrToTestPoints[chr].add(j)
                    if j in pointsInTraining:
                        rowsWhereTestSequenceHadOverlapWithTrainingSequence += 1
                        break

            new_row = {}
            new_row['splitNumber'] = counter
            new_row['testIDs'] = triplet
            new_row['overlap'] = rowsWhereTestSequenceHadOverlapWithTrainingSequence/len(testChromosomes) #len testChromosomes is really just the number of rows in testData (I just used that as a proxy for some reason I believe)
            fillIn.append(new_row)

            writer.writerows(fillIn)
            print(counter, rowsWhereTestSequenceHadOverlapWithTrainingSequence/len(testChromosomes))



        
        






#makeFile('LeakagesBesidesForCoordinatesADDEDTHISHERESOCANTMESSITUPACCIDENTALLY.csv')
#overlapLeakage()













