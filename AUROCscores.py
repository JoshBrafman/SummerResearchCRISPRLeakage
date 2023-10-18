import csv
from pandas import *
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import shutil
import os


#This method was taken from piCRISPR
def enforceSeqLength(sequence, requireLength):
    if (len(sequence) < requireLength): sequence = '0'*(requireLength-len(sequence))+sequence # in case sequence is too short, fill in zeros from the beginning (or sth arbitrary thats not ATCG)
    return sequence[-requireLength:] # in case sequence is too long

#taken from piCRISPR: I believe this is the method that corresponds with figure A under "Feature Encoding"
def seqToOneHot(sequence, seq_guide):
    sequence = enforceSeqLength(sequence, len(seq_guide))
    bases = ['A', 'T', 'C', 'G']
    onehot = np.zeros(6 * len(sequence), dtype=int)

    for i in range(min(len(sequence), len(seq_guide))):
        for key, base in enumerate(bases):
            if sequence[i] == base:
                onehot[6 * i + key] = 1
            if seq_guide[i] == base:
                onehot[6 * i + key] = 1

        if sequence[i] != seq_guide[i]:  # Mismatch
            try:
                if bases.index(sequence[i]) < bases.index(seq_guide[i]):
                    onehot[6 * i + 4] = 1
                else:
                    onehot[6 * i + 5] = 1
            except ValueError:  # Non-ATCG base found
                pass

    return onehot.tolist()


#ended up giving up on adding GCContent feature information but this method would have been useful for that
# # conversion function to split the string and convert to floats
# def convert_string_to_floats(value):
#     if isinstance(value, str):
#         float_values = [float(x) for x in value[1:-1].split()]
#         #floats = [float(x) for x in data['GCContent'][0][1:-1].split()]
#         return np.array(float_values)
#         #return float_values
#     else:
#         return value

    
#remove all data besides for this one chromosome from the test data
def keepOneChromosome(testData, chr):
     return testData[(testData['target_chr'] == chr)]

#remove one chromosome from the training data
def removeOneChromosome(trainingData, chr):
    return trainingData[(trainingData['target_chr'] != chr)]

def rf(X_train, y_train, X_test, y_test): # Train and get auroc score of Random Forest classifier

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)

    #I hope all the following steps to evaluate the auroc is correct but should confirm (same goes for each ML model)
    y_scores = rf_classifier.predict(X_test) 

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auroc_rf = auc(fpr, tpr)
    print("rf", auroc_rf)
    return auroc_rf

def logreg(X_train, y_train, X_test, y_test): # Train and get auroc score of Logistic Regression classifier
    logreg_classifier = LogisticRegression(random_state=42)

    logreg_classifier.fit(X_train, y_train)

    y_scores_logreg = logreg_classifier.predict(X_test)

    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_scores_logreg)
    auroc_logreg = auc(fpr_logreg, tpr_logreg)
    print("logreg", auroc_logreg)
    return auroc_logreg


def svm(X_train, y_train, X_test, y_test): # Train and get auroc score of SVM classifier
    svm_classifier = SVC(probability=True, random_state=42)

    svm_classifier.fit(X_train, y_train)

    y_scores_svm = svm_classifier.predict(X_test)

    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_scores_svm)
    auroc_svm = auc(fpr_svm, tpr_svm)
    print("svm", auroc_svm)
    return auroc_svm

def mlp(X_train, y_train, X_test, y_test): # Train and get auroc score of MLP classifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)

    mlp_classifier.fit(X_train, y_train)

    y_scores_mlp = mlp_classifier.predict(X_test)

    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_scores_mlp)
    auroc_mlp = auc(fpr_mlp, tpr_mlp)
    print("mlp", auroc_mlp)

    return auroc_mlp

def doXgb(X_train, y_train, X_test, y_test): # Train and get auroc score of XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)

    xgb_classifier.fit(X_train, y_train)

    y_scores_xgb = xgb_classifier.predict(X_test)

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_scores_xgb)
    auroc_xgb = auc(fpr_xgb, tpr_xgb)
    print("xgb", auroc_xgb)
    return auroc_xgb


def trainModelsReturnAUROCs(data, exp1,exp2,exp3, onlySeqInfo): #returns a dictionary with the AUROC score for each model. The three experiments define the train-test partition    
    #There are other feature labels that I should  add in the future                                                                                                                                             #after 5 used to be GCContent and then used to be WSScore and bunch of other stuff
    feature_labels = ['epigen_ctcf', 'epigen_dnase', 'epigen_rrbs', 'epigen_h3k4me3', 'epigen_drip', 'energy_1', 'energy_2', 'energy_3', 'energy_4', 'energy_5']

    if data.empty:
        print("Dataset is empty")
        exit()

    testData = data[(data['experiment_id'] == exp1) | (data['experiment_id'] == exp2) | (data['experiment_id'] == exp3)] #testData consists of three experiments
    trainingData = data[(data['experiment_id'] != exp1) & (data['experiment_id'] != exp2) & (data['experiment_id'] != exp3)] #every othre experiment is in the training data

    #remove chr1 from training data and keep only chr1 in test data: may want to include one or both of these lines
    # testData = keepOneChromosome(testData, 'chr1')
    # trainingData = removeOneChromosome(trainingData, 'chr1')

    X_test = testData[feature_labels].values #the test input features

    length = len(np.array(seqToOneHot(enforceSeqLength("A", 23), enforceSeqLength("A", 23)))) #just a way to figure out how the length of the ultimate encoded genetic sequence feature
    seqInfo = np.ones((X_test.shape[0], length)) #initializing an array to store the genetic sequence encoding information
    i = 0
    for sequence, sequence2 in zip(testData['target_sequence'], testData['grna_target_sequence']):
        sequence = enforceSeqLength(sequence, 23)
        sequence2 = enforceSeqLength(sequence2, 23)
        hotInListForm = seqToOneHot(sequence, sequence2)
        seqInfo[i] = np.array(hotInListForm)
        i += 1

    #append genetic seq info to the rest of the features (or make the genetic seq info the only feature)
    if not onlySeqInfo:
        X_test = np.append(X_test, seqInfo, axis = 1)
    else:
        X_test = seqInfo  

    y_test = np.where(testData['measured'] < 1e-5, 0, 1) #binary labels. If under a certain threshold it's considered to be 0


    X_train = trainingData[feature_labels].values 


    length = len(np.array(seqToOneHot(enforceSeqLength("A", 23), enforceSeqLength("A", 23))))
    seqInfo = np.ones((X_train.shape[0], length)) #reassigning the seqInfo variable
    i = 0
    for sequence, sequence2 in zip(trainingData['target_sequence'], trainingData['grna_target_sequence']):
        sequence = enforceSeqLength(sequence, 23)
        sequence2 = enforceSeqLength(sequence2, 23)
        hotInListForm = seqToOneHot(sequence, sequence2)
        seqInfo[i] = np.array(hotInListForm)
        i += 1

    if not onlySeqInfo:
        X_train = np.append(X_train, seqInfo, axis = 1)
    else:
        X_train = seqInfo

    y_train = np.where(trainingData['measured'] < 1e-5, 0, 1) #binary labels like above

    #use the following lines if you don't want to train in parallel

    auroc_rf = rf(X_train, y_train, X_test, y_test)

    # auroc_logreg = logreg(X_train, y_train, X_test, y_test)

    # # auroc_svm = svm(X_train, y_train, X_test, y_test)

    # # auroc_mlp = mlp(X_train, y_train, X_test, y_test)

    # auroc_xgb = doXgb(X_train, y_train, X_test, y_test)


    #use the following lines if you want to train in parallel

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     future_rf = executor.submit(rf, X_train, y_train, X_test, y_test)
    #     future_logreg = executor.submit(logreg, X_train, y_train, X_test, y_test)
    #     #future_svm = executor.submit(svm, X_train, y_train, X_test, y_test)
    #     #future_mlp = executor.submit(mlp, X_train, y_train, X_test, y_test)
    #     future_xgb = executor.submit(doXgb, X_train, y_train, X_test, y_test)

    #     # Wait for all tasks to complete 
    #     concurrent.futures.wait([future_rf, future_logreg, future_xgb]) #add future_svm and future_mlp here if desired, but they take a really long time to train

    # # Now, get the results of each function
    # auroc_rf = future_rf.result()
    # auroc_logreg = future_logreg.result()
    # #auroc_svm = future_svm.result()
    # #auroc_mlp = future_mlp.result()
    # auroc_xgb = future_xgb.result()


    scores = {}
    scores['rf'] = auroc_rf
    #scores['logreg'] = auroc_logreg
    #scores['svm'] = auroc_svm
    #scores['mlp'] = auroc_mlp
    #scores['xgb'] = auroc_xgb

    return scores


def makeCSVFile(): #use this method to make the file that will store the data you are calculating if you do not have the file already (I used this method on its own to create the file and then ran the rest of the code but it can also just be used at the beginning of running the code)

    # filename = 'splitOnChr1.csv'
    # filename = 'onlySeqInfo.csv'
    # filename = 'trainOnAllTestOn1.csv'
    filename = 'your_filename_here'
    if os.path.exists(filename): #WARNING THAT THIS IS HERE. Can remove this line if desired
        os.remove(filename)
    with open(filename, mode='w', newline='') as file:
        #writer = csv.DictWriter(file, fieldnames=['splitNumber', 'testIDs', 'leakage', 'rf', 'logreg', 'svm', 'mlp', 'xgb'])
        writer = csv.DictWriter(file, fieldnames=['splitNumber', 'testIDs', 'rf'])
        #fieldnames are the columns that you want the file to have
        writer.writeheader()

    print(f"CSV file '{filename}' has been created.")
    destination_path = './' + filename
    shutil.move(filename, destination_path)



def calculateAndAddAurocScoresToExistingFile(data, expIDs): #use makeCSVFile() before this so you have the file to save the info to
    #filename is the file you will be saving the info to
    #filename = 'withSequenceInfo.csv'
    #filename = '/mnt/c/users/joshu/yaron/Bar-Ilan/withSequenceInfo.csv'
    filename = '/content/drive/MyDrive/yaron/AllModelsWithSequenceInfo (4).csv'
    #filename = '/content/AllModelsWithSequenceInfo.csv'
    #filename = 'notReal'
    fillIn = [] #where we temporarily store the calculated data before adding it to the file
    counter = 1
    for triplet in combinations(expIDs, 3): #go through every possible train-test split where test set is defined by three experiments
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['splitNumber', 'testIDs', 'leakage', 'rf', 'logreg', 'svm', 'mlp', 'xgb'])
            #update writer so it has whatever columns the csv has

            #use the following lines if the spreadsheet is already partially filled in and you want to continue where you leftoff
            # if counter <= [last split number that was already filled in]:
            #      counter+=1 
            #      continue

            modelToAuroc = trainModelsReturnAUROCs(data,triplet[0], triplet[1], triplet[2]) #get the auroc scores for each model. the triplet is the experiments that make up the test set. The boolean is whether you want only the seq info
            new_row  = {}
            new_row['splitNumber'] = counter
            new_row['testIDs'] = triplet
            new_row['rf'] = modelToAuroc['rf']
            #new_row['logreg'] = modelToAuroc['logreg']
            #new_row['svm'] = modelToAuroc['svm']
            #new_row['mlp'] = modelToAuroc['mlp']
            #new_row['xgb'] = modelToAuroc['xgb']
            fillIn.append(new_row)
            counter+=1
            print(new_row)
            if counter % 1 == 0:
                writer.writerows(fillIn)
                fillIn = []
                print("filled in")






# makeCSVFile()
# exit()

#use nrows = x when testing so you read only part of the dataset to save time

data = read_csv('/content/drive/MyDrive/yaron/offtarget_260520_nuc.csv.zip', usecols=['experiment_id', 'epigen_ctcf', 'epigen_dnase', 'epigen_rrbs', 'epigen_h3k4me3', 'epigen_drip', 'energy_1', 'energy_2', 'energy_3', 'energy_4', 'energy_5', 'GCContent', 'measured', 'target_sequence', 'grna_target_sequence', 'target_chr'])
data = data.copy() #just in case

expIDs = np.unique(data['experiment_id'])
calculateAndAddAurocScoresToExistingFile(data, expIDs)

