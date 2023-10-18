# SummerResearchCRISPRLeakage
CRISPR/Cas9 is a widely used gene-editing technique, but the occurrence of unexpected off-target editing presents a significant limitation in its application within biological and clinical contexts. Consequently, researchers have developed and trained machine-learning models to predict when and where off-target editing occurs. However, existing studies have largely overlooked the potential impact of train-test leakage on the evaluation of these models.

Train-test leakage occurs when a machine-learning model is tested on data that closely resembles the data on which it was trained. In such cases, the model may achieve high performance during testing because it has simply "memorized" the characteristics that these data points share. For instance, imagine training a model to recognize cats by showing it images of felines that happen to have dark birthmarks near their eyes. If the model is then tested with a picture of a cat exhibiting a similar birthmark, it might accurately classify the image as depicting a cat. However, we cannot be certain whether the model has genuinely learned what a cat looks like; there’s a possibility that it has merely “memorized” an association between dark birthmarks and cats. Therefore, if confronted with an image of a cat lacking a birthmark, the model may fail. 
In a similar vein, if there was leakage between the data used to train and test off-target prediction models, it is possible that they performed better than they should have during testing because they memorized the “leaked” characteristics. Addressing train-test leakage is therefore critical for evaluating model performance in real-world scenarios.

The purpose of my study is to determine the correlation between train-test leakage and the performance of off-target prediction models, if it exists, and to offer strategies to minimize this risk so a more realistic evaluation of these models can be performed.

To analyze this issue, I partitioned a popular off-target dataset into training and test sets in 680 different ways. For each of these train-test splits, I calculated the leakage between the two sets using seven different metrics. This was done in **compute_leakages.py** The metrics included various way of evaluating the similarity between the guide RNA sequences across the split, as well as calculating the percentage of target locations (regions on the genome where off-target editing may have occurred) in the test set that physically overlapped with a target location in the training set.

In addition to calculating the leakage, for each train-test split I trained a Random-Forest, Logistic-Regression, and XGBoost model (machine-learning types). I evaluated these models using AUROC, which is a common metric to evaluate binary classification, because the models were trained to classify between experimental off-targets (positives) and potential off-targets (negatives) that were found in the genome. This was done in **AUROCscores.py**

Finally, I calculated the correlation between the train-test leakage and model performance for each leakage metric and machine learning model across the 680 different partitions. 





