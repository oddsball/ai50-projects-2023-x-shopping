import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

#You should not modify anything else in shopping.py other than the functions the specification calls for you to implement, though you may write additional functions 
# and/or import other Python standard library modules. 
# You may also import numpy or pandas or anything from scikit-learn, if familiar with them, but you should not use any other third-party Python modules. 
# You should not modify shopping.csv.

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    
    evidence = []
    labels = []
    
    dictmonth = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    #The load_data function should accept a CSV filename as its argument, open that file, 
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        ##Since you’ll have one piece of evidence and one label for each row of the spreadsheet, the length of the evidence list and the length of the labels list should ultimately be equal to the number of rows in the CSV spreadsheet (excluding the header row). The lists should be ordered according to the order the users appear in the spreadsheet. That is to say, evidence[0] should be the evidence for the first user, and labels[0] should be the label for the first user.
        #notes : solve by for loop. evidence row for each row, append to evidence list, matching labels 1 if true 0 if not. Processes each row "chronologically"
        for row in reader:
            #evidence should be a list of all of the evidence for each of the data points.
            #The values in each evidence list should be in the same order as the columns that appear in the evidence spreadsheet. You may assume that the order of columns in shopping.csv will always be presented in that order.
            #Note that, to build a nearest-neighbor classifier, all of our data needs to be numeric. Be sure that your values have the following types:
                #Administrative, Informational, ProductRelated, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, and Weekend should all be of type int
                #Administrative_Duration, Informational_Duration, ProductRelated_Duration, BounceRates, ExitRates, PageValues, and SpecialDay should all be of type float.
                #Month should be 0 for January, 1 for February, 2 for March, etc. up to 11 for December.
                #VisitorType should be 1 for returning visitors and 0 for non-returning visitors.
                #Weekend should be 1 if the user visited on a weekend and 0 otherwise.
                #Each value of labels should either be the integer 1, if the user did go through with a purchase, or 0 otherwise.
            evidence_row = [
                int(row[0]), float(row[1]), int(row[2]), float(row[3]), int(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), dictmonth[row[10]], int(row[11]), int(row[12]), int(row[13]), int(row[14]),
                #notes : most efficient way of making row 14+15 binary is to result 1 if it matches the string.
                int(row[15] == 'Returning_Visitor'), int(row[16] == 'TRUE'),
            ]
            evidence.append(evidence_row)
            #labels should be a list of all of the labels for each data point.
            #Each element in the evidence list should itself be a list. The list should be of length 17: the number of columns in the spreadsheet excluding the final column (the label column).
            labels.append(int(row[17] == 'TRUE'))

    #and return a tuple (evidence, labels)
    return evidence, labels
    
def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    #Notice that we’ve already imported for you from sklearn.neighbors import KNeighborsClassifier. You’ll want to use a KNeighborsClassifier in this function.
    #return a scikit-learn nearest-neighbor classifier (a k-nearest-neighbor classifier where k = 1) fitted on that training data.
    #notes : classifier from library, setting neighbors to 1.
    shoppingmodel = KNeighborsClassifier(n_neighbors=1)
    #The train_model function should accept a list of evidence and a list of labels
    #notes : fit model by evidence and labels in def train_model
    shoppingmodel.fit(evidence, labels)
    return shoppingmodel

    
    

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    #The evaluate function should accept a list of labels (the true labels for the users in the testing set) and a list of predictions (the labels predicted by your classifier), and return two floating-point values (sensitivity, specificity).
    truepos = 0; actualpos = 0; trueneg = 0; actualneg = 0
    #notes : loop over labels list to calc correct negs and pos
    for index in range (len(labels)):
        if labels[index] == 1:
              actualpos += 1
              if predictions[index] == 1:
                  truepos += 1
        else:
            actualneg += 1
            if predictions[index] == 0:
                trueneg += 1
 
    #sensitivity should be a floating-point value from 0 to 1 representing the “true positive rate”: the proportion of actual positive labels that were accurately identified.
    #specificity should be a floating-point value from 0 to 1 representing the “true negative rate”: the proportion of actual negative labels that were accurately identified.
    #notes : vars to calculate sens and spec pos neg
    #You may assume each label will be 1 for positive results (users who did go through with a purchase) or 0 for negative results (users who did not go through with a purchase).
    #You may assume that the list of true labels will contain at least one positive label and at least one negative label.
    sensitivity = truepos / actualpos #true positive rate
    specificity = trueneg / actualneg #true negative
    
    return sensitivity, specificity

if __name__ == "__main__":
    main()
