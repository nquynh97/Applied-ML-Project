import sys
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mydecisiontree import decisiontree
from myknn import knn
from myrandomforest import randomforest
from sklearn.preprocessing import StandardScaler

# Get the parameters from command line input
classifier_name = sys.argv[1].lower()
data_path = sys.argv[2]

# Select and initialize the corresponding classifier
if classifier_name == "random_forest":
    n_estimators = int(sys.argv[3])
    criterion = str(sys.argv[4])
    max_depth = int(sys.argv[5])
    classifier = randomforest(n_estimators, criterion, max_depth)
elif classifier_name == "decision_tree":
    criterion = str(sys.argv[3])
    max_depth = int(sys.argv[4])
    classifier = decisiontree(criterion, max_depth)
elif classifier_name == "knn":
    neighbor = int(sys.argv[3])
    classifier = knn(neighbor)
else:
    raise ValueError('Classifier name is not correct')
    
    

#### Insert data_path


	# Start to measure running time of training process
    start_time = time.time()

	# training
    classifier.fit(X_train_std, y_train)
    print("Time for training is  %s seconds" % (time.time() - start_time))

	# predict
    y_pred = classifier.predict(X_test)
    print('Accuracy is: ' + str(classifier.score(X_test, y_test)))