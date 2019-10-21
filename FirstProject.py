## MY first machine learning project
# Load libraries
import pandas
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
##
#
#
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
# Shuffle data - good for sorted data
dataset = dataset.sample(frac=1)
#making data_only for training
data_only = dataset.drop('class',axis=1).values
#
#making classes_only for training
classes_only = dataset['class'].values
#
#setting validation size and slicing what for training and what for testing
validation_size = 0.8
training_row_count = int(dataset.__len__() * validation_size)
test_row_count = dataset.__len__() - training_row_count
#
data_learn = dataset.head(training_row_count).drop('class',axis=1).values
classes_learn = dataset['class'].head(training_row_count).values
#
test_data = dataset.tail(test_row_count).drop('class',axis=1).values
test_classes = dataset['class'].tail(test_row_count).values
knn = KNeighborsClassifier()
#train dataset ?
knn.fit(data_learn, classes_learn)
#
success_count = 0
failure_count = 0
# manual test of prediction after training
for test,result in zip(test_data,test_classes):
    test_list_item=[]
    test_list_item.append(test)
    predictions = knn.predict(test_list_item)
    if predictions == result:
        success_count+=1
        print("prediction succeed: " + predictions)
    else:
        failure_count+=1
        print("prediction failed: " + predictions + " should be: " + result)
print("prediction results: " + str(success_count) + " success and " + str(failure_count) +"  failed.")
