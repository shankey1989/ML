
######################################################################################################################
from sklearn import tree

#Simple decision tree classifier https://www.youtube.com/watch?v=cKxRvEZd3Mw&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=7
#smooth = 1, bumpy = 0
features = [[140,1],[130,1],[150,0],[170,0]]
#apple = 0, orange = 1
labels = [0,0,1,1]

#Create empty tree
clf = tree.DecisionTreeClassifier()

#Edit tree using input and output
clf = clf.fit(features, labels)

#Predict output for a list of inputs
# print(clf.predict([[160,0],[135,1]]))

######################################################################################################################
#Simple decision tree classifier for iris dataset https://www.youtube.com/watch?v=tNa99PG8hR8
from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data)
# print(iris.target)
# print(iris.data[0])
# print(iris.data[0][0])
# print(type(iris))
# print(type(iris.data))

test_ids = [0,5,50,55,100,105]

#Split into training and testing datasets
train_target = np.delete(iris.target,test_ids)
train_data = np.delete(iris.data,test_ids, axis = 0)

test_target = iris.target[test_ids]
test_data = iris.data[test_ids]

#Create empty tree
clf = tree.DecisionTreeClassifier()

#Edit tree using input and output
clf = clf.fit(train_data, train_target)
print("Actual")
print(clf.predict(test_data))
print("Predicted")
print(clf.predict(test_data))

#Visualizing the Decision tree model
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot

dot_data = StringIO()
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f, feature_names = iris.feature_names, class_names = iris.target_names, filled = True, rounded = True, impurity = False)
print("Actuals\t\t:" , test_target)
print("Predicted\t:" , clf.predict(test_data))

# graph.write_pdf(r"C:\Users\Rabindranath KB\Documents\GitHub\ML\iris.pdf")
# Image(graph.create_png())
#
#
# "C:\Program Files\Anaconda\Library\bin\graphviz\dot.exe" -Tpdf iris.dot -o iris.pdf
#
# print(type(graph))
# print(graph[0])
#
