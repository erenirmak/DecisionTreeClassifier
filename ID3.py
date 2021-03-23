#Read data and preprocessing
import pandas as pd
import numpy as np
#Encoding string values into integer to be able to pass Decision Tree Classifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#Decision Tree Classifier and creating Graph
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz, plot_tree
#Visualizing the graph
import pydotplus
from matplotlib import pyplot as plt

#Read data from .CSV input file
data_table = pd.read_csv('./input/data.csv')

print(data_table) #print the table

#Transform the datble to DataFrame for preprocessing
data_table = pd.DataFrame(data_table)
#Read columns of the table
print(data_table.columns)

#Delete Day column from the table
data_table = data_table.drop(columns = 'Day')
print(data_table)

#Turn DataFrame table to numpy array format to split the values into input & output for training
dataVal = np.array(data_table)
print("dataVal:\n", dataVal)

data = np.array(data_table)[:,:-1] #input data (without last column - PlayTennis)
target = np.array(data_table)[:,-1] #output data (only last column - PlayTennis

#check the values
print("Data:\n", data)
print("Target:\n", target)

#One-Hot-Encoder encodes string categorical values
#into integers to pass them to Decision Tree Classifier
encoder = OneHotEncoder() #drop = 'first' -> Handled from the DataFrame table (deleted Day column)
encoder.fit(data)
X = encoder.transform(data)
#X2 = encoder.get_feature_names(input_features = data_table.columns[:-1])
#print(X2)
#Label Encoder encodes target values into integers
encoder2 = LabelEncoder()
encoder2.fit(target)
Y = encoder2.transform(target)

#Decision Tree classifier
tree_clf = DecisionTreeClassifier(criterion = 'entropy') #, min_weight_fraction_leaf = 0.1
tree_clf.fit(X, Y)

export_graphviz(
        tree_clf, #Decision Tree Classifier object
        out_file='./output/data_table.dot', #.dot file that contains graph
        feature_names = encoder.get_feature_names(input_features = data_table.columns[:-1]), #feature names pulled from One-Hot-Encoder
        class_names = encoder2.classes_, #class names of encoder2
        rounded=True, #shape of the graph
        filled=True #color of the graph
    )

#Create graph.png file
graph = pydotplus.graphviz.graph_from_dot_file('./output/data_table.dot')
graph.write_png('./output/data_table_file.png')

plot_tree(
            tree_clf,
            feature_names = encoder.get_feature_names(input_features = data_table.columns[:-1]), #feature names pulled from One-Hot-Encoder
            class_names = encoder2.classes_, #class names of encoder2
            rounded=True, #shape of the graph
            filled=True, #color of the graph
        )
plt.show()
