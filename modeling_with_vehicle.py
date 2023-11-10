import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pydotplus
import matplotlib.pyplot as plt
import statsmodels.api as sm

features = [
    'Cross-Street Type',
    'Collision Type',
    'Weather',
    'Surface Condition',
    'Light',
    'Traffic Control',
    'Driver At Fault',
    'Driver Distracted By',
    'Vehicle Damage Extent',
    'Vehicle First Impact Location',
    'Vehicle Second Impact Location',
    'Vehicle Body Type',
    'Vehicle Movement',
    'Speed Limit',
    'Injury Severity'
]
classnames = [
    'NO APPARENT INJURY',
    'POSSIBLE INJURY',
    'SUSPECTED MINOR INJURY',
    'SUSPECTED SERIOUS INJURY',
    'FATAL INJURY'
]
crash_data = pd.read_csv("imputed_Data.csv")
crash_data = crash_data[crash_data['Driverless Vehicle'] == 'No']
crash_data = crash_data[crash_data['Parked Vehicle'] == 'No']
data = crash_data[features]
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])
data_x = data.drop('Injury Severity', axis = 1)
data_y = data['Injury Severity']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 42)
'''
### Decision tree
decisiontree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42, splitter = 'random', max_depth = 10, min_samples_split = 10)
decisiontree.fit(x_train, y_train)
dot_data = tree.export_graphviz(decisiontree, out_file = None, feature_names = features[:-1], class_names = classnames, max_depth = 1)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("decisiontree_with_vehicle.png")
node_impurities = decisiontree.tree_.impurity
feature_importances = decisiontree.feature_importances_
for node, impurity in enumerate(node_impurities):
    print(f"node {node}: entropy = {impurity:.4f}")
n_features = data_x.shape[1]
for node in range(len(node_impurities)):
    print(f"node {node}:")
    for feature in range(n_features):
        print(f"features {feature}: entropy = {feature_importances[node * n_features + feature]:.4f}")

y_pred1 = decisiontree.predict(x_test)
acc1 = accuracy_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1, average = 'weighted')
precision1 = precision_score(y_test, y_pred1, average = 'weighted')
recall1 = recall_score(y_test, y_pred1, average = 'weighted')
'''
'''
### Random forest
n_trees = 10
rf = RandomForestClassifier(n_estimators = n_trees, criterion = 'entropy', max_depth = 10, min_samples_split = 10)
rf.fit(x_train, y_train)
y_pred2 = rf.predict(x_test)
acc2 = accuracy_score(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2, average = 'weighted')
precision2 = precision_score(y_test, y_pred2, average = 'weighted')
recall2 = recall_score(y_test, y_pred2, average = 'weighted')
'''
'''
### Linear regression
def stepwise_regression(X, y):
    selected_features = []
    while len(selected_features) < len(X.columns):
        remaining_features = [feature for feature in X.columns if feature not in selected_features]
        best_p_value = None
        best_feature = None
        for feature in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
            p_value = model.pvalues[feature]
            if best_p_value is None or p_value < best_p_value:
                best_p_value = p_value
                best_feature = feature
        if best_p_value < 0.05:  # 设置一个显著性水平来控制特征的入选
            selected_features.append(best_feature)
            print("new feature: ", best_feature)
            print("current p_value: ", best_p_value)
        else:
            break
    return selected_features
# selected_features = stepwise_regression(x_train, y_train)
features_selected = [
    'Cross-Street Type',
    'Light',
    'Traffic Control',
    'Driver At Fault',
    'Driver Distracted By',
    'Vehicle Damage Extent',
    'Vehicle Second Impact Location',
    'Vehicle Body Type',
    'Vehicle Movement',
    'Speed Limit'
]
x_train_selected = x_train[features_selected]
x_test_selected = x_test[features_selected]
lr = LinearRegression()
lr.fit(x_train_selected, y_train)
pred = lr.predict(x_test_selected)
y_pred3 = []
for i in pred:
    if i >= 4.5:
        y_pred3.append(5)
    elif i >= 3.5:
        y_pred3.append(4)
    elif i >= 2.5:
        y_pred3.append(3)
    elif i >= 1.5:
        y_pred3.append(2)
    else:
        y_pred3.append(1)
y_pred3 = np.array(y_pred3)
acc3 = accuracy_score(y_test, y_pred3)
f1_3 = f1_score(y_test, y_pred3, average = 'weighted')
precision3 = precision_score(y_test, y_pred3, average = 'weighted')
recall3 = recall_score(y_test, y_pred3, average = 'weighted')
'''
'''
### Logistic regression
logisticregression = LogisticRegression(multi_class = 'multinomial', random_state = 42, max_iter = 1000)
logisticregression.fit(x_train, y_train)
y_pred4 = logisticregression.predict(x_test)
acc4 = accuracy_score(y_test, y_pred4)
f1_4 = f1_score(y_test, y_pred4, average = 'weighted')
precision3 = precision_score(y_test, y_pred4, average = 'weighted')
recall3 = recall_score(y_test, y_pred4, average = 'weighted')
'''

### Neural network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out
input_size = 14
hidden_size1 = 10
hidden_size2 = 5
num_classes = 5
model = Net(input_size, hidden_size1, hidden_size2, num_classes)
criterion = nn.CrossEntropyLoss()  # 适用于多分类问题
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
num_epochs = 1000
x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
for epoch in range(num_epochs):
    outputs = model(torch.Tensor(x_train))
    loss = criterion(outputs, torch.LongTensor(y_train))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
model.eval()
x_test_tensor = torch.Tensor(x_test)
with torch.no_grad():
    outputs = model(x_test_tensor)
_, predicted = torch.max(outputs, 1)
y_pred5 = predicted.numpy()
acc5 = accuracy_score(y_test, y_pred5)
f1_5 = f1_score(y_test, y_pred5, average = 'weighted')
precision5 = precision_score(y_test, y_pred5, average = 'weighted')
recall5 = recall_score(y_test, y_pred5, average = 'weighted')





