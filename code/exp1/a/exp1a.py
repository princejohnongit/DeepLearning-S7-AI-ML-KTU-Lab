import math
import pandas as pd

# Calculate entropy
def entropy(target_col):
    elements, counts = pd.Series(target_col).value_counts().index, pd.Series(target_col).value_counts().values
    entropy = 0
    for i in range(len(elements)):
        probability = counts[i] / sum(counts)
        entropy -= probability * math.log2(probability)
    return entropy

# Calculate Information Gain
def info_gain(data, split_attribute_name, target_name="PlayTennis"):
    total_entropy = entropy(data[target_name])
    vals, counts = data[split_attribute_name].value_counts().index, data[split_attribute_name].value_counts().values
    weighted_entropy = 0
    for i in range(len(vals)):
        subset = data[data[split_attribute_name] == vals[i]]
        weighted_entropy += (counts[i] / sum(counts)) * entropy(subset[target_name])
    gain = total_entropy - weighted_entropy
    return gain

# ID3 Algorithm
def ID3(data, original_data, features, target_attribute_name="PlayTennis", parent_node_class=None):
    # If all target_values have same value, return that value
    if len(data[target_attribute_name].unique()) == 1:
        return data[target_attribute_name].iloc[0]

    # If dataset is empty
    if len(data) == 0:
        return parent_node_class

    # If no more features, return mode of target
    if len(features) == 0:
        return data[target_attribute_name].mode()[0]

    # Set default value for parent node
    parent_node_class = data[target_attribute_name].mode()[0]

    # Choose best feature
    item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
    best_feature_index = item_values.index(max(item_values))
    best_feature = features[best_feature_index]

    # Build tree
    tree = {best_feature: {}}

    feature_values = data[best_feature].unique()
    for value in feature_values:
        sub_data = data[data[best_feature] == value].drop(columns=[best_feature])
        subtree = ID3(sub_data, original_data, [f for f in features if f != best_feature], target_attribute_name, parent_node_class)
        tree[best_feature][value] = subtree

    return tree

# Predict using the decision tree
def predict(query, tree, default='Yes'):
    for key in list(query.keys()):
        if key in tree:
            try:
                result = tree[key][query[key]]
            except:
                return default
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result
    return default

# Data
data = pd.DataFrame([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
], columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target = 'PlayTennis'

# Build Tree
tree = ID3(data, data, features, target)
print("Decision Tree:")
print(tree)

s1=input() #sunny
s2=input() #cool
s3=input() #High
s4=input() #strong
sample = {'Outlook': s1, 'Temperature': s2, 'Humidity': s3, 'Wind': s4}


# Classify a new sample
#sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
result = predict(sample, tree)
print("\nPrediction for sample", sample, ":", result)
