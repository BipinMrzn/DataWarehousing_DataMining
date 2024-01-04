import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image

# Load the dataset from a CSV file
df = pd.read_csv('loan_risk.csv')

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy_val

def information_gain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    information_gain_val = total_entropy - weighted_entropy
    return information_gain_val

def ID3(data, original_data, features, target_attribute_name, parent_node_class=None):
    # Base cases
    if len(np.unique(data[target_attribute_name])) == 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    # Select the best feature to split on
    best_feature = choose_best_feature(data, features, target_attribute_name)
    
    # Create the tree structure
    tree = {best_feature: {}}
    
    # Remove the best feature from the feature space
    features = [i for i in features if i != best_feature]

    # Grow the tree by recursively calling the ID3 algorithm on the subdatasets
    for value in np.unique(data[best_feature]):
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)
        tree[best_feature][value] = subtree

    return tree

def choose_best_feature(data, features, target_attribute_name):
    information_gains = [information_gain(data, feature, target_attribute_name) for feature in features]
    best_feature_index = np.argmax(information_gains)
    return features[best_feature_index]

# Usage example
features = df.columns[:-1].tolist()
target_attribute = df.columns[-1]
tree = ID3(df, df, features, target_attribute)

# Print the decision tree
import pprint
pprint.pprint(tree)