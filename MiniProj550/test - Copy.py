#%%
import csv
import math
import numpy as np
import pandas as pd
from statistics import median, mode, mean
from collections import Counter
from enum import Enum
#%%

class AttrType(Enum):
    cat = 0  # categorical (qualitative) attribute
    num = 1  # numerical (quantitative) attribute
    target = 2  # target label


class NodeType(Enum):
    internal = 1
    root = 0
    leaf = 2


class SplitType(Enum):
    bin = 0  # binary split
    multi = 1  # multi-way split


class OperatorType(Enum):
    leq = 0  # Less than or equal to.. <=
    gt = 1  # Greater Than... >
    eq = 2 # Equal to =
 

class Attribute(object):
    def __init__(self, label, type):
        self.label = label
        self.stat = None  # holds mean for numerical and mode for categorical attributes
        self.type = None
        
        if type is AttrType :
            self.type = type
        elif type == 'O' or type == 'object':
            self.type = AttrType.cat
        elif type == 'int64' or type == 'float64' : 
            self.type = AttrType.num
        else:
            raise Exception(
                type,
                 f'Data type was {type} was not recognized or is not a proopper Attribute')

    def __str__(self):
        return f"{self.label}:{self.type}"
    
    def __unicode__(self):
        return f"{self.label}:{self.type}"
    
    def __repr__(self):
        return f"{self.label}:{self.type}"


class Splitting(object):
    def __init__(self, attr, infogain, split_type, cond, splits):
        self.attr = attr  # attribute ID (index in ATTR)
        self.infogain = infogain  # information gain if splitting is done on this attribute
        self.split_type = split_type  # one of SplitType
        self.cond = cond  # splitting condition, i.e., values on outgoing edges
        # list of training records (IDs) for each slitting condition
        self.splits = splits


class Node(object):
    def __init__(self, id, type,attr, parent_id, children=None, edge_value=None, val=None, split_type=None,operator_type = None, split_cond=None,
                 infogain=None):
        self.id = id  # ID (same as the index in DT.model list)
        self.type = type  # one of NodeType
        self.parent_id = parent_id  # ID of parent node (None if root)
        self.children = children  # list of IDs of child nodes
        self.attr = attr
        self.operator_type = operator_type
        # the value of the incoming edge (only if not root node)
        self.edge_value = edge_value
        self.val = val  # if root or internal node: the attribute that is compared at that node; if leaf node: the target value
        self.split_type = split_type  # one of SplitType
        # splitting condition (median value for binary splits on numerical values; otherwise a list of categorical values (corresponding to child nodes))
        self.split_cond = split_cond
        self.infogain = infogain
        self.str = f"Node id: {self.id}-{self.type} EdgeValue={self.edge_value}, Value={self.val} Type:{self.split_type} Conditions={self.split_cond} Gain:{self.infogain}" 

    def operator_type_str(self):
        if self.operator_type == OperatorType.leq:
            return "<="
        if self.operator_type == OperatorType.gt:
            return ">"
        if self.operator_type == OperatorType.eq:
            return "="

    def __str__(self):
        return self.str
    
    def __unicode__(self):
        return self.str
    
    def __repr__(self):
        return self.str

    def append_child(self, node_id):
        self.children.append(node_id)

#%%

class DT(object):
    def __init__(self):
        self.data = None  # training data set (loaded into memory)
        self.model = None  # decision tree model
        self.default_target = 0 # default target class
        self.target_attribute = None
        self.attributes = None
    
    def load_data(self):
        columns = [x.label for x in self.attributes]
        self.data = pd.read_csv(INFILE, names=columns)
        self.data = self.data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        self.target_attribute = [x for x in self.attributes if  x.type == AttrType.target ][0]
        self.default_target = self.attributes.index(self.target_attribute)

    def calculate_entropy(self,attrs,subset):
        entropy = 0.0
        sample_size = len(subset)
        target_col = subset.iloc[:,attrs.index(self.target_attribute)]
        unique_vals = target_col.unique()
        for unique_val in unique_vals:
            unique_val_count = target_col[target_col == unique_val].count()
            p_unique_val = unique_val_count / sample_size
            entropy += -p_unique_val * math.log(p_unique_val,2)
        return entropy

    def calculate_gain(self,attrs,subset,idx_attribute):
        entropy = self.calculate_entropy(attrs,subset)
        sample_size = len(subset)
        attr_col = subset.iloc[:, idx_attribute]
        unique_vals = attr_col.unique()
        for unique_val in unique_vals:
            subset_data = subset[subset.iloc[:,idx_attribute] == unique_val]
            entropy_val = self.calculate_entropy(attrs,subset_data)
            unique_val_count = attr_col[attr_col == unique_val].count()
            p_unique_val = unique_val_count / sample_size
            entropy += -p_unique_val * entropy_val
        return entropy


    def __mean_squared_error(self, records):
        """
        Calculates mean squared error for a selection of records.

        :param records: Data records (given by indices)
        """
        result = 0.0
        if records.empty:
            return result
        mean = records.mean()
        for record in records:
            result += (record - mean)**2
        return result/len(records)

    def __find_best_attr(self,attrs, subset):
        """
        Finds the attribute with the largest gain.
        :param attrs: Set of attributes
        :param subset: Training set (Pandas dataFrame with corresponding subset)
        :return:
        """
        #mse_p = self.__mean_squared_error(subset)  # parent's MSE
        splittings = []  # holds the splitting information for each attribute
        split_mode = None
        for attr in attrs:
            attr_idx = attrs.index(attr)
            splits = {}  # record IDs corresponding to each split
            # splitting condition depends on the attribute type
            if attr.type == AttrType.target:  # skip target attribute
                continue
            elif attr.type == AttrType.cat:  # categorical attribute, multi-way split on each possible value
                split_mode = SplitType.multi
                split_cond = self.data.iloc[:,attr_idx].unique()
            elif attr.type == AttrType.num:  # numerical attribute => binary split on median value
                split_mode = SplitType.bin
                unique_vals = self.data.iloc[:,attr_idx].unique()
                unique_vals.sort()
                med_vals = {}
                for i in range(0,unique_vals.size,2):
                    med_val = (unique_vals[i] + unique_vals[i]) /2
                    ss = subset[ subset.iloc[:,attr_idx] <= med_val]
                    gain = self.calculate_gain(attrs,ss, attr_idx)
                    med_vals.update({med_val:gain})
                
                split_cond = sorted(med_vals.items(), key=lambda x: x[1], reverse=True)[0][0]
                print(f"best split cond is {split_cond}")
                #split_cond = self.data.iloc[:,attr_idx].median()
            infogain = self.calculate_gain(attrs,subset, attr_idx)
            splitting = Splitting(attr, infogain, split_mode, split_cond, splits)
            splittings.append(splitting)

        # find best splitting
        best_splitting = sorted(splittings, key=lambda x: x.infogain, reverse=True)[0]
        return best_splitting

    def __add_node(self, parent_id,attr, node_type=NodeType.internal, edge_value=None, val=None, split_type=None,
                   operator_type=None,split_cond=None):
        """
        Adds a node to the decision tree.

        :param parent_id:
        :param node_type:opera
        :param edge_value:
        :param val:
        :param split_type:
        :param split_cond:
        :return:
        """
        node_id = len(self.model)  # id of the newly assigned node
        if not self.model:  # the tree is empty
            node_type = NodeType.root

        node = Node(
            node_id,
            node_type,
            attr, 
            parent_id, 
            children=[], 
            edge_value=edge_value, 
            val=val,
            split_type=split_type,
            operator_type=operator_type,
            split_cond=split_cond)
        self.model.append(node)

        # also add it as a child of the parent node
        if parent_id is not None:
            self.model[parent_id].append_child(node_id)

        return node_id
    
    def __id3(self, attrs, subset, parent_id=None, value=None,operator_type = None):
        """
        Function ID3 that returns a decision tree.

        :param attrs: Set of attributes
        :param records: Training set (list of record ids)
        :param parent_id: ID of parent node
        :param value: Value corresponding to the parent attribute, i.e., label of the edge on which we arrived to this node
        :return:
        """
        # empty training set or empty set of attributes => create leaf node with default class
        if subset.empty or not attrs:
            self.__add_node(parent_id,None, node_type=NodeType.leaf, edge_value=value, val=self.default_target,operator_type=operator_type)
            return
        # if all records have the same target value => create leaf node with that target value
        target_idx = attrs.index(self.target_attribute)
        uniquevals = subset.iloc[:, target_idx].unique()
        same = len(uniquevals)
        if same == 1:
            self.__add_node(parent_id,None, node_type=NodeType.leaf, edge_value=value, val=uniquevals,operator_type=operator_type)
            return

        # find the attribute with the largest gain
        splitting = self.__find_best_attr(attrs, subset)
        splitting_idx = self.attributes.index(splitting.attr)
        print(f"Best splitting attribute is {splitting.attr.label}")
        # add node
        node_id = self.__add_node(parent_id,splitting.attr, edge_value=target_idx, val=splitting_idx, split_type=splitting.split_type,
                                  split_cond=splitting.cond,operator_type=operator_type)
        #Call tree construction recursively for each split
        split_attrs = [x for x in attrs if x is not splitting.attr]
        split_attr_idx = attrs.index(splitting.attr)
        print('New subset Splitting attributes',[x.label for x in split_attrs])
        if splitting.split_type == SplitType.bin:
            ss1 = subset[ subset.iloc[:,split_attr_idx] <= splitting.cond]
            ss1 = ss1.drop(columns=ss1.columns[split_attr_idx])
            ss2 = subset[ subset.iloc[:,split_attr_idx] > splitting.cond]
            ss2 = ss2.drop(columns=ss2.columns[split_attr_idx])
            self.__id3(split_attrs,ss1,node_id,splitting.cond,operator_type=OperatorType.leq)
            self.__id3(split_attrs,ss2,node_id,splitting.cond,operator_type=OperatorType.gt)
        elif splitting.split_type  == SplitType.multi:
            for split_cond in splitting.cond:
                print(f"finding subset for category [{splitting.attr.label}-{split_cond}]")
                ss = subset[subset.iloc[:, split_attr_idx] == split_cond]
                ss = ss.drop(columns=ss.columns[split_attr_idx])
                self.__id3(split_attrs,ss,node_id,split_cond,operator_type=OperatorType.eq)
        


    def build_model(self):
        self.load_data()
        self.model = []  # holds the decision tree model, represented as a list of nodes
        # Get majority class
        #   Note: Counter returns a dictionary, most_common(x) returns a list with the x most common elements as
        #         (key, count) tuples; we need to take the first element of the list and the first element of the tuple
        #self.default_target = self.data.iloc[:,self.target_attribute.index].mean()
        self.__id3(self.attributes, self. data)

    def apply_model(self, record):
        node = self.model[0]
        oldNode = None
        while node.type != NodeType.leaf:
            oldNode = node
            print(f"appliyting for node {node.id}-{node.attr.label}-{node.split_cond}")
            attr_idx = self.attributes.index(node.attr)
            record_val = record[attr_idx]
            if node.split_type == SplitType.bin :
                if record_val <= node.split_cond :
                   node = self.model[ node.children[0] ]
                else :
                    node = self.model[ node.children[1] ]
            if node.split_type == SplitType.multi : 
                for child_node_idx in node.children:
                    child_node = self.model[child_node_idx]
                    print(f"Test {record_val} with {child_node.edge_value}")
                    if child_node.edge_value == record_val :
                        node = child_node
                        break
            if oldNode == node:
                break
        return node.val
    
    def predict(self, records):
        predictions = []
        for record in records:
            pred_val = self.apply_model(record)
            predictions.append(predictions)
        return predictions


    def print_model(self, node_id=0, level=0):
        node = self.model[node_id]
        indent = "  " * level
        if node.type == NodeType.leaf:
            print(indent + str(node.edge_value) + " [Leaf node] class=" + str(node.val))
        else:
            cond = f" {node.operator_type_str()} " + str(node.split_cond) if node.attr.type == AttrType.num else f" {node.operator_type_str()} ? "
            if node.type == NodeType.root:
                print("[Root node] '" + node.attr.label + "'" + cond)
            else:
                print(indent + str(node.edge_value) + " [Internal node] '" + node.attr.label + "'" + cond)
            # print tree for child notes recursively
            for n_id in node.children:
                self.print_model(n_id, level + 1)

class RF(object):
    def __init__(self):
        self.data = None  # training data set (loaded into memory)
        self.trees = None  # decision trees
        self.attributes = None
        
    def load_data(self,file_name):
        self.data = pd.read_csv(INFILE)
        self.data = self.data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        self.target_attribute = [x for x in ATTR if  x.type == AttrType.target ][0]
        self.default_target = ATTR.index(self.target_attribute)
        
    def __subsampling(self, train_set, sample_size_ratio):
        sample_number = round(len(self.data) * sample_size_ratio)
        # TODO: generate a subsample with replacement
        
    def build_model(self, train_set, split_conditions, sample_size_ratio, number_of_trees):
        for i in range(number_of_trees):
            pass
            #sample = self.__subsampling(train_set, sample_size_ratio)
            #tree = DT() # build a tree with sample data and split conditions
            #self.trees.append(tree)
            
    def predict(self, test_set):
        rf_predictions = []
        for row in test_set:
            predictions = [tree.predict(row) for tree in self.trees]
            rf_predictions.append(np.mean(predictions))
        return rf_predictions

# %%
def createSubmission(test_ids, predictions):
    sub = pd.DataFrame()
    sub['Id'] = test_ids
    sub['SalePrice'] = predictions
    sub.to_csv('submission.csv',index=False)

# %%
file_name = "data/housing_price_test.csv"
data = pd.read_csv(file_name)
#%%
attributes = []
for col in data.columns :
    if data[col].dtype == 'object' :
        data[col] = data[col].str.strip() 
    attr = Attribute(col,data[col].dtype)
    attributes.append(attr)
    


#%%
# dt = DT()
# print("Build model:")
# dt.build_model()
# dt.print_model()
# ##Outlook ,Temperature,/Humidity,Windy
# print("\nApply model:")
# print(dt.apply_model(['sunny', 85, 85, 'false']))
# print(dt.apply_model(['overcast', 75, 85, 'true']))
# print(dt.apply_model(['rain', 75, 85, 'false']))
#%%
INFILE = r"C:\Users\amoreno15\OneDrive - Universitetet i Stavanger\Semester-2\DAT55O-120V-Data Mining\asahicantu-labs\assignment2\data\example.csv"

ATTR = [
    Attribute("Outlook", AttrType.cat), 
    Attribute("Temperature", AttrType.num),
    Attribute("Humidity", AttrType.num), 
    Attribute("Windy", AttrType.cat), 
    Attribute("Play?", AttrType.target)
]

def main():
    rf = RF()
    print("Build model:")
    rf.build_model()
    
    createSubmission(test_ids, rf.predict(test_data))

if __name__ == "__main__":
    main()


# %%

# %%

# %%


# %%
