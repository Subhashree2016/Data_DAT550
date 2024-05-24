#%%
import csv
import math
import numpy as np
import pandas as pd
from statistics import median, mode, mean
from collections import Counter
from enum import Enum
import os

#%%

class AttrType(Enum):
    cat = 0  # categorical (qualitative) attribute
    num = 1  # numerical (quantitative) attribute


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
    def __init__(self, label, type,is_target,stat):
        self.label = label
        self.stat = stat  # holds mean for numerical and mode for categorical attributes
        self.is_target = is_target
        self.type = type

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
        self.str = f"id: {self.id}type:{self.type},Attr:{self.attr} EdgeValue:{self.edge_value}, Value={self.val} Type:{self.split_type} Conditions={self.split_cond} Gain:{self.infogain}" 

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


class DT(object):
    def __init__(self, data,attributes,sampling_attr_size):
        self.data = data  # training data set (loaded into memory)
        self.model = None  # decision tree model
        self.default_target = 0 # default target class
        self.attributes = attributes
        self.target_attribute = [x for x in self.attributes if  x.is_target ][0]
        self.default_target = self.attributes.index(self.target_attribute)
        self.sampling_attr_size = sampling_attr_size 

    def __subsampling_attributes(self,attrs):
        attribute_length = len(attrs)
        attr_indexes = np.random.choice(attribute_length,self.sampling_attr_size,replace=False).tolist()
        #attr_indexes.append(attribute_length -1) #append target Node
        return [attrs[i] for i in attr_indexes if i != attrs.index(self.target_attribute)]

    def calculate_entropy(self,subset):
        entropy = 0.0
        sample_size = len(subset)
        target_col = subset[self.target_attribute.label]
        unique_vals = target_col.unique()
        for unique_val in unique_vals:
            unique_val_count = target_col[target_col == unique_val].count()
            p_unique_val = unique_val_count / sample_size
            entropy += -p_unique_val * math.log(p_unique_val,2)
        return entropy

    def calculate_gain(self,subset,attr):
        entropy = self.calculate_entropy(subset)
        sample_size = len(subset)
        attr_col = subset[attr.label]
        unique_vals = attr_col.unique()
        for unique_val in unique_vals:
            subset_data = subset[subset[attr.label] == unique_val]
            entropy_val = self.calculate_entropy(subset_data)
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
            #attr_idx = attrs.index(attr)
            splits = {}  # record IDs corresponding to each split
            # splitting condition depends on the attribute type
            if attr.is_target :# skip target attribute
                continue
            elif attr.type == AttrType.cat:  # categorical attribute, multi-way split on each possible value
                split_mode = SplitType.multi
                split_cond = subset[attr.label].unique()
            elif attr.type == AttrType.num:  # numerical attribute => binary split on median value
                split_mode = SplitType.bin
                split_cond = subset[attr.label].mean()
                # unique_vals = self.data[attr.label].unique()
                # unique_vals.sort()
                # med_vals = {}
                # for i in range(0,unique_vals.size,2):
                #     med_val = (unique_vals[i] + unique_vals[i]) /2
                #     ss = subset[ subset[attr.label] <= med_val]
                #     gain = self.calculate_gain(ss, attr)
                #     med_vals.update({med_val:gain})
                
                #split_cond = sorted(med_vals.items(), key=lambda x: x[1], reverse=True)[0][0]
                #print(f"\t best split cond is {split_cond}")
                #print(f"\t Calculating gain... {split_cond}")
                #split_cond = self.data.iloc[:,attr_idx].median()
            infogain = self.calculate_gain(subset, attr)
            splitting = Splitting(attr, infogain, split_mode, split_cond, splits)
            splittings.append(splitting)

        # find best splitting
        best_splitting = sorted(splittings, key=lambda x: x.infogain, reverse=True)
        return best_splitting[0]

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
    
    def __id3(self, attrs,subset, parent_id=None,edge_value=None, value=None,operator_type = None,subsample_trees = False):
        """
        Function ID3 that returns a decision tree.

        :param attrs: Set of attributes
        :param records: Training set (list of record ids)
        :param parent_id: ID of parent node
        :param value: Value corresponding to the parent attribute, i.e., label of the edge on which we arrived to this node
        :return:
        """

        sample_attrs = attrs
        if subsample_trees:
            sample_attrs = self.__subsampling_attributes(attrs)

        #print(f"{subset.columns}")

        # empty training set or empty set of attributes => create leaf node with default class
        if subset.empty or not sample_attrs or len(sample_attrs) == 0:
            self.__add_node(parent_id,None, node_type=NodeType.leaf, edge_value=edge_value, val=value,operator_type=operator_type)
            return
        # if all records have the same target value => create leaf node with that target value
        if len(attrs)  == 1:
            node_value = subset[self.target_attribute.label].mean()
            self.__add_node(parent_id,None, node_type=NodeType.leaf, edge_value=edge_value, val=node_value,operator_type=operator_type)
            return

        # find the attribute with the largest gain
        splitting = self.__find_best_attr(sample_attrs, subset)
        
        # add node
        node_id = self.__add_node(parent_id,splitting.attr,node_type=NodeType.internal, edge_value=edge_value, val=value, split_type=splitting.split_type,
                                  split_cond=splitting.cond,operator_type=operator_type)
        #Call tree construction recursively for each split
        split_attrs = [x for x in attrs if x is not splitting.attr]
        #print(split_attrs)
        #print('New subset Splitting attributes',[x.label for x in split_attrs])
        split_label = splitting.attr.label
        subset_value = round(subset[self.target_attribute.label].mean())
        if splitting.split_type == SplitType.bin:
            ss1 = subset[ subset[split_label] <= splitting.cond]
            ss1 = ss1.drop([split_label],axis=1)
            ss2 = subset[ subset[split_label] > splitting.cond]
            ss2 = ss2.drop(split_label,axis=1)
            self.__id3(split_attrs,ss1,node_id,edge_value=subset_value,value=subset_value,operator_type=OperatorType.leq)
            self.__id3(split_attrs,ss2,node_id,edge_value=subset_value,value=subset_value,operator_type=OperatorType.gt)
        elif splitting.split_type  == SplitType.multi:
            for split_cond in splitting.cond:
                #print(f"finding subset for category [{splitting.attr.label}-{split_cond}]"),
                ss = subset[subset[split_label] == split_cond]
                ss = ss.drop([split_label],axis=1)
                if ss.empty:
                    print("WARNING!!! Empty subset not allowed")
                self.__id3(split_attrs,ss,node_id,edge_value=subset_value,value=subset_value,operator_type=OperatorType.eq)
        
    def build_model(self,subsample_trees):
        self.model = []  # holds the decision tree model, represented as a list of nodes
        self.__id3(self.attributes, self.data,subsample_trees=subsample_trees)

    
    def predict(self, record):
        node = self.model[0]
        oldNode = None
        while node.type != NodeType.leaf:
            oldNode = node
            #print(f"\t Applying DT model for  node {node.id}-{node.attr.label}-{node.split_cond}")
            record_val = record[node.attr.label]
            if node.split_type == SplitType.bin :
                for child_idx in node.children:
                    child_node = self.model[child_idx]
                    if child_node.operator_type == OperatorType.leq  and record_val <= node.split_cond:
                        node = child_node
                        break
                    if child_node.operator_type == OperatorType.gt  and record_val > node.split_cond:
                        node = child_node
                        break
            elif node.split_type == SplitType.multi : 
                for child_node_idx in node.children:
                    child_node = self.model[child_node_idx]
                    if child_node.val == record_val :
                        node = child_node
                        break
            if oldNode == node:
                break
        return node.val


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
    def __init__(self,data):
        self.data = data  # training data set (loaded into memory)
        self.forest = [] # decision trees
        self.attributes =[]
        self.MAX_CATEGORY_SIZE = 10
       
        col_length = len(self.data.columns) 
        for col_idx in range(col_length):
            col = self.data.columns[col_idx]
            attr_type = AttrType.num
            stat = data.mean()
            #Lets consider a max number of minimum values = 20 to classify the attribute as Categorical 
            colUniqueLen = len(data[col].unique())
            if colUniqueLen <= self.MAX_CATEGORY_SIZE:
                attr_type = AttrType.cat
                stat = data.mode()
            is_target = col_idx == col_length - 1 
            attr = Attribute(col,attr_type,is_target,stat)
            self.attributes.append(attr)
             

    def __subsampling(self, data, sample_size_ratio):
        data_length = len(data)
        sample_number = round(data_length * sample_size_ratio)
        sample_indexes = np.random.randint(low=0,high= data_length, size=sample_number)
        subsample  = data.iloc[sample_indexes,:]
        return subsample
        
    def build_model(self,number_of_trees=1, sample_size_ratio=None,sample_attr_size=None):
        for i in range(number_of_trees):
            print(f'Creating tree # {i}...')
            print(f'\t Subsampling...')
            sample = self.data
            subsample_trees = number_of_trees > 1 
            if  subsample_trees or not sample_size_ratio is None:
                sample = self.__subsampling(self.data, sample_size_ratio)
            
            print(f'\t Initializing...')
            tree = DT(sample,self.attributes,sample_attr_size) 
            print(f'\t Modelling...')
            tree.build_model(subsample_trees)
            self.forest.append(tree)
            print(f'\t Model completed {len(tree.model)}')
            
    def predict(self, test_data):
        rf_predictions = pd.DataFrame(columns=['Id','SalePrice'])
        n = test_data.shape[0]
        print(f"Starting predictions ({n})...")
        for row_idx in test_data.index:
            row = test_data.loc[row_idx]
            predictions = []
            for tree in self.forest:
                prediction = tree.predict(row) 
                predictions.append(prediction)
            print(predictions)
            result = np.mean(predictions)
            rf_predictions = rf_predictions.append({'Id':row_idx,'SalePrice':result},ignore_index=True)
            #print(f"{row_idx} of {n} = {result}")
        rf_predictions = rf_predictions.astype({'Id': 'int32','SalePrice':'float32'})
        print("Prediction finished")
        return rf_predictions

    
#%%
def fill_missing_vals(data):
    for col in data.columns :
        if data[col].dtype == 'object' :
            data[col] = data[col].str.strip()
            data[col] = data[col].fillna('NA')
    #After a quuick check nulll data was found           
    # x=data.isna()
    # x = x.sum()
    # y = x[x > 0].sort_values()
    # types = data[y.keys()].dtypes
    # The rest of the values consider NA as an acceptable category, for these it will be
    # Necessary to fill the nulls
    null_cols = ['Electrical','MasVnrArea','GarageYrBlt','LotFrontage']
    for col in null_cols:
        data_col = data[col]
        null_rows = data_col.isnull() 
        non_nulls = data_col[null_rows == False]
        mean_val = None
        if data[col].dtype == 'object':
            mean_val = non_nulls.value_counts().keys()[0]
        else:
            mean_val = non_nulls.mean()
        data[col].fillna(mean_val,inplace=True)

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
            data[col]  = data[col].cat.codes

def transform_data(data):
    data['WoodDeckSF'] = data['WoodDeckSF'].apply(lambda x: 0 if x == 0  else 1)
    data['OpenPorchSF'] = data['OpenPorchSF'].apply(lambda x: 0 if x == 0  else 1)
    data['LotFrontage'] = data['LotFrontage'].apply(lambda x: round(x/10))
    data['YearBuilt'] = data['YearBuilt'].apply(lambda x: round(x/1000,2))
    data['GarageYrBlt'] = data['GarageYrBlt'].apply(lambda x: round(x/1000,2))
    data['YearRemodAdd'] = data['YearRemodAdd'].apply(lambda x: round(x/1000,2))
    data['GrLivArea'] = data['GrLivArea'].apply(lambda x: round(x/100))
    data['BsmtFinSF1'] = data['BsmtFinSF1'].apply(lambda x: 0 if x == 0 else 1)
    return data



def load_train_data(file_name,plot=False,filter=False):
    data = pd.read_csv(file_name,index_col='Id')
    fill_missing_vals(data)
    data_d = data.describe()
    ## Finding columns with the mayority of values = to 0
    zero_features = data_d.iloc[6,:] == 0
    filtered_zero_features = zero_features[zero_features]
    data.drop(filtered_zero_features.keys(),axis=1, inplace=True)
    if plot:
        data.hist(figsize=(20,20), bins=20)
        plt.show()
    corr = data.corr()
    corr_values = corr['SalePrice'].sort_values(ascending=False)
    if plot:
        fig,ax = plt.subplots(figsize=(12,9))
        sns.heatmap(corr,vmax=.8,square=True)
    ##Let's consider now correlation values above 0.3, which is a good value representing correlation
    good_correlation = corr_values >= 0.3
    feature_values = good_correlation[good_correlation]
    
    drop_columns = [c for c in  data.columns if c not in feature_values  ]
    # drop_columns.append('MasVnrArea')
    # drop_columns.append('2ndFlrSF')
    # drop_columns.append('TotRmsAbvGrd')
     

    data.drop(drop_columns,axis=1, inplace=True)
    
    corr = data.corr()
    corr_values = corr['SalePrice'].sort_values(ascending=False)
    if plot :
        sns.set(font_scale=1.25)
        fig,ax = plt.subplots(figsize=(12,9))
        sns.heatmap(corr,vmax=.8,square=True,annot=True)


    #remove outliers
    data = data[data['SalePrice'] <= 225000]
    data = data[data['SalePrice'] >= 100000]
    data = data[data['TotalBsmtSF'] <= 2500]
    data = data[data['TotalBsmtSF'] >= 600]
    data = data[data['1stFlrSF'] <= 1800]
    #data = data[data['MasVnrArea'] <= 500]
    #data = data[data['MasVnrArea'] < 300]
    data = data[data['GrLivArea'] <= 1550]
    data = data[data['GrLivArea'] > 800]
    data = data[data['FullBath'] > 0]
    data = data[data['FullBath'] < 3]
    # data = data[data['TotRmsAbvGrd'] > 3]
    # data = data[data['TotRmsAbvGrd'] < 9]
    data = data[data['OverallQual'] > 3]
    data = data[data['YearBuilt'] >= 1950]
    data = data[data['YearBuilt'] < 2006]
    data = data[data['YearRemodAdd'] < 2005]
    data = data[data['LotFrontage'] < 90]
    data = data[data['Foundation'] <= 2]
    data = data[data['BsmtFinSF1'] <= 1100]
    data = data[data['GarageCars'] < 3]
    data = data[data['WoodDeckSF'] < 350]

    data = data[data['WoodDeckSF'] < 350]
    

    data = transform_data(data)
    
    # for col in data.columns:
    #     if col != 'SalePrice':
    #         data[col] = np.around( np.log1p(data[col]),1)
    # if not filter:
    #     return data
    # for col in data.columns:
    #     unique_vals = data[col].unique()
    #     #print(col, len(unique_vals))
    #     if len(unique_vals) > 600:
    #         d = data[col].describe()
    #         mean = d['mean'] 
    #         std = d['std']
    #         min = mean - std
    #         max = mean + std
    #         data = data[data[col] >= min]
    #         data = data[data[col] <= max]
    return data
    # fig,ax = plt.subplots(figsize=(12,9))
    # sns.heatmap(corr,vmax=.8,square=True)

def load_test_data(file_name,keep_cols): 
    test_data = pd.read_csv(file_name,index_col='Id')
    data_cols = test_data.columns
    drop_cols = [col for col in data_cols if col not in keep_cols ]
    fill_missing_vals(test_data)
    test_data.drop(drop_cols,axis=1, inplace=True)
    test_data = transform_data(test_data)

    # for col in data.columns:
    #     if col != 'SalePrice':
    #         test_data[col] = np.around( np.log1p(test_data[col]),1)

    return test_data


#%%
# result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.getcwd()) for f in filenames]
# print(result) 
# print(os.getcwd())

path = "user/assignment2/"
path = ""

# train_file_name = path + "data/housing_price_train.csv"
# test_file_name = path + "data/housing_price_test.csv"
train_file_name = path + "D:\\housing_price_train.csv"
test_file_name = path + "D:\\housing_price_test.csv"

submission_file_name = path + "submission.csv"
train_data = load_train_data(train_file_name)
test_data = load_test_data(test_file_name,train_data.columns) 
rf =  RF(train_data)
# for col in train_data.columns:
#     if col != 'SalePrice': 
#         train_data.plot(x=col, y='SalePrice', style='o')
#%%
rf.build_model()
#rf.build_model()
test_results = rf.predict(test_data)
test_results.to_csv( submission_file_name,index=False)


 # %%
