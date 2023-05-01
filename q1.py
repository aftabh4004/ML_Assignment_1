import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pprint import pprint,pformat


#loaded the dataset into pandas dataframe 
def load_data():
  data = pd.read_csv('Train_D_Tree.csv')
  data= data.drop(['Restaurant'],axis=1)     #removed the 'Restaurant Name' feature fron the data
  #converting the categorical variables into binary variables it will ease in building the regression tree
  data['Extra Cheeze'] = data['Extra Cheeze'].replace({'yes':1,'no':0})
  data['Extra Mushroom'] = data['Extra Mushroom'].replace({'yes':1,'no':0})
  data['Extra Spicy'] = data['Extra Spicy'].replace({'yes':1,'no':0})
  return data



#fucntion to check whether the given node is pure or not
def is_pure_node(data):
    Price = data[:, -1]
    unique_vals = np.unique(Price)
    if len(unique_vals) == 1:
        return True
    else:
        return False


#fucntion will create a leaf node
def leaf_node(data):
    Price = data[:, -1]
    leaf = np.mean(Price)    
    return leaf


#finds the combinations of various splits in the dataset
def get_splits(data):
  n_rows, n_columns = data.shape
  splits = {}
  for index in range(n_columns - 1):            
      values = data[:,index]
      unique_values = np.unique(values)
      splits[index] = unique_values
  
  return splits



#splitting the data with respect to the given feature and it's threshold
def split_data(data, feature, threshold):
    split_column_values = data[:, feature]
    type_of_feature = FEATURE_TYPES[feature]
     # feature is continous
    if type_of_feature == "continuous":
        left = data[split_column_values <= threshold]
        right = data[split_column_values >  threshold]
    # feature is categorical   
    else:
        left = data[split_column_values == threshold]
        right = data[split_column_values != threshold]
    
    return left,right



#function tells the type of feature :- continous or categorical
def type_of_feature(data):
    feature_types = []
    for feature in data.columns:
        if feature != "Price":
            unique_values = data[feature].unique()
            if (len(unique_values) <= 2):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types


#function to calculate the squared error metric for the nodes 
def calculate_overall_metric(data_left, data_right, metric_function):
    n = len(data_left) + len(data_right)
    overall_metric =  (metric_function(data_left) 
                     +  metric_function(data_right))
    return overall_metric


#function to calculate the squared error
def calculate_se(data):
    actual_values = data[:, -1]
    if len(actual_values) == 0:   
        se = 0
    else:
        prediction = np.mean(actual_values)
        se = np.sum((actual_values - prediction) **2)
    
    return se


def determine_best_split(data, get_splits): 
    flag = True   #checking for first iteration
    for column_index in get_splits:
        for value in get_splits[column_index]:
            data_left, data_right = split_data(data, column_index, value)
            overall_metric = calculate_overall_metric(data_left, data_right, metric_function=calculate_se)
            if flag or overall_metric <= best_overall_metric:
                flag = False
                best_overall_metric = overall_metric
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value



#function to determine the type of the feature
def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 2
    for feature in df.columns:
        if feature != "Price":
            unique_values = df[feature].unique()
            example_value = unique_values[0]
            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

#function to predict the testing examples
def predict_example(example, tree):
    if not isinstance(tree,dict):
      return tree
    rule = list(tree.keys())[0]
    feature_name, comparison_operator, value = rule.split("  ")

    # get rule
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[rule][0]
        else:
            answer = tree[rule][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[rule][0]
        else:
            answer = tree[rule][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)



def filter_df(df, rule):

    feature, comparison_operator, value = rule.split("  ")
    
    # continuous feature
    if comparison_operator == "<=":
        df_yes = df[df[feature] <= float(value)]
        df_no =  df[df[feature] >  float(value)]
        
    # categorical feature
    else:
        df_yes = df[df[feature].astype(str) == value]
        df_no  = df[df[feature].astype(str) != value]
    
    return df_yes, df_no

def make_predictions(df, tree):
    
    if len(df) != 0:
        predictions = df.apply(predict_example, args=(tree,), axis=1)
    else:
        # "df.apply()"" with empty dataframe returns an empty dataframe,
        # but "predictions" should be a series instead
        predictions = pd.Series()
        
    return predictions



def determine_leaf(df_train):
  return df_train.Price.mean()


def determine_errors(df_val, tree):
    predictions = make_predictions(df_val, tree) 
    actual_values = df_val.Price
    return ((predictions - actual_values) **2).sum()


def post_pruning(tree, df_train, df_val):
    
    rule = list(tree.keys())[0]
    yes_answer, no_answer = tree[rule]

    # base case
    if not isinstance(yes_answer, dict) and not isinstance(no_answer, dict):
        return pruning_result(tree, df_train, df_val)
        
    # recursive part
    else:
        df_train_yes, df_train_no = filter_df(df_train, rule)
        df_val_yes, df_val_no = filter_df(df_val, rule)
        
        if isinstance(yes_answer, dict):
            yes_answer = post_pruning(yes_answer, df_train_yes, df_val_yes)
            
        if isinstance(no_answer, dict):
            no_answer = post_pruning(no_answer, df_train_no, df_val_no)
        
        tree = {rule: [yes_answer, no_answer]}
    
        return pruning_result(tree, df_train, df_val)

def pruning_result(tree, df_train, df_val):
    
    leaf = determine_leaf(df_train)
    errors_leaf = determine_errors(df_val, leaf)
    errors_decision_node = determine_errors(df_val, tree)

    if errors_leaf <= errors_decision_node:
        return leaf
    else:
        return tree



#calculating the total squared error
def calculate_squared_e(df, tree):    
    labels = df.Price
    predictions = df.apply(predict_example, args=(tree,), axis=1)
    ss_res = sum((labels - predictions) ** 2)
    return ss_res



#complete decision tree algorithm to build the tree
def decision_tree_algorithm(df, counter=0, min_samples=5, max_depth=2):
    if counter==0:
      global COLUMN_HEADERS, FEATURE_TYPES
      COLUMN_HEADERS = df.columns
      FEATURE_TYPES = determine_type_of_feature(df)
      data = df.values
    else:
      data=df          

      
    if (is_pure_node(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = leaf_node(data)
        return leaf    
  
    else:    
        counter += 1
      
      
        p_splits = get_splits(data)
        split_column, split_value = determine_best_split(data, p_splits)
        data_left, data_right = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_left) == 0 or len(data_right) == 0:
            leaf = leaf_node(data)
            return leaf
        
        # determine rule
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        #creating the rules of the regression treee for continous features
        if type_of_feature == "continuous":
            rule = "{}  <=  {}".format(feature_name, split_value)
            
        # creaitng the rules of the regression tree for categorical variables
        else:
            rule = "{}  =  {}".format(feature_name, split_value)
       
        
        # creating the subtree sub-tree
        sub_tree = {rule: []}
        
        # finding the subtrees
        yes_answer = decision_tree_algorithm(data_left, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_right, counter, min_samples, max_depth)
        
       
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[rule].append(yes_answer)
            sub_tree[rule].append(no_answer)
        
        return sub_tree


f = open("output.txt", "a")
def buildtree(data):
    data = data.sample(frac=1)
    train_df = data.iloc[:14]
    test_df  = data.iloc[14:]
    tree = decision_tree_algorithm(train_df, max_depth=5,min_samples=1)
    r_squared_train = calculate_squared_e(train_df, tree)
    r_squared_test = calculate_squared_e(test_df, tree)
    print("Regression Tree Generated",file=f)
    print("----------------------------------------------------------------------------------------------------------",file=f)
    print(pformat(tree),file=f)
    print(file=f)
    print(file=f)
    print("Training squared error",r_squared_train,file=f)
    print("Testing squared error",r_squared_test,file=f)

    print("------------------------------------------------------------------------------------------------------------",file=f)
    return 0



def find_best_tree(data):
    #finding the best tree taking minimum samples = 1 
    min = 1e20
    best_tree= {}
    best_parameters = {"max_depth": [], "min_samples": [], "r_squared_train": [],"r_squared_test": []}
    for max_depth in range(1, 7):
        #for min_samples in range(1,7):
        data = data.sample(frac=1)
        train_df = data.iloc[:13]
        test_df  = data.iloc[13:]
        min_samples=1
        for i in range(10):
            tree = decision_tree_algorithm(train_df,  max_depth=max_depth, min_samples=min_samples)
            r_squared_train = calculate_squared_e(train_df, tree)
            r_squared_test = calculate_squared_e(test_df, tree)
        
            best_parameters["max_depth"].append(max_depth)
            best_parameters["min_samples"].append(min_samples)
            best_parameters["r_squared_train"].append(r_squared_train)
            best_parameters["r_squared_test"].append(r_squared_test)
            
            if r_squared_test < min:
                min = r_squared_test
                best_tree=tree
    
        
    best_parameters = pd.DataFrame(best_parameters)
    best_parameters = best_parameters.sort_values("r_squared_test", ascending=True).head()
    print(file=f)
    print("----------------------------------------------------------------------------------------------------------",file=f)
    print("Best tree",file=f)
    print(file=f)
    print(pformat(best_tree),file=f)
    print(file=f)
    print("Parameters for the Best tree",file=f)
    print(file=f)
    print(best_parameters.head(1),file=f)
    print("------------------------------------------------------------------------------------------------------------",file=f)
    print("------------------------------------------------------------------------------------------------------------",file=f)
    return best_tree

def apply_pruning(data,best_tree):
    data = data.sample(frac=1)
    df_train = data.iloc[:13]
    df_val = data.iloc[13:15]
    df_test  = data.iloc[15:]

    tree = best_tree
    print(file=f)
    print("Tree Before Prunning",file=f)
    print(file=f)
    print(pformat(tree),file=f)
    print(file=f)
    tree_pruned = post_pruning(tree, df_train, df_val)
    print("Tree After Prunning",file=f)
    print(file=f)
    print(pformat(tree_pruned),file=f)
    mse_tree = determine_errors(df_test, tree)
    mse_tree_pruned = determine_errors(df_test, tree_pruned)
    print(file=f)
    print(f"Squared Error of Tree:        {int(mse_tree):,}",file=f)
    print(f"Squared Error of Pruned Tree: {int(mse_tree_pruned):,}",file=f)
    print(file=f)
    print("------------------------------------------------------------------------------------------------------------",file=f)
    print("------------------------------------------------------------------------------------------------------------",file=f)
    return 0

def find_variations(data):
    M =[]
    A=[]
    T=[]
    data = data.sample(frac=1)
    train_df = data.iloc[:13]
    test_df  = data.iloc[13:]
    for max_depth in range(1, 7):
        min_samples=1
        tree = decision_tree_algorithm(train_df,  max_depth=max_depth, min_samples=min_samples)
        r_squared_train = calculate_squared_e(train_df, tree)
        r_squared_test = calculate_squared_e(test_df, tree)
        T.append(r_squared_train)
        M.append(max_depth)
        A.append(r_squared_test)
   
    ind = T.index(min(T))
    print("------------------------------------------------------------------------------------------------------------",file=f)
    print("------------------------------------------------------------------------------------------------------------",file=f)
    print("The Depth of the Tree for which it overfits is ",M[ind],file=f)
    print(file=f)
   

    plt.plot(M,A)
    plt.savefig('plot.png')
    print("Plot the variation in test accuracy with varying depths is saved as plot.png",file=f)
    print("------------------------------------------------------------------------------------------------------------",file=f)
    print("------------------------------------------------------------------------------------------------------------",file=f)

    
data = load_data()

buildtree(data)


best_tree = find_best_tree(data)

find_variations(data)

apply_pruning(data,best_tree)

f.close()




