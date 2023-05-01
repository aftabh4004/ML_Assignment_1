import pandas as pd
from sklearn.model_selection import train_test_split
import math


# ## Normalizing training and testing feature 

def minmax_train(x_train):
    minmax = {}
    for feature in x_train:
        if feature != 'Class_att':
            min = x_train[feature].min(axis=0)
            max = x_train[feature].max(axis=0)
            dif = max - min
            minmax[feature] = [min, max]
            for i in x_train.index:
                x_train.loc[i, feature] = (x_train.loc[i, feature] - min)/dif

    return minmax

def minmax_test(x_test, minmax):
    for feature in x_test:
        if feature != 'Class_att':
            min = minmax[feature][0]
            max = minmax[feature][1]
            dif = max - min
            minmax[feature] = [min, max]
            for i in x_test.index:
                x_test.loc[i, feature] = (x_test.loc[i, feature] - min)/dif


# ## Training-testing helper functions


def training(train_set):
    
    train_set_0 = train_set[train_set['Class_att'] == 0]
    train_set_1 = train_set[train_set['Class_att'] == 1]
    
    y_0 = train_set_0.Class_att
    x_0 = train_set_0.drop('Class_att', axis = 1)
    
    y_1 = train_set_1.Class_att
    x_1 = train_set_1.drop('Class_att', axis = 1)
    
    prior_prob = [0]*2
    prior_prob[0] = len(train_set_0)/(len(train_set_0) + len(train_set_1))
    prior_prob[1] = len(train_set_1)/(len(train_set_0) + len(train_set_1))
    
    gpdf_0 = {}
    gpdf_1 = {}
    
    for feature in x_0:
        mean = x_0[feature].mean()
        std = x_0[feature].std()
        gpdf_0[feature] = [mean, std]
    
    for feature in x_1:
        mean = x_1[feature].mean()
        std = x_1[feature].std()
        gpdf_1[feature] = [mean, std]
        
    return prior_prob, gpdf_0, gpdf_1


# adding alpha to the probabilty for laplace correction
def normpdf(x, mean, sd, alpha):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return (num/denom) + alpha

def predict(instance, prior_prob, gpdf_0, gpdf_1, alpha):
    post_0 = 1;
    
    for feature in instance:
        post_0 += math.log(normpdf(instance[feature], gpdf_0[feature][0], gpdf_0[feature][1], alpha))
    post_0 += math.log(prior_prob[0])
    
    
    post_1 = 1;
    
    for feature in instance:
        post_1 += math.log(normpdf(instance[feature], gpdf_1[feature][0], gpdf_1[feature][1], alpha))
    post_1 += math.log(prior_prob[1])
    
    if post_0 > post_1:
        return 0
    return 1

def testing(test_set, prior_prob, gpdf_0, gpdf_1, alpha):
    y = test_set.Class_att
    x = test_set.drop('Class_att', axis = 1)
    
    acc = 0;
    
    for i in range(len(x)):
        instance = pd.DataFrame(columns=x.columns)
        instance.loc[0] = list(x.iloc[i])
        res = predict(instance, prior_prob, gpdf_0, gpdf_1, alpha)
        if res == y.iloc[i]:
            acc += 1
            
    return acc/len(test_set)
    


# ## 5-fold cross validation



def five_cross_fold(df):
    df = df.sample(frac=1)
    df['Class_att'] = df['Class_att'].replace("Abnormal", 0)
    df['Class_att'] = df['Class_att'].replace("Normal", 1)
    set_len = len(df)//5


    accuracy = []
    for i in range(5):
        test_set = df.iloc[i*set_len : (i+1) *  set_len, :]
        train_set = pd.concat([df.iloc[:i*set_len,:], df.iloc[(i+1) *  set_len:, :]] )


        #normalizing
        minmax = minmax_train(train_set)
        minmax_test(test_set, minmax)

        prior_prob, gpdf_0, gpdf_1 = training(train_set)
        acc = testing(test_set, prior_prob, gpdf_0, gpdf_1, 0.00001)
        accuracy.append(acc)
    print("Accuracy for all five fold")
    print(accuracy)
    print("Average accuracy = ", sum(accuracy)/len(accuracy))


# ## With Laplace correction


def laplace(x_train,x_test,y_train,y_test, alpha):
    train_set = pd.concat([x_train, y_train], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    prior_prob, gpdf_0, gpdf_1 = training(train_set)
    acc = testing(test_set, prior_prob, gpdf_0, gpdf_1, alpha)
    
    print("Accuracy for 70-30 split with laplace correction")
    print(acc)



def main():
    # Encoding of class attribute
    dataset = pd.read_csv("./Train_D_Bayesian.csv")
    dataset['Class_att'] = dataset['Class_att'].replace("Abnormal", 0)
    dataset['Class_att'] = dataset['Class_att'].replace("Normal", 1)
    
    #Spliting the dataset into 70-30 split
    y = dataset.Class_att
    x = dataset.drop('Class_att', axis = 1)
    
    #Training Testing Split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
    
    
    #Checking for outliers
    
    for feature in x_train:
        mean = x_train[feature].mean()
        std = x_train[feature].std()
        thershold = 2 * mean + 5 * std
        count = 0
        for fv in x_train[feature]:
            if fv > thershold:
                count += 1
        if count > 0.5 * len(x_train[feature]):
            x_train = x_train.drop(feature, axis = 1)
            x_test = x_test.drop(feature, axis = 1)
    print("Final set of features")
    print(x.columns.tolist())
    
    
    # Normalize the training set and testing set
    minmax = minmax_train(x_train)
    minmax_test(x_test, minmax)
    
    
    # calling 5 fold cross validatation on the original dataset
    five_cross_fold(dataset)

    
    # Laplace correction with alpha = 0.00001
    # on the 70-30 splited and nomalized training and testing dataset
    laplace(x_train,x_test,y_train,y_test, 0.00001)
    
    


if __name__ == "__main__":
    main()

