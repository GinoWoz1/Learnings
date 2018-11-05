# try GA

import pandas as pd
from deap import creator, base, tools, algorithms
import random
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,RepeatedKFold
from sklearn.linear_model import ElasticNet
import warnings

warnings.filterwarnings('ignore')


# Encode the classification labels to numbers
# Get classes and one hot encoded feature vectors


from sklearn.metrics import mean_squared_error,make_scorer

rkfold = RepeatedKFold(n_splits=5,n_repeats=5)

X_train_poly = pd.read_csv('C:\\Users\\jstnj\\Google Drive\\Kaggle\\Learnings\\X_train_poly.csv',index_col = ['Unnamed: 0'])

y_train = pd.read_csv('C:\\Users\\jstnj\\Google Drive\\Kaggle\\Learnings\\y_train.csv',header=None,index_col=0)


# Encode the classification labels to numbers
# Get classes and one hot encoded feature vectors


from sklearn.metrics import mean_squared_error,make_scorer
import numpy as np

rkfold = RepeatedKFold(n_splits=5,n_repeats=5)


def rmse_cv(y_true, y_pred) : 
    assert len(y_true) == len(y_pred)
    y_pred = np.exp(y_pred)
    y_true = np.exp(y_true)
    return np.sqrt(mean_squared_error(y_true,y_pred))

rmse_cv = make_scorer(rmse_cv,greater_is_better=False)

# select all features

allFeatures = X_train_poly

# Feature subset fitness function
def getFitness(individual, X_train_poly, y_train):
    # Parse our feature columns that we don't use
    # Apply one hot encoding to the features
    # Apply logistic regression on the data, and calculate accuracy
    elnet_final = ElasticNet(alpha=0,l1_ratio=0.0,tol=1e-05)
    cross_val = cross_val_score(elnet_final,X_train_poly,y_train,scoring=rmse_cv,cv=rkfold)
    score = np.mean(cross_val)
       
    feature_count =  0
    for i in range(len(individual)):
        binary = individual[i]
        if binary == 1:
            feature_count+=1
    
    # Return calculated accuracy as fitness
    return (score,feature_count)

#========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

# Create Individual
creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X_train_poly.columns) - 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness, X_train_poly=X_train_poly, y_train=y_train)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
toolbox.register("select", tools.selNSGA2)


#========

def getHof():
    # Initialize variables to use eaSimple
    numPop = 730
    numGen = 100
    pop = toolbox.population(n=numPop)
    hof = tools.HallOfFame(numPop * numGen)    
    stats_loss = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    mstats = tools.MultiStatistics(fitness=stats_loss, size=stats_size)
    mstats.register("avgLoss", np.mean)
    mstats.register("stdLoss", np.std)
    mstats.register("minLoss", np.min)
    mstats.register("maxLoss", np.max)

        
    # Launch genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=numGen, stats=mstats, halloffame=hof, verbose=True)
    
    # Return the hall of fame
    return pop,log,hof

def getMetrics(hof):

	# Get list of percentiles in the hall of fame
	percentileList = [i / (len(hof) - 1) for i in range(len(hof))]
	
	# Gather fitness cdata from each percentile
	Scorelist = []
	individualList = []
	for individual in hof:
		cv_scores = getFitness(individual,X_train_poly,y_train)
		Scorelist.append(cv_scores[0])
	individualList.reverse()
	return Scorelist,individualList, percentileList

hof = getHof()

Scorelist,individualList, percentileList =  getMetrics(hof)