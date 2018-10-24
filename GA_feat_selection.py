# try GA

import pandas as pd

from deap import creator, base, tools, algorithms
import random
import numpy
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,RepeatedKFold



# Encode the classification labels to numbers
# Get classes and one hot encoded feature vectors


from sklearn.metrics import mean_squared_error,make_scorer
import numpy as np

rkfold = RepeatedKFold(n_splits=5,n_repeats=5)


def rmse_cv(y_true, y_pred) : 
    assert len(y_true) == len(y_pred)
    if not (y_true >= 0).all() and not (y_pred >= 0).all():
        raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")
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

	# Return calculated accuracy as fitness
	return (score,)

#========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X_train_poly.columns) - 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness, X_train_poly=X_train_poly, y_train=y_train)
toolbox.register("mate", tools.cxUniform,indpb=0.10)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.10)
toolbox.register("select", tools.selTournament,tournsize=128)

#========

def getHof():

	# Initialize variables to use eaSimple
	numPop = 730
	numGen = 300
	pop = toolbox.population(n=numPop)
	hof = tools.HallOfFame(numPop * numGen)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	# Launch genetic algorithm
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

	# Return the hall of fame
	return hof

def getMetrics(hof):

	# Get list of percentiles in the hall of fame
	percentileList = [i / (len(hof) - 1) for i in range(len(hof))]
	
	# Gather fitness data from each percentile
	Scorelist = []
	individualList = []
	for individual in hof:
		cv_scores = getFitness(individual,X_train_poly,y_train)
		Scorelist.append(cv_scores[0])
	individualList.reverse()
	return Scorelist,individualList, percentileList

hof = getHof()

Scorelist,individualList, percentileList =  getMetrics(hof)