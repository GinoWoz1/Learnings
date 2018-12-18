from sklearn.model_selection import cross_val_score,RepeatedKFold
from sklearn.metrics import mean_squared_error,make_scorer
from deap import creator, base, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import numpy as np
import warnings
import random
import numpy
from jfunc import rmse_cv

warnings.filterwarnings('ignore')

rkfold = RepeatedKFold(n_splits=5,n_repeats=5)

url = 'https://github.com/GinoWoz1/Learnings/raw/master/'
    
X_train = pd.read_csv(url + 'X_trainGA.csv',index_col= 'Unnamed: 0')
y_train = pd.read_csv(url +'y_trainGA.csv',header=None,index_col=0)

X_train.rename(columns={'Constant Term':'tax'},inplace=True)

warnings.filterwarnings('ignore')

rkfold = RepeatedKFold(n_splits=3,n_repeats=5)

"""
def rmse_cv(y_true, y_pred) : 
    assert len(y_true) == len(y_pred)
    y_pred = np.exp(y_pred)
    y_true = np.exp(y_true)
    return np.sqrt(mean_squared_error(y_true,y_pred))

rmse_cv = make_scorer(rmse_cv,greater_is_better=False)
"""
# select all features

allFeatures = X_train

# scale data

elnet_final = ElasticNet(alpha=0,l1_ratio=0.0,tol=1e-05)

elnet_pipe = Pipeline([('std',StandardScaler()),
                       ('elnet',elnet_final)])
        


def getFitness(individual, X_train, y_train):
    # Parse our feature columns that we don't use for training
    cols = [index for index in range(len(individual)) if individual[index]==0]
    X_train_reduced = X_train.drop(X_train.columns[cols],axis=1)
    feature_count = sum(individual)    
    cross_val = cross_val_score(elnet_pipe,X_train_reduced,y_train,scoring=make_scorer(rmse_cv,greater_is_better=False),cv=rkfold,n_jobs=-1,error_score=0.0)
    score = np.mean(cross_val)
       
    
    # Return calculated accuracy as fitness
    return (score,feature_count)

#========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

# Create Individual
creator.create('FitnessMulti', base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(X_train.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness, X_train=X_train, y_train=y_train)
toolbox.register("mate", tools.cxUniform,indpb=0.10)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.60)
toolbox.register("select", tools.selNSGA2)

def getHof():
    # Initialize variables to use eaSimple
    numPop = 730
    numGen =  100
    pop = toolbox.population(n=numPop)
    hof = tools.ParetoFront()  
    stats_loss = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    mstats = tools.MultiStatistics(fitness=stats_loss, size=stats_size)
    mstats.register("avgLoss", np.mean)
    mstats.register("stdLoss", np.std)
    mstats.register("minLoss", np.min)
    mstats.register("maxLoss", np.max)
    
    
    # Launch genetic algorithm
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu= 730, lambda_=730,cxpb=0.4, mutpb=0.6, ngen=numGen, stats=mstats, halloffame=hof, verbose=True)
    
    # Return the hall of fame
    return hof

hof = getHof()

# look at keys

hof.keys

hof_individuals = hof.items
hof_individuals = list(reversed(hof_individuals))

for x in enumerate(hof.keys):
    num= x[0]
    columns = [list(allFeatures)[index] for index in range(len(hof_individuals[num])) if hof_individuals[num][index] == 1]  
    cross_val = cross_val_score(elnet_pipe,X_train[columns],y_train,scoring=make_scorer(rmse_cv,greater_is_better=False),cv=rkfold,n_jobs=-1)
    print('run #'+' '+str(num)+'\n\n' + str(x[1]),np.mean(cross_val))
    cross_val = cross_val_score(elnet_pipe,X_train[columns],y_train,scoring=make_scorer(rmse_cv,greater_is_better=False),cv=rkfold,n_jobs=-1)
          
