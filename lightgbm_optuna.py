# Dependencies
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import optuna.integration.lightgbm as lgb
from optuna.integration import LightGBMPruningCallback
from lightgbm import early_stopping, log_evaluation


# Function for Shaping Data for Modeling
def model_data_prep(X_train, y_train):
    
    X      = X_train
    y      = y_train
    data   = pd.concat([y,X], axis=1)
    target = data['TARGET']
    target = target.to_numpy()
    data.drop('TARGET', axis=1, inplace=True)
    data   = data.to_numpy()
    
    return data, target

"""
Optuna optimizes a binary classifier configuration using LightGBM tuner.
"""

def objective(trial):
    
    # Prep Data for Modeling
    data, target = model_data_prep(X_train, y_train)
    
    # Create LightGBM Dataset Object
    dtrain = lgb.Dataset(data, label=target)
    
    # LightGBM Hyperparameters That Apply to All Boosters
    param = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']), # With Domino I know we have access to GPUs so added this
        "objective"         : "binary",
        "metric"            : "binary_logloss",
        "verbosity"         : -1,
        "boosting"          : trial.suggest_categorical("boosting", ["gbdt", "rf", "dart"]), # Defines booster new if statement for "goss"
        "n_estimators"      : trial.suggest_int("n_estimators", 1, 10000),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves"        : trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth"         : trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf"  : trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1"         : trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2"         : trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split" : trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction"  : trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
        "min_child_samples" : trial.suggest_int("min_child_samples", 5, 100),
    }

    # Select the Metric to Prune With
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")
    
    # Cross-Validation
    history = lgb.cv(
        params                = param, 
        train_set             = dtrain,
        num_boost_round       = 500,
        nfold                 = 3,
        stratified            = True,
        shuffle               = True,
        metrics               = {'binary_logloss'},
        seed                  = 111,
        early_stopping_rounds = 10,
        callbacks             = [pruning_callback]
    )
      
    # Extract the best score
    eval_metric = min(history["binary_logloss-mean"])
    
    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe()
    trial.set_user_attr('n_estimators', len(history))
    
    return eval_metric
    
if __name__ == "__main__":
    
    # Pruners
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10) # Median Pruner
    #puner  = optuna.pruners.PercentilePruner(25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10) # Percentile Pruner
    #pruner = optuna.pruners.SuccessiveHalvingPruner() 
    #pruner = optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3)
    
    # Samplers
    sampler = optuna.samplers.RandomSampler() # Random Sampler
    #sampler = optuna.samplers.TPESampler() # Tree-structured Parzen Estimator Sampler
    #sampler = optuna.samplers.CmaEsSampler() # CMA Sampler
    #sampler = optuna.samplers.NSGAIISampler() # Nondominated Sorting Genetic Algorithm II Sampler
    #sampler = optuna.samplers.MOTPESampler() # Multi-Objective Tree-structured Parzen Estimator Sampler
    #sampler = optuna.samplers.IntersectionSearchSpace(include_pruned=False) # Intersection Search Space Sampler
    #sampler = optuna.samplers.intersection_search_space() # Intersection Search Space Sampler II
    
    # Studies
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='minimize')
    
    study.optimize(objective, n_trials=500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))