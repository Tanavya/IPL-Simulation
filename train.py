#!/usr/bin/env python

import yaml
import os
import datetime
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pymc3 as pm
import pymc3.distributions.transforms as tr
import shutil
import theano
import theano.tensor as tt
import random
import math
import pandas as pd
import sys
from model import create_model
from helper import initialise, load_data, get_situation, get_outcome

print("Running on PyMC3 v{}".format(pm.__version__))

start_year = 2015
end_year = 2019
n_iter = 5000
train_flag = 1 # 1 if train on First Innings, 2 if train on Second Innings, 3 if train on all data
save_directory = "latest_save"
target_accept = 0.9

argumentList = sys.argv 

for arg in argumentList[1:]:
    if arg[0] == "-":
        if arg[1] == "n":
            n_iter = int(arg[2:])
        elif arg[1] == "s":
            start_year = int(arg[2:])
        elif arg[1] == "e":
            end_year = int(arg[2:])
        elif arg[1] == "t":
            train_flag = int(arg[2:])
        elif arg[1] == "a":
            target_accept = float(arg[2:])/100
    else:
        save_directory = arg
            
if save_directory[-1] == "/":
    save_directory = save_directory[:-1]

print("Start Year:", start_year)
print("End Year:", end_year)
print("Number of iterations:", n_iter)
print("Train Flag:", train_flag)
print("Save Directory:", save_directory)
print("Target Accept:", target_accept)

deliveries_data, matches, first_innings_data, second_innings_data, both_innings_data = load_data(start_year, end_year)

first_innings_data = both_innings_data[both_innings_data["inning"] == 1]
second_innings_data = both_innings_data[both_innings_data["inning"] == 2]

print("First innings data size:", len(first_innings_data))
print("Second innings data size:", len(second_innings_data))

if train_flag == 1:
    batsmen, bowlers, batsman_index, bowler_index, batsman_stats, bowler_stats, X, id1, id2, noballs_and_wides = initialise(first_innings_data)
elif train_flag == 2:
    batsmen, bowlers, batsman_index, bowler_index, batsman_stats, bowler_stats, X, id1, id2, noballs_and_wides = initialise(second_innings_data)
else:
    batsmen, bowlers, batsman_index, bowler_index, batsman_stats, bowler_stats, X, id1, id2, noballs_and_wides = initialise(both_innings_data)
    
noballs_and_wides_count = sum(list(noballs_and_wides.values()))
total_balls = sum([len(X[i]) for i in range(9)]) + noballs_and_wides_count

print("Number of noballs and wides:", noballs_and_wides_count)
print("Number of balls bowled:", total_balls)

for i in range(9):
    print("Balls in situation %d: %d" % (i+1, len(X[i])))

model = create_model(batsmen, bowlers, id1, id2, X)

print("Model initialised.")

with model:
    trace = pm.sample(n_iter, target_accept=target_accept)

# two ways to save_trace. necessary only for diagnostics, can be removed:

with model:
    pm.save_trace(trace, directory=save_directory + "/trace", overwrite=True)

with open(save_directory + "/trace.pkl", 'wb') as buff:
    pickle.dump(trace, buff)


cutpoints = np.mean(trace.get_values("cutpoints", burn=n_iter//2, combine=True), axis=0)
mu_1 = np.mean(trace.get_values("mu_1", burn=n_iter//2, combine=True), axis=0)
mu_2 = np.mean(trace.get_values("mu_2", burn=n_iter//2, combine=True), axis=0)

delta_1 = np.mean(trace.get_values("delta_1", burn=n_iter//2, combine=True), axis=0)
delta_2 = np.mean(trace.get_values("delta_1", burn=n_iter//2, combine=True), axis=0)
delta = np.greater_equal([i for i in range(9)], 3) * delta_1 + np.greater_equal([i for i in range(9)], 6) * delta_2

# save the parameters
np.savetxt(save_directory + "/cutpoints.txt", cutpoints)
np.savetxt(save_directory + "/mu_1.txt", mu_1)
np.savetxt(save_directory + "/mu_2.txt", mu_2)
np.savetxt(save_directory + "/delta.txt", delta)

