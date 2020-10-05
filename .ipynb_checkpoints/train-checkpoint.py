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

print("Running on PyMC3 v{}".format(pm.__version__))

def get_situation(w, b):
    if 0 <= w and w <= 3:
        if 1 <= b and b <= 36:
            return 1
        elif 36 <= b and b <= 96:
            return 2
        else:
            return 3
    elif 4 <= w and w <= 6:
        if 1 <= b and b <= 36:
            return 4
        elif 36 <= b and b <= 96:
            return 5
        else:
            return 6
    else:
        if 1 <= b and b <= 36:
            return 7
        elif 36 <= b and b <= 96:
            return 8
        else:
            return 9

def get_outcome(r, wicket):
    if wicket is True:
        return 1
    elif r == 0:
        return 2
    elif r == 1:
        return 3
    elif r == 2:
        return 4
    elif r == 3:
        return 5
    elif r == 4:
        return 6
    elif r == 6:
        return 7
    else:
        return 0

def initialise(data, batsman_index, bowler_index):
    
    noballs_and_wides = {}
    batsman_stats_columns = ["M", "Outs", "Runs", "Avg", "BF", "SR", "4s", "6s"]
    bowler_stats_columns = ["M", "B", "Runs", "Wkts", "Econ", "Avg", "SR"]
    
    batsman_stats = [{} for i in range(len(batsmen))]
    bowler_stats = [{} for j in range(len(bowlers))]

    for i in range(len(batsman_index)):
        for col in batsman_stats_columns:
            batsman_stats[i][col] = 0
        batsman_stats[i]["M"] = set([])
        batsman_stats[i]["Name"] = batsmen[i]

    for j in range(len(bowler_index)):
        for col in bowler_stats_columns:
            bowler_stats[j][col] = 0
        bowler_stats[j]["M"] = set([])
        bowler_stats[j]["Name"] = bowlers[j]

    X = [[] for i in range(9)]
    id1 = [[] for i in range(9)]
    id2 = [[] for i in range(9)]

    for k in range(0, 8):
        noballs_and_wides[k] = 0
    
    current_id = -1
    w = 0
    b = 0
    for i in range(len(data)):
        ball_data = data.iloc[i]
        if current_id != ball_data["match_id"]:
            current_id = ball_data["match_id"]
            w = 0
            b = 0

        b += 1

        batsman = ball_data["batsman"]
        bowler = ball_data["bowler"]

        i = batsman_index[batsman]
        j = bowler_index[bowler]

        player_dismissed = None
        if pd.notnull(ball_data["player_dismissed"]):
            player_dismissed = ball_data["player_dismissed"]
            w += 1

        if ball_data["wide_runs"] >= 1 or ball_data["noball_runs"] >= 1:
            runs = ball_data["batsman_runs"] # check should be total_runs ?
            noballs_and_wides[get_outcome(runs, player_dismissed)] += 1
            continue

        runs = ball_data["batsman_runs"]
        l = get_situation(w, b) - 1
        k = get_outcome(runs, player_dismissed != None)
        
        if k == 0:
            continue
        
        X[l].append(k)
        id1[l].append(i)
        id2[l].append(j)

        batsman_stats[i]["M"].add(current_id)
        batsman_stats[i]["BF"] += 1
        batsman_stats[i]["Runs"] += runs
        batsman_stats[i]["4s"] += (runs == 4)
        batsman_stats[i]["6s"] += (runs == 6)
        if player_dismissed:
            batsman_stats[batsman_index[player_dismissed]]["Outs"] += (player_dismissed != None)

        bowler_stats[j]["M"].add(current_id)
        bowler_stats[j]["B"] += 1
        bowler_stats[j]["Runs"] += runs
        bowler_stats[j]["Wkts"] += (player_dismissed != None)
    
    X = np.asarray([np.array(X[i]) for i in range(9)], dtype=object)
    id1 = np.asarray([np.array(id1[i]) for i in range(9)], dtype=object)
    id2 = np.asarray([np.array(id2[i]) for i in range(9)], dtype=object)
    
    return (batsman_stats, bowler_stats, X, id1, id2, noballs_and_wides)


start_year = 2012
end_year = 2019
n_iter = 5000
train_flag = 1 # 1 if train on First Innings, 2 if train on Second Innings, 3 if train on all data
save_directory = "latest_save"

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
    else:
        save_directory = arg
            
if save_directory[-1] == "/":
    save_directory = save_directory[:-1]

print("Start Year:", start_year)
print("End Year:", end_year)
print("Number of iterations:", n_iter)
print("Train Flag:", train_flag)
print("Save Directory", save_directory)


deliveries_data = pd.read_csv("data/deliveries.csv")

matches = pd.read_csv("data/matches.csv")

selected_ids = matches[(matches["season"] >= start_year) & (matches["season"] <= end_year)]["id"]
selected_data = deliveries_data[deliveries_data["match_id"].isin(selected_ids)]


first_innings_data = selected_data[selected_data["inning"] == 1]
second_innings_data = selected_data[selected_data["inning"] == 2]
# first_innings_data = selected_data # just to compare all innings
print("First innings data size:", len(first_innings_data))
print("Second innings data size:", len(second_innings_data))


# batsmen = selected_data["batsman"].unique()
# bowlers = selected_data["bowler"].unique()

if train_flag == 1:
    batsmen = first_innings_data["batsman"].unique()
    bowlers = first_innings_data["bowler"].unique()
elif train_flag == 2:
    batsmen = second_innings_data["batsman"].unique()
    bowlers = second_innings_data["bowler"].unique()
else:
    batsmen = selected_data["batsman"].unique()
    bowlers = selected_data["bowler"].unique()

batsman_index = {}
bowler_index = {}

for i in range(len(batsmen)):
    batsman_index[batsmen[i]] = i

for i in range(len(bowlers)):
    bowler_index[bowlers[i]] = i

if train_flag == 1:
    batsman_stats, bowler_stats, X, id1, id2, noballs_and_wides = initialise(first_innings_data, batsman_index, bowler_index)
elif train_flag == 2:
    batsman_stats, bowler_stats, X, id1, id2, noballs_and_wides = initialise(second_innings_data, batsman_index, bowler_index)
else:
    batsman_stats, bowler_stats, X, id1, id2, noballs_and_wides = initialise(selected_data, batsman_index, bowler_index)

noballs_and_wides_count = sum(noballs_and_wides.values())
total_balls = sum([len(X[i]) for i in range(9)]) + noballs_and_wides_count

print("Number of noballs and wides:", noballs_and_wides_count)
print("Number of balls bowled:", total_balls)

for i in range(9):
    print("Balls in situation %d: %d" % (i+1, len(X[i])))

model = create_model(batsmen, bowlers, id1, id2, X)

print("Model initialised")

with model:
    trace = pm.sample(n_iter)

pm.save_trace(trace, directory=save_directory + "/trace", overwrite=True)

cutpoints = np.mean(trace.get_values("cutpoints", burn=n_iter//2, combine=True), axis=0)
mu_1 = np.mean(trace.get_values("mu_1", burn=n_iter//2, combine=True), axis=0)
mu_2 = np.mean(trace.get_values("mu_2", burn=n_iter//2, combine=True), axis=0)

delta_1 = np.mean(trace.get_values("delta_1", burn=n_iter//2, combine=True), axis=0)
delta_2 = np.mean(trace.get_values("delta_1", burn=n_iter//2, combine=True), axis=0)
delta = np.greater_equal([i for i in range(9)], 3) * delta_1 + np.greater_equal([i for i in range(9)], 6) * delta_2

np.savetxt(loc + "/cutpoints.txt", cutpoints)
np.savetxt(loc + "/mu_1.txt", mu_1)
np.savetxt(loc + "/mu_2.txt", mu_2)
np.savetxt(loc + "/delta.txt", delta)
