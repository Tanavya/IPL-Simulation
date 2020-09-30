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
from pymc3.math import *

print("Running on PyMC3 v{}".format(pm.__version__))

def get_situation(w, b):
    if 0 <= w and w <= 3:
        if 1 <= b and b <= 36:
            return 1
        elif 90 <= b and b <= 96:
            return 2
        else:
            return 3
    elif 4 <= w and w <= 6:
        if 1 <= b and b <= 36:
            return 4
        elif 90 <= b and b <= 96:
            return 5
        else:
            return 6
    else:
        if 1 <= b and b <= 36:
            return 7
        elif 90 <= b and b <= 96:
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


directory = "data"

data = pd.read_csv("data/deliveries.csv")
matches = pd.read_csv("data/matches.csv")


selected_ids = matches[matches["season"] >= 2015]["id"]
selected_data = data[data["match_id"].isin(selected_ids)]

noballs_and_wides_count = 0
noballs_and_wides = {}
batsmen = []
bowlers = []
situation_data = {}


first_innings = selected_data[selected_data["inning"] == 1]
print(len(first_innings))


batsman_index = {}
bowler_index = {}
batsmen = first_innings["batsman"].unique()
bowlers = first_innings["bowler"].unique()
for i in range(len(batsmen)):
    batsman_index[batsmen[i]] = i
for i in range(len(bowlers)):
    bowler_index[bowlers[i]] = i

X = [[] for i in range(9)]
id1 = [[] for i in range(9)]
id2 = [[] for i in range(9)]

for k in range(0, 8):
    noballs_and_wides[k] = 0
for l in range(0, 10):
    situation_data[l] = []

current_id = 1
w = 0
b = 0
for i in range(len(first_innings)):
    ball_data = first_innings.iloc[i]
    if current_id != ball_data["match_id"]:
        current_id = ball_data["match_id"]
        w = 0
        b = 0
    b += 1
    
    batsman = ball_data["batsman"]
    bowler = ball_data["bowler"]
    
    player_dismissed = False
    if pd.notnull(first_innings["player_dismissed"].iloc[i]):
        player_dismissed = True
        w += 1
        
    if ball_data["wide_runs"] >= 1 or ball_data["noball_runs"] >= 1:
        noballs_and_wides_count += 1
        runs = ball_data["batsman_runs"]
        noballs_and_wides[get_outcome(runs, player_dismissed)] += 1
        continue
        
    runs = ball_data["batsman_runs"]
    l = get_situation(w, b) - 1
    k = get_outcome(runs, player_dismissed)
    if k == 0:
        continue
    X[l].append(k)
    id1[l].append(batsman_index[batsman])
    id2[l].append(bowler_index[bowler])
    situation_data[l].append((batsman_index[batsman], bowler_index[bowler], k))


X = np.asarray([np.array(X[i]) for i in range(9)], dtype=object)
id1 = np.asarray([np.array(id1[i]) for i in range(9)], dtype=object)
id2 = np.asarray([np.array(id2[i]) for i in range(9)], dtype=object)


INF = 5
testval = [[-INF + x * (2 * INF)/5.0 for x in range(6)] for i in range(0, 9)]
l = [i for i in range(9)]


model = pm.Model()
Print = tt.printing.Print("shape:")
with model:
    delta_1 = pm.Uniform("delta_1", lower=0, upper=1)
    delta_2 = pm.Uniform("delta_2", lower=0, upper=1)
    inv_sigma_sqr = pm.Gamma("sigma^-2", alpha=1.0, beta=1.0)
    inv_tau_sqr = pm.Gamma("tau^-2", alpha=1.0, beta=1.0)
    mu_1 = pm.Normal("mu_1", mu=0, sigma=1/pm.math.sqrt(inv_tau_sqr), shape=len(batsmen))
    mu_2 = pm.Normal("mu_2", mu=0, sigma=1/pm.math.sqrt(inv_tau_sqr), shape=len(bowlers))
    delta = pm.math.ge(l, 3) * delta_1 + pm.math.ge(l, 6) * delta_2
    Print(delta.shape)
    Print(mu_1.shape)
    eta = [pm.Deterministic("eta_" + str(i), mu_1[id1[i]] - mu_2[id2[i]]) for i in range(9)]
    cutpoints = pm.Normal("cutpoints", mu=[-5,-3,-1,1,3,5], sigma=1/pm.math.sqrt(inv_sigma_sqr), transform=pm.distributions.transforms.ordered, shape=(9,6), testval=testval)
    Print(cutpoints.shape)
    X_ = [pm.OrderedLogistic("X_" + str(i), cutpoints=cutpoints[i], eta=eta[i], observed=X[i]-1) for i in range(9)]


with model:
    trace = pm.sample(5000)


az.plot_trace(trace);

az.summary(trace, round_to=2)



for b in range(balls_bowled, 120):

  if wickets_1 == 10:
    X1[b] = -1

  else:
    while np.random.uniform(0, 1) < v:       
        random = np.random.uniform(0, 1)
        Y1[b] = (random > pw[0]) + (random > pw[0] + pw[1]) + (random > pw[0] + pw[1] + pw[2]) + \
                (random > pw[0] + pw[1] + pw[2] + pw[3]) + (random > pw[0] + pw[1] + pw[2] + p2[3] + pw[4]) + \
                (random > pw[0] + pw[1] + pw[2] + pw[3] + pw[4] + pw[5])
        # should fix Y1[b] replaces previous Y1[b]
        if Y1[b] == 0:
          wickets1 += 1
        elif Y1[b] == 1:
          runs_1 += 0
        elif Y1[b] == 2:
          runs_1 += 1
        elif Y1[b] == 3:
          runs_1 += 2
        elif Y1[b] == 4:
          runs_1 += 3
        elif Y1[b] == 5:
          runs_1 += 4
        elif Y1[b] == 6:
          runs_1 += 6  
      #batsman to face the delivered ball
      if(b == b0){
        q1[b] = batsman_order1[1]
      }else{
        if(((b-1)/6 - as.integer((b-1)/6)) == 0){       #First ball of any over
          if(X1[b-1] == 1){                             #Wicket on last ball
            out_batsman = q1[which(X1[1:(b-1)] == 1)]
            batsman_batted = batsman_order1[1:(wickets1+2)]
            batsman_field = setdiff(batsman_batted,out_batsman)
            q1[b] = setdiff(batsman_field,q1[b-1])
          }else{
            if(X1[b-1] == 3 | X1[b-1] == 5){            #Batsmen rotated places while running btw wickets on last ball
              q1[b] = q1[b-1]
            }else{
              out_batsman = q1[which(X1[1:(b-1)] == 1)]
              batsman_batted = batsman_order1[1:(wickets1+2)]
              batsman_field = setdiff(batsman_batted,out_batsman)
              q1[b] = setdiff(batsman_field,q1[b-1])
            }
          }
        }else{
          if(X1[b-1] == 1){                             #Wicket on last ball
            q1[b] = batsman_order1[wickets1+2]
          }else{
            if(X1[b-1] == 3 | X1[b-1] == 5){            #Batsmen rotated places while running btw wickets on last ball
              out_batsman = q1[which(X1[1:(b-1)] == 1)]
              batsman_batted = batsman_order1[1:(wickets1+2)]
              batsman_field = setdiff(batsman_batted,out_batsman)
              q1[b] = setdiff(batsman_field,q1[b-1])
            }else{
              q1[b] = q1[b-1]
            }
          }
        }  
      }
      
      #bowler to ball 

      j = bowling_order_1[ceiling((b+1)/6)]
      l = get_situation(wickets_1, b) - 1
      
      rand = np.random.uniform(0, 1)
      X1[b] = 1 + rand > p[q1[b]][]
      X1[b] = 1 + ifelse(random > p[q1[b],j,l,1],1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]),1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]+p[q1[b],j,l,3]),1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]+p[q1[b],j,l,3]+p[q1[b],j,l,4]),1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]+p[q1[b],j,l,3]+p[q1[b],j,l,4]+p[q1[b],j,l,5]),1,0) + ifelse(random > (p[q1[b],j,l,1]+p[q1[b],j,l,2]+p[q1[b],j,l,3]+p[q1[b],j,l,4]+p[q1[b],j,l,5]+p[q1[b],j,l,6]),1,0)
      
      if(X1[b] == 3){
        runs_1 = runs_1+1
      }
      if(X1[b] == 4){
        runs1 = runs1+2
      }
      if(X1[b] == 5){
        runs1 = runs1+3
      }
      if(X1[b] == 6){
        runs1 = runs1+4
      }
      if(X1[b] == 7){
        runs1 = runs1+6
      }
      if(X1[b] == 1){
        wickets1 = wickets1+1
      }
      check = 0
        
      }
    }
  }
}