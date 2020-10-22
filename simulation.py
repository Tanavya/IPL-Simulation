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
from helper import load_data, initialise, get_situation, get_outcome
from model import create_model

print("Running on PyMC3 v{}".format(pm.__version__))

start_year = 2015
end_year = 2019
n_simulation = 10
verbose = True # flag used in simulation
train_flag = 3 # 1 if trained on first innings data, 2 if trained on second innings data, 3 if trained on all data
save_directory = "2015-2019-5k-t3"


argumentList = sys.argv 

for arg in argumentList[1:]:
    if arg == "--verbose":
        verbose = True
    elif arg[0] == "-":
        if arg[1] == "n":
            n_simulation = int(arg[2:])
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
print("Number of simulations:", n_simulation)
print("Save Directory:", save_directory)
print("Train flag:", train_flag)
print("Verbose:", verbose)

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
print("Loaded model.")

cutpoints = np.loadtxt(save_directory + "/cutpoints.txt")
mu_1 = np.loadtxt(save_directory + "/mu_1.txt")
mu_2 = np.loadtxt(save_directory + "/mu_2.txt")
delta = np.loadtxt(save_directory + "/delta.txt")

mu_1_sorted = sorted([(mu_1[i], batsmen[i]) for i in range(len(mu_1))])
mu_2_sorted = sorted([(mu_2[i], bowlers[i]) for i in range(len(mu_2))])

pw = np.zeros(shape=7)          
for i in range(7):
    pw[i] = float(noballs_and_wides[i])/noballs_and_wides_count
v = float(noballs_and_wides_count)/total_balls

print("Loaded pre-trained parameters.")

DLS = pd.read_csv("data/dls-simplified.csv").rename(columns={"Unnamed: 0": "Balls Consumed"}).set_index("Balls Consumed")
DLS.columns = DLS.columns.astype(int)[::-1]
DLS.index = DLS.index.astype(int)[::-1]

#Resources lost due to wicket
y = np.zeros(shape=(121, 10))
for i in range(10):
    y[:,i] = DLS.loc[:,i] - DLS.loc[:,i+1]

x = np.zeros(shape = (120, 11))
for i in range(120):
    x[i,:] = DLS.loc[i,:] - DLS.loc[i+1,:]

team1 = "Delhi Capitals" # batting first
team2 = "Rajashtan Royals" # batting second

batting_order_names_1 = ['PP Shaw', 'S Dhawan', 'AM Rahane', 'SS Iyer', 'MP Stoinis', 'AT Carey', 'AR Patel', 'R Ashwin', 'K Rabada', 'A Nortje', 'T Deshpande']
batting_order_names_2 = ['BA Stokes', 'JC Buttler', 'SPD Smith', 'SV Samson', 'RV Uthappa', 'R Parag', 'R Tewatia', 'JC Archer', 'S Gopal', 'JD Unadkat', 'Kartik Tyagi']

batting_order_1 = []
batting_order_2 = []
debutant_batsmen = []

for i in range(11):
    for b in [batting_order_names_1[i], batting_order_names_2[i]]:
        if b not in batsman_index:
            batsman_index[b] = len(batsmen) + len(debutant_batsmen)
            debutant_batsmen.append(b)
            
for i in range(11):
    batting_order_1.append(batsman_index[batting_order_names_1[i]])
    batting_order_2.append(batsman_index[batting_order_names_2[i]])
    
bowling_order_names_1 = ['JC Archer', 'JD Unadkat', 'JC Archer', 'Kartik Tyagi', 'BA Stokes', 'Kartik Tyagi', 'S Gopal', 'R Tewatia', 'S Gopal', 'BA Stokes', 'R Tewatia', 'S Gopal', 'R Tewatia', 'S Gopal', 'JD Unadkat', 'Kartik Tyagi', 'JC Archer', 'Kartik Tyagi', 'JC Archer', 'JD Unadkat']
bowling_order_names_2 = ['K Rabada', 'T Deshpande', 'A Nortje', 'R Ashwin', 'A Nortje', 'K Rabada', 'AR Patel', 'R Ashwin', 'AR Patel', 'R Ashwin', 'T Deshpande', 'AR Patel', 'A Nortje', 'AR Patel', 'T Deshpande', 'R Ashwin', 'K Rabada', 'A Nortje', 'K Rabada', 'T Deshpande']

bowling_order_1 = []
bowling_order_2 = []
debutant_bowlers = []

for bowler in range(20):
    for b in [bowling_order_names_1[i], bowling_order_names_2[i]]:
        if b not in bowler_index:
            bowler_index[b] = len(bowlers) + len(debutant_bowlers)
            debutant_bowlers.append(b)

for i in range(20):
    bowling_order_1.append(bowler_index[bowling_order_names_1[i]])
    bowling_order_2.append(bowler_index[bowling_order_names_2[i]])

print("Debutant Batsmen:", debutant_batsmen)
print("Debutant Bowlers:", debutant_bowlers)

batsmen = np.append(batsmen, debutant_batsmen)
bowlers = np.append(bowlers, debutant_bowlers)

p = np.zeros(shape = (len(batsmen),len(bowlers),9,7))

for i in range(len(debutant_batsmen)):
    mu_1 = np.append(mu_1, 0)
for i in range(len(debutant_bowlers)):
    mu_2 = np.append(mu_2, 0)

for i in range(len(batsmen)):
    for j in range(len(bowlers)):
        for l in range(9):
            for k in range(7):
                if k == 0:
                    p[i,j,l,k] = 1/(1 + np.exp(-(cutpoints[l,k] - mu_1[i] + mu_2[j] - delta[l])))
                elif k == 6:
                    p[i,j,l,k] = 1 - 1/(1 + np.exp(-(cutpoints[l,k-1] - mu_1[i] + mu_2[j] - delta[l])))
                else:
                    p[i,j,l,k] = 1/(1 + np.exp(-(cutpoints[l,k] - mu_1[i] + mu_2[j] - delta[l]))) - 1/(1 + np.exp(-(cutpoints[l,k-1] - mu_1[i] + mu_2[j] - delta[l])))
                    

def first_innings_simulation(bowling_team, batting_team, bowling_order, batting_order, first_innings = [], wickets = 0, runs = 0, balls_bowled = 0):
    
    columns=["inning", "batting_team", "bowling_team", "over", "ball", "batsman", "non_striker", "bowler", "wide_runs", "batsman_runs", "total_runs", "player_dismissed"]
    
    X1 = [-1 for i in range(120)]
    Y1 = [0 for i in range(120)]
    q1 = [0 for i in range(121)]
    
    q1[balls_bowled] = (batting_order[wickets], batting_order[wickets + 1])
        
    for b in range(balls_bowled, 120):

        if wickets == 10:
            X1[b] = -1

        else:
            balls_bowled = b
            
            wide_runs = 0 
            batsman_runs = 0
            total_runs = 0
            player_dismissed = None
            
            j = bowling_order[int(b/6)]

            bowler = bowlers[j]
            inning = 1
            over = int(b/6)
            ball = (b%6) + 1

            while np.random.uniform(0, 1) < v:       
                random = np.random.uniform(0, 1)
                Y1[b] = (random > pw[0]) + (random > pw[0] + pw[1]) + (random > pw[0] + pw[1] + pw[2]) + \
                        (random > pw[0] + pw[1] + pw[2] + pw[3]) + (random > pw[0] + pw[1] + pw[2] + pw[3] + pw[4]) + \
                        (random > pw[0] + pw[1] + pw[2] + pw[3] + pw[4] + pw[5])
                # should fix Y1[b] replaces previous Y1[b]

                wide_runs = 1            
                
                if Y1[b] == 0:
                    wickets += 1 # fix wicket fallen on wide/noball

                elif Y1[b] == 2:
                    wide_runs = 2
                    
                elif Y1[b] == 3:
                    wide_runs = 3
                    
                elif Y1[b] == 4:
                    wide_runs = 4
                    
                elif Y1[b] == 5:
                    wide_runs = 5

                elif Y1[b] == 6:
                    wide_runs = 7

                if b:
                    batsman = batsmen[q1[b-1][0]]
                    non_striker = batsmen[q1[b-1][1]]
                else:
                    batsman = batsmen[q1[b][0]]
                    non_striker = batsmen[q1[b][1]]
                
                total_runs = wide_runs + batsman_runs
                runs += total_runs

                first_innings.append([inning, batting_team, bowling_team, over, ball, batsman, non_striker, bowler, wide_runs, batsman_runs, total_runs, player_dismissed])

            #bowler to ball 

            l = get_situation(wickets, b+1) - 1
            rand = np.random.uniform(0, 1)
            player_dismissed = None

            # following can be simplified
            q = q1[b][0]

            X1[b] = 0 + (rand > p[q,j,l,0]) + (rand > (p[q,j,l,0]+p[q,j,l,1])) + (rand > (p[q,j,l,0]+p[q,j,l,1]+p[q,j,l,2])) + (rand > (p[q,j,l,0]+p[q,j,l,1]+p[q,j,l,2]+p[q,j,l,3])) + (rand > (p[q,j,l,0]+p[q,j,l,1]+p[q,j,l,2]+p[q,j,l,3]+p[q,j,l,4])) + (rand > (p[q,j,l,0]+p[q,j,l,1]+p[q,j,l,2]+p[q,j,l,3]+p[q,j,l,4]+p[q,j,l,5]))
            
#             print("X1[%d] = %d" % (b, X1[b]))
        
            if X1[b] == 2:
                batsman_runs = 1
                
            elif X1[b] == 3:
                batsman_runs = 2

            elif X1[b] == 4:
                batsman_runs = 3

            elif X1[b] == 5 :
                batsman_runs = 4

            elif X1[b] == 6:
                batsman_runs = 6

            elif X1[b] == 0:
                wickets += 1
                player_dismissed = batsmen[q1[b][0]]
            
            total_runs = batsman_runs
            runs += total_runs
            
            batsman = batsmen[q1[b][0]]
            non_striker = batsmen[q1[b][1]]
            first_innings.append([inning, batting_team, bowling_team, over, ball, batsman, non_striker, bowler, wide_runs, batsman_runs, total_runs, player_dismissed])

            if wickets == 10:
                continue
            elif X1[b] == 0:
#                 print(b, batting_order, q1)
                q1[b+1] = (batting_order[wickets + 1], q1[b][1])
            else:
                q1[b+1] = q1[b]

            if ((b+1) % 6) == 0 and not (X1[b] == 2 or X1[b] == 4):
                q1[b+1] = (q1[b+1][1], q1[b+1][0])

    first_innings_df = pd.DataFrame(first_innings, columns = columns)
    
    return (first_innings_df, runs, wickets, balls_bowled+1)

# SECOND INNINGS SIMULATION
def second_innings_simulation(bowling_team, batting_team, bowling_order, batting_order, target, second_innings = [], wickets = 0, runs = 0, balls_bowled = 0):
    
    columns=["inning", "batting_team", "bowling_team", "over", "ball", "batsman", "non_striker", "bowler", "wide_runs", "batsman_runs", "total_runs", "player_dismissed"]
    
    X2 = [-1 for i in range(120)]
    Y2 = [0 for i in range(120)]
    q2 = [0 for i in range(121)]
    
    q2[balls_bowled] = (batting_order[wickets], batting_order[wickets + 1])
        
    for b in range(120):

        balls_bowled = b

        wide_runs = 0 
        batsman_runs = 0
        total_runs = 0
        player_dismissed = None

        j = bowling_order[int(b/6)]

        bowler = bowlers[j]
        inning = 1
        over = int(b/6)
        ball = (b%6) + 1

        while np.random.uniform(0, 1) < v:       
            random = np.random.uniform(0, 1)
            Y2[b] = (random > pw[0]) + (random > pw[0] + pw[1]) + (random > pw[0] + pw[1] + pw[2]) + \
                    (random > pw[0] + pw[1] + pw[2] + pw[3]) + (random > pw[0] + pw[1] + pw[2] + pw[3] + pw[4]) + \
                    (random > pw[0] + pw[1] + pw[2] + pw[3] + pw[4] + pw[5])
            # should fix Y2[b] replaces previous Y2[b]

            wide_runs = 1            

            if Y2[b] == 0:
                wickets += 1 # fix wicket fallen on wide/noball

            elif Y2[b] == 2:
                wide_runs = 2

            elif Y2[b] == 3:
                wide_runs = 3

            elif Y2[b] == 4:
                wide_runs = 4

            elif Y2[b] == 5:
                wide_runs = 5

            elif Y2[b] == 6:
                wide_runs = 7

            if b:
                batsman = batsmen[q2[b-1][0]]
                non_striker = batsmen[q2[b-1][1]]
            else:
                batsman = batsmen[q2[b][0]]
                non_striker = batsmen[q2[b][1]]

            total_runs = wide_runs + batsman_runs
            runs += total_runs
            second_innings.append([inning, batting_team, bowling_team, over, ball, batsman, non_striker, bowler, wide_runs, batsman_runs, total_runs, player_dismissed])

        if runs > target or wickets == 10:
            break
        #estimation of probability of outcome in 2nd inning

        p2 = np.zeros(7)

        l = get_situation(wickets, b+1) - 1
        q = q2[b][0]

        E1 = p[q,j,l,2] + 2 * p[q,j,l,3] + 3 * p[q,j,l,4] + 4 * p[q,j,l,5] + 6 * p[q,j,l,6] # expected number of runs to be scored 
        E2 = x[b, wickets] + y[b, wickets] * p[q,j,l,0]                                   # expected proportion of resources consumed

        d = E2/(E2 + y[b, wickets] * (1 - p[q,j,l,0] - p[q,j,l,1]))

        c = min(DLS.loc[b, wickets] * E1 / ((target - runs + 1) * E2), 1)
        
        if d > 1 or d < 0:
            print(c,d)
        
        p2[0] = p[q,j,l,0] + d * p[q,j,l,1] * (1 - c)
        p2[1] = c * p[q,j,l,1]

        for k in range(2,7):
            p2[k] = ((1 - p[q,j,l,0] - (c + d * (1 - c)) * p[q,j,l,1]) / (1 - p[q,j,l,0] - p[q,j,l,1])) * p[q,j,l,k]

        random = np.random.uniform(0, 1)
        X2[b] = 0 + (random > p2[0]) + (random > (p2[0]+p2[1])) + (random > (p2[0]+p2[1]+p2[2])) + (random > (p2[0]+p2[1]+p2[2]+p2[3])) + (random > (p2[0]+p2[1]+p2[2]+p2[3]+p2[4])) + (random > (p2[0]+p2[1]+p2[2]+p2[3]+p2[4]+p2[5]))

#             print("X2[%d] = %d" % (b, X2[b]))

        if X2[b] == 2:
            batsman_runs = 1

        elif X2[b] == 3:
            batsman_runs = 2

        elif X2[b] == 4:
            batsman_runs = 3

        elif X2[b] == 5:
            batsman_runs = 4

        elif X2[b] == 6:
            batsman_runs = 6

        elif X2[b] == 0:
            wickets += 1
            player_dismissed = batsmen[q2[b][0]]

        total_runs = batsman_runs
        runs += total_runs

        batsman = batsmen[q2[b][0]]
        non_striker = batsmen[q2[b][1]]
        second_innings.append([inning, batting_team, bowling_team, over, ball, batsman, non_striker, bowler, wide_runs, batsman_runs, total_runs, player_dismissed])

        if runs > target or wickets == 10:
            break

        if X2[b] == 0:
            q2[b+1] = (batting_order[wickets + 1], q2[b][1])
        else:
            q2[b+1] = q2[b]

        if ((b+1) % 6) == 0 and not (X2[b] == 2 or X2[b] == 4):
            q2[b+1] = (q2[b+1][1], q2[b+1][0])
    
    
    second_innings_df = pd.DataFrame(second_innings, columns = columns)
    
    return (second_innings_df, runs, wickets, balls_bowled+1)

print(team1, "vs", team2)
print("Team batting first:", team1)
print()
print("Beginning simulation...")
print()

average_fi_runs = 0
average_fi_wickets = 0
average_si_runs = 0
average_si_wickets = 0
winner1 = 0
winner2 = 0
tie = 0

n_simulation = 1
verbose = True

for i in range(n_simulation):
    first_innings_df, runs1, wickets1, balls1 = first_innings_simulation(team2, team1, bowling_order_1, batting_order_1)
    second_innings_df, runs2, wickets2, balls2 = second_innings_simulation(team1, team2, bowling_order_2, batting_order_2, runs1)
    
    if verbose:
        print("Simulation number: ", i+1)
        print("First Innings Score: %d-%d (%d.%d)  " % (runs1, wickets1, balls1//6, (balls1%6)))
        print("Second Innings Score: %d-%d (%d.%d) " % (runs2, wickets2, balls2//6, (balls2%6)))
    if runs1 > runs2:
        winner1 += 1
        if verbose:
            print("Winner:", team1)
    elif runs2 > runs1:
        winner2 += 1
        if verbose:
            print("Winner:", team2)
    else:
        tie += 1
        if verbose:
            print("Its a tie!")

    average_fi_runs += runs1
    average_fi_wickets += wickets1

    average_si_runs += runs2
    average_si_wickets += wickets2
    
    for batsman in batting_order_1:
        
average_fi_runs /= n_simulation
average_fi_wickets /= n_simulation
average_si_runs /= n_simulation
average_si_wickets /= n_simulation

print()
print()
print("Average First Innings Score %d-%d: " % (int(average_fi_runs), int(average_fi_wickets)))
print(team1 + " wins:", winner1)
print(team2 + " wins:", winner2)
print("Ties:", tie)
