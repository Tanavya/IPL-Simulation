import pandas as pd
import numpy as np

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
    
def initialise(data):
    
    batsmen = data["batsman"].unique()
    bowlers = data["bowler"].unique()
    
    batsman_index = {}
    bowler_index = {}

    for i in range(len(batsmen)):
        batsman_index[batsmen[i]] = i

    for i in range(len(bowlers)):
        bowler_index[bowlers[i]] = i
    
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
#         seasons[l].append(ball_data["season"])

        batsman_stats[i]["M"].add(current_id)
        batsman_stats[i]["BF"] += 1
        batsman_stats[i]["Runs"] += runs
        batsman_stats[i]["4s"] += (runs == 4)
        batsman_stats[i]["6s"] += (runs == 6)
        if player_dismissed:
            try:
                batsman_stats[batsman_index[player_dismissed]]["Outs"] += (player_dismissed != None)
            except:
                pass

        bowler_stats[j]["M"].add(current_id)
        bowler_stats[j]["B"] += 1
        bowler_stats[j]["Runs"] += runs
        bowler_stats[j]["Wkts"] += (player_dismissed != None)
    
    X = np.asarray([np.array(X[i]) for i in range(9)], dtype=object)
    id1 = np.asarray([np.array(id1[i]) for i in range(9)], dtype=object)
    id2 = np.asarray([np.array(id2[i]) for i in range(9)], dtype=object)
    
    for i in range(len(batsmen)):        
        batsman_stats[i]["M"] = len(batsman_stats[i]["M"])
        if batsman_stats[i]["BF"] == 0:
            batsman_stats[i]["SR"] = 0
        else:
            batsman_stats[i]["SR"] = batsman_stats[i]["Runs"]/batsman_stats[i]["BF"] * 100
        if batsman_stats[i]["Outs"] == 0:
            batsman_stats[i]["Avg"] = batsman_stats[i]["Runs"]
        else:
            batsman_stats[i]["Avg"] = batsman_stats[i]["Runs"]/batsman_stats[i]["Outs"]
    for j in range(len(bowlers)):
        bowler_stats[j]["M"] = len(bowler_stats[j]["M"])
        if bowler_stats[j]["Wkts"] == 0:
            bowler_stats[j]["SR"] = bowler_stats[j]["B"]
            bowler_stats[j]["Avg"] = bowler_stats[j]["Runs"]
        else:
            bowler_stats[j]["SR"] = bowler_stats[j]["B"]/bowler_stats[j]["Wkts"]
            bowler_stats[j]["Avg"] = bowler_stats[j]["Runs"]/bowler_stats[j]["Wkts"]
        bowler_stats[j]["Econ"] = bowler_stats[j]["Runs"]/bowler_stats[j]["B"] * 6
        
    batsman_stats = pd.DataFrame(batsman_stats).sort_values(by=["Avg"], ascending=False)
    bowler_stats = pd.DataFrame(bowler_stats).sort_values(by=["Avg"])
    return (batsmen, bowlers, batsman_index, bowler_index, batsman_stats, bowler_stats, X, id1, id2, noballs_and_wides)

def load_data(start_year, end_year):      
    
    # very messy here. please make sure dataset has all int/string values with NaN for player_dismissed, and dropped rows if bowler or batsman columns have any nan values
    
    deliveries_data = pd.read_csv("data/del_puneet.csv")
#     deliveries_data = pd.read_csv("data/ipl_cricsheet_deliveries.csv")


    deliveries_data.dropna(subset = ["bowler"], inplace=True)
    deliveries_data.dropna(subset = ["batsman"], inplace=True)
    deliveries_data["bowler"] = deliveries_data["bowler"].astype(int)
    deliveries_data["batsman"] = deliveries_data["batsman"].astype(int)
    deliveries_data = deliveries_data.replace(np.nan, -1, regex=True)
    deliveries_data["player_dismissed"] = deliveries_data["player_dismissed"].astype(int)
    deliveries_data = deliveries_data.replace(-1, np.nan, regex=True)

#     deliveries_data = deliveries_data.replace(np.nan, -1, regex=True)
#     deliveries_data = deliveries_data.sort_values("date", kind='mergesort')
    both_innings_data = deliveries_data[(deliveries_data["season"] >= start_year) & (deliveries_data["season"] <= end_year)]
    first_innings_data = both_innings_data[both_innings_data["inning"] == 1]
    second_innings_data = both_innings_data[both_innings_data["inning"] == 2]
    
    return (first_innings_data, second_innings_data, both_innings_data)