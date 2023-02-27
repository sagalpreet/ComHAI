from itertools import combinations
from random import random
import numpy as np
import pandas as pd

df = None

df_preds = pd.DataFrame()

def log(combiner, hx, tx, mx, num_humans):
    global df

    df = pd.DataFrame(hx)
    cols = []
    for i in range(len(df.columns)):
        cols.append(f'Human {df.columns[i]}')
    df.columns = cols
    df['Model'] = pd.DataFrame(mx)[0]
    df['True'] = pd.DataFrame(tx)[0]
    
    with open(f'./log/Confusion-Matrix.md', 'w') as f:
        f.write("# Confusion Matrix\n")
        for i in range(num_humans):
            f.write(f"## {i}\n")
            for j in combiner.confusion_matrix[i]:
                for k in j:
                    f.write(f"{k: .2f} ")
                f.write("\n")
    return df

def add_predictions(policy_name, predictions):
    df_preds[f'Prediction with {policy_name}'] = predictions

def add_predictions_to_policy():
    for c in df_preds:
        df[c] = df_preds[c]


def update_data_policy(df, optimal, policy_name):
    df[f'Using {policy_name}'] = pd.DataFrame(optimal)[0].transform(lambda x: '['+' '.join(map(str, x))+']')

def dump_policy(i):
    df.to_csv(f"./log/policy_{i}.csv")

def lb_best_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    def f(x):
        return x / (1 - x)

    policy_name = "lb_best_policy"

    optimal = []

    t = 0

    for p in hx:
        m = np.array([[f(combiner.confusion_matrix[i][p[i]][j]) for j in range(num_classes)] for i in range(num_humans)])
        m *= (m > 1)
        m += (m == 0) * 1

        y_opt = tx[t]

        optimal.append([i for i, x in enumerate(m[:, y_opt]) if x != 1])

        t += 1
    
    optimal = np.array(optimal)
    
    return optimal
        
def single_best_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    '''
    return best human only in all the cases
    '''

    policy_name = "single_best_policy"

    # we can estimate accuracy for i_th human as (combiner.confusion_matrix[i].trace() / num_classes)

    accuracy = [(combiner.confusion_matrix[i].trace() / num_classes) for i in range(num_humans)]
    best = accuracy.index(max(accuracy))

    optimal = np.array([None for _ in range(len(hx))], dtype=object)
    for i in range(len(hx)):
        optimal[i] = [best]
    
    return optimal

def mode_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    '''
    return a single human which denotes the mode of the subset
    '''

    policy_name = "mode_policy"

    mode = []

    for t in range(len(hx)):
        majority = [0 for _ in range(num_classes)]
        for i in range(num_humans):
            majority[hx[t][i]] += 1
        mode.append([(list(hx[t])).index(majority.index(max(majority)))])
    
    optimal = np.array([None for _ in range(len(mode))])
    for i in range(len(mode)): optimal[i] = mode[i]

    return mode

def weighted_mode_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    '''
    return a single human which denotes the weighted mode of the subset
    '''

    policy_name = "mode_policy"

    mode = []

    for t in range(len(hx)):
        weighted_majority = [0 for _ in range(num_classes)]
        for i in range(num_humans):
            weighted_majority[hx[t][i]] += combiner.confusion_matrix[i].trace() / num_classes
        mode.append([(list(hx[t])).index(weighted_majority.index(max(weighted_majority)))])
    
    optimal = np.array([None for _ in range(len(mode))])
    for i in range(len(mode)): optimal[i] = mode[i]

    return mode

def select_all_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    return np.array([[i for i in range(num_humans)] for _ in range(len(hx))])

def random_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    '''
    return a random subset
    '''

    random = []
    policy_name = "random_policy"

    humans = list(range(num_humans))

    for t in range(len(hx)):
        random_selection = []
        for i in humans:
            if (np.random.random() < 0.5):
                random_selection.append(i)

        random.append(random_selection)

    optimal = np.array(random)

    return optimal

def pseudo_lb_best_policy_overloaded(combiner, hx, tx, mx, num_humans, num_classes=10):

    def f(x):
        return x / (1 - x)

    policy_name = "pseudo_lb_best_policy_overloaded"

    optimal = []

    for p in hx:
        m = np.array([[f(combiner.confusion_matrix[i][p[i]][j]) for j in range(num_classes)] for i in range(num_humans)])
        m *= (m > 1)
        m += (m == 0) * 1

        y_opt = np.argmax(np.prod(m, axis=0))

        optimal.append([i for i, x in enumerate(m[:, y_opt]) if x != 1])
    
    optimal = np.array(optimal)
    
    return optimal