# 1. Plot the timeseries data

# 2. What is the approximate duration of each event? (just put this in the value in the code that says approx_duration = "your value here")

# 3. Code the maximum likelihood state sequence for each time point (transitions and likelihood functions given in the code).

# 4. Plot the output of problem 3 with 4 overlayed plots with 4 different colors and a legend (this will require making a dataframe with base or state and 0 or 1 for whether that state is active or not, you can use a line plot with 0 and 1 for active or inactive).

# 5. Given the approximate duration of each event found in 2, give the most likely DNA sequence this data corresponds to.

# 6. Code the forward/backward algorithm giving the posterior probability of each time point to each level.

# 7. Plot the posterior probability of each time point to each level (this time, use facet_wrap to make 4 graphs from one dataframe, this will require making a dataframe with posterior and base columns).
import numpy as np
import pandas as pd
import scipy
from plotnine import *

df = pd.read_csv('nanopore.csv')

'''
Homework 4 problem 1 -- Plot data (please save to file, dont just print it)
plot the timeseries data for simulated nanopore
'''
def plot_timeseries_data(data):
    plot = (
        ggplot(df, aes(x='time', y='level')) +
        geom_line(color='steelblue') +
        geom_point(color='steelblue', alpha=0.5) +
        labs(
            title='Nanopore Time Series Data',
            x='Time',
            y='Level'
        )
    )   
    plot.save("nanopore_timeseries.png", width=8, height=6, dpi=300)

'''
Homework 4 problem 2
What is the approximate duration of each "event" in this data given this plot?
'''
approx_duration = "200"

'''
Homework 4 problem 3 -- HMM maximum likelihood state sequence with 4 states
state 1 - T corresponds to a normal distribution with mean 100 and sd 15
state 2 - A corresponds to a normal dist with mean 150 and sd 25
state 3 - G correcponds to a normal dist with mean 300 and sd 50
state 4 - C corresponds to a normal dist with mean 350 and sd 25
transitions between states are 1/50 and transitions to same state is 49/50
'''
states = ['T', 'A', 'G', 'C']
state_params = {
    'T': {'mean': 100, 'std': 15},
    'A': {'mean': 150, 'std': 25},
    'G': {'mean': 300, 'std': 50},
    'C': {'mean': 350, 'std': 25}
}
stay_prob = 49/50
switch_prob = (1/50)/(len(states)-1)

transition_probs = {}

for i in range(len(states)):
    for j in range(len(states)):
        if i == j:
            transition_probs[(states[i], states[j])] = stay_prob
        else:
            transition_probs[(states[i], states[j])] = switch_prob
def HMM_MLE(df):
    values = df['level'].values
    # Define a time x len(states) matrix
    V = np.zeros(len(values), len(states))
    B = np.zeros(len(values), len(states), dtype=int)
    
    # init the likelihood of where the first datapoint came form
    for i, state in enumerate(states):
        mean = state_params[state]['mean']
        std = state_params[state]['std']
        V[0, i] = scipy.stats.norm.pdf(values[0], mean, std)
    
    for time in range(1, len(values)):
        for curr_idx, state in enumerate(states):
            max_prob = -1
            best_prev_state_idx = 0
            curr_mean = state_params[state]['mean']
            curr_std = state_params[state]['std']
            
            curr_prob = scipy.stats.norm.pdf(values[time], curr_mean, curr_std)
            
            # figure out prob it came from prev state
            for prev_idx, prev_state in enumerate(states):
                prob = V[time-1, prev_idx] * transition_probs[(prev_state, state)] * curr_prob
                
                if prob > max_prob:
                    max_prob = prob
                    best_prev_state_idx = prev_idx
                    
            V[time, curr_idx] = max_prob
            B[time, curr_idx] = best_prev_state_idx
            
    # Go backwards from B and figure out the path we took
    path = []
    curr_idx = np.argmax(V[-1])
    path.append(states[curr_idx])
    
    for time in range(len(values)-1, 0, -1):
        curr_idx = B[time, curr_idx]
        path.append(states[curr_idx])
    
    path.reverse()
    return path
        
        
'''
Homework 4 problem 4
plot output of problem 3. Here, please make 1 plot with 4 plots overlayed with different colors.
'''
def plot_MLE(state_sequence):
    print("your code here")

'''
Homework 4 problem 5
Give the most likely sequence this data corresponds to given the likely 
event length you found from plotting the data
print this sequence of A/C/G/Ts
'''
def MLE_seq(df, event_length):
    print("your code here")


'''
Homework 4 problem 6
Forward/backward algorithm giving posterior probabilities for each time point for each level
'''
def HMM_posterior(df):
    print("your code here")


'''
Homework 4 problem 7
plot output of problem 5, this time, plot with 4 facets using facet_wrap
'''
def plot_posterior(posteriors):
    print("your code here")


df = pd.read_csv("nanopore.csv")
plot_timeseries_data(df)
# state_sequence = HMM_MLE(df)
# plot_MLE(state_sequence)
# MLE_seq(df, event_length)
# posteriors = HMM_posterior(df)
# plot_posteriors(posteriors)

