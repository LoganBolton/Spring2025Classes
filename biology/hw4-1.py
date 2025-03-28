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

'''
Homework 4 problem 1 -- Plot data (please save to file, dont just print it)
plot the timeseries data for simulated nanopore
'''
def plot_timeseries_data(data):
    plot = (
        ggplot(data, aes(x='time', y='level')) +
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
approx_duration = 200

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
            transition_probs[(states[i], states[j])] = np.log(stay_prob)
        else:
            transition_probs[(states[i], states[j])] = np.log(switch_prob)
            
def HMM_MLE(df):
    values = df['level'].values
    # Define a time x len(states) matrix
    V = np.full((len(values), len(states)), -np.inf)  # log(0) = -inf so that's why I'm not using 0
    B = np.zeros((len(values), len(states)), dtype=int)
    
    # init the likelihood of where the first datapoint came form
    for i, state in enumerate(states):
        mean = state_params[state]['mean']
        std = state_params[state]['std']
        V[0, i] = np.log(scipy.stats.norm.pdf(values[0], mean, std))
    
    for time in range(1, len(values)):
        for curr_idx, state in enumerate(states):
            max_prob = -np.inf
            best_prev_state_idx = 0
            curr_mean = state_params[state]['mean']
            curr_std = state_params[state]['std']
            
            curr_prob = np.log(scipy.stats.norm.pdf(values[time], curr_mean, curr_std))
            
            # figure out prob it came from prev state
            for prev_idx, prev_state in enumerate(states):
                prob = V[time-1, prev_idx] + transition_probs[(prev_state, state)] + curr_prob
                
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
    data = []
    for t, state in enumerate(state_sequence):
        for s in states:
            data.append({
                'time': t,
                'base': s,
                'active': 1 if s == state else 0
            })

    df_plot = pd.DataFrame(data)

    plot = (
        ggplot(df_plot, aes(x='time', y='active', color='base'))
        + geom_line(size=1)
        + labs(
            title='Most Likely Genetic Base Over Time',
            x='Time',
            y='Active (1=Yes, 0=No)'
        )
        + scale_y_continuous(breaks=[0, 1])
        + theme()
    )

    plot.save("MLE_state_sequence_overlayed.png", width=10, height=6, dpi=300)
    return plot

'''
Homework 4 problem 5
Give the most likely sequence this data corresponds to given the likely 
event length you found from plotting the data
print this sequence of A/C/G/Ts
'''
def MLE_seq(df, event_length):
    state_sequence = HMM_MLE(df)
    sequence = []

    for i in range(0, len(state_sequence), event_length):
        chunk = state_sequence[i:i + event_length]
        
        counts = {'T': 0, 'A': 0, 'G': 0, 'C': 0}
        for base in chunk:
            counts[base] += 1
        
        most_common_base = max(counts, key=counts.get)
        sequence.append(most_common_base)
    print("---------------------------------------")
    print("Most likely DNA sequence:")
    print(''.join(sequence))
    print("---------------------------------------")


'''
Homework 4 problem 6
Forward/backward algorithm giving posterior probabilities for each time point for each level
'''
def emission_prob(value, state):
    mean = state_params[state]['mean']
    std = state_params[state]['std']
    return np.log(scipy.stats.norm.pdf(value, mean, std))

def HMM_posterior(df):
    values = df['level'].values
    
    # init forward pass to be probability zero
    alpha = np.full((len(df), len(states)), -np.inf)
    for state_idx, state in enumerate(states):
        alpha[0, state_idx] = emission_prob(values[0], state)
        
    # forward pass
    for time in range(1, len(values)):
        for state_idx, state in enumerate(states):
            emis_prob = emission_prob(values[time], state)
            all_prev = []
            for prev_state_idx, prev_state in enumerate(states):
                all_prev.append(alpha[time-1, prev_state_idx] + transition_probs[(prev_state, state)] + emis_prob)

            max_val = np.max(all_prev)
            sum_exp = np.sum(np.exp(all_prev - max_val))
            alpha[time, state_idx] = max_val + np.log(sum_exp)
    
    # backward pass
    # init to 100% == log(1) == log(0)
    beta = np.full((len(df), len(states)), 0)
    
    for time in range(len(df)-2, -1, -1):
        for state_idx, state in enumerate(states):
            all_next = []
            for next_state_idx, next_state in enumerate(states):
                emis = emission_prob(values[time+1], next_state)
                transition = transition_probs[(state, next_state)]
                all_next.append(transition + emis + beta[time+1, next_state_idx])
            
            max_val = np.max(all_next)
            sum_exp = np.sum(np.exp(all_next - max_val))
            beta[time, state_idx] = max_val + np.log(sum_exp)
    
    # combine forward and backwards
    posteriors = np.zeros((len(df), len(states)))
    for time in range(len(df)):
        for state_idx in range(len(states)):
            posteriors[time, state_idx] = alpha[time, state_idx] + beta[time, state_idx]
    
    # normalize
    for t in range(len(df)):
        row = posteriors[t]     
        max_val = np.max(row)
        sum_exp = np.sum(np.exp(row - max_val))
        log_sum = max_val + np.log(sum_exp)
        
        posteriors[t] = np.exp(row - log_sum)
            
    return posteriors

'''
Homework 4 problem 7
plot output of problem 5, this time, plot with 4 facets using facet_wrap
'''
def plot_posteriors(posteriors):
    plot_data = []
    for t in range(len(df)):
        for s_idx, s in enumerate(states):
            plot_data.append({
                'time': t,
                'state': s,
                'posterior': posteriors[t, s_idx]
            })

    df_plot = pd.DataFrame(plot_data)
    
    p = (
        ggplot(df_plot, aes(x='time', y='posterior')) +
        geom_line() +
        facet_wrap('~state', ncol=2) +
        labs(
            title='Posterior Probabilities',
            x='Time',
            y='Posterior Probability'
        ) +
        theme(subplots_adjust={'wspace': 0.25, 'hspace': 0.3})
    )
    
    file_name = "posterior_probabilities_faceted.png"
    p.save(file_name, width=10, height=8, dpi=300)
    print(f"saved to {file_name}")
    return p

df = pd.read_csv("nanopore.csv")
plot_timeseries_data(df)
state_sequence = HMM_MLE(df)
plot_MLE(state_sequence)
MLE_seq(df, event_length=approx_duration)
posteriors = HMM_posterior(df)
plot_posteriors(posteriors)