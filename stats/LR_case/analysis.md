```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
```

# Problem 1

### Hypothesis Testing using Simulation (10 pts)
The dataset iris contains measurements on three species of iris flowers. In this assignment, you’ll focus only on the 50 observations with Species == "versicolor". Complete both of the following tasks and report results and interpretations.
A botanist claims that the average sepal length of her Iris versicolor plants is 5.5 cm. You will test this claim via simulation.


```python
iris = load_iris(as_frame=True)
df = iris.frame
df=df[df['target'] == 1] # only versicolor
observed_mean = df['sepal length (cm)'].mean()
print(observed_mean)
```

    5.936



```python
observed_mean = df['sepal length (cm)'].mean()
num_simulations = 1_000_000_000
n = len(df)
sample_std = df['sepal length (cm)'].std(ddof=1)
print(sample_std)

CLAIMED_MEAN = 5.5

simulated_means = np.random.normal(loc=CLAIMED_MEAN, scale=sample_std/np.sqrt(n), size=num_simulations)

```

    0.5161711470638635



```python
observed_diff = np.abs(observed_mean - CLAIMED_MEAN)

extreme_count = np.sum(np.abs(simulated_means - CLAIMED_MEAN) >= observed_diff)

p_value = extreme_count / num_simulations
print(f"p-value: {p_value}")
```

    p-value: 2e-09



```python
plt.hist(simulated_means, bins=40, alpha=0.7, edgecolor='black')
plt.axvline(observed_mean, color='red', linestyle='--', label=f'Observed mean = {observed_mean:.2f}')
plt.title("Simulated Distribution of Lengths")
plt.xlabel("Mean")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```


    
![png](analysis_files/analysis_5_0.png)
    


# Intepretation and Results

The actual observed mean of the data was 5.94 with a standard deviation of 1.88.

In order to test the assumption that the true mean was 5.5, I ran 10,000,000 simulations. This resulted in a p-value of about 2e-9 Since this is dramatically less than 0.05, we **reject the null hypothesis** that the true mean is 5.5.

# Problem 2

### Simple linear regression: investigate if wind speed largely explains ozone variation
a. Initial exploration - make a scatterplot and calculate correlation coefficient to identify the strength of their relationship.

b. Fit a model, write down the fitted model and interpret: the coefficient estimates, model performance.

c. Predict the ozone concentration when wind = 10 mph.

d. Check model assumptions using diagnostic plots.
### Multiple Linear regression: include wind speed, temperature and solar radiation to improve predictions
a. Multicollinearity check using pairwise correlation plot.

b. Fit a model, write down the fitted model and interpret: the coefficient estimates, model performance.

c. Check model assumptions using diagnostic plots.


```python
df = pd.read_csv('airquality.csv')

```


