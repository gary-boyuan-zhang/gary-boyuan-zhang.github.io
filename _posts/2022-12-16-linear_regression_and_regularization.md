---
title: 'Linear Regression and Regularization'
date: 2022-12-16
permalink: /posts/2022/12/linear_regression_and_regularization/
tags:
  - Machine Learning
  - Linear Regression
  - Regularization
---


# Recap: Linear Regression

## Problem

In linear regression, suppose we have some input data $X$ which is a matrix with dimensions $d \times n$, and some label $Y$ which is a vector with dimension $n \times 1$. Note that here $n$ corresponds to the number of observations, and $d$ corresponds to the number of features in the input data.

Thus, each $(\vec{x_i}, y_i) \,\, \forall \,\, x_i \in X, y_i \in Y$ would corresponds to an observation in the input data. And each observation(i.e. data point) has dimension $(\mathbb{R}^d, \mathbb{R}^1)$.

Our goal is to find some weight vector $\vec{w}$ with dimension $d \times 1$ such that ideally, for each observation $(\vec{x_i}, y_i)$, we could have $\vec{w} \cdot \vec{x_i} = y_i$.

However, things are always not so simple, and we more usually couldn't find such weight vector $\vec{w}$ that $\vec{w} \cdot \vec{x_i}$ would exaclty equal to $y_i$ every observation. This is because that in reality, we could not always expect our input data to be in perfect linear relationship. Note that this does not means that we couldn't assume a linear relationship between all the features and labels within the input data, but we should always be aware and acknowledge the fact that there might be some noise within the input data which would disturb the 'perfectness' of the linear relationship.

With that being said and acknowledging the fact that noise being existed in the input data, we could then revise our goal to become that finding some weight vector $\vec{w}$ such that the difference between $\vec{w} \cdot \vec{x_i}$ and $y_i$ could be as small as possible for all the observation overall. For a single observation(data point), we want to minimize the difference between $\vec{w} \cdot \vec{x_i}$ and $y_i$ which is just $\vec{w} \cdot \vec{x_i} - y_i$. For all the observations within the dataset, we could consider to sum up the difference $\vec{w} \cdot \vec{x_i}$ and $y_i$ for each single observation, which is just $\sum_i^n{ \left( \vec{w} \cdot \vec{x_i} - y_i \right) }$. However, since some of the difference might be positive and some of the difference might be negative, those positive difference and negative difference would be cancel out when summing together. Thus, in order to avoid the difference being cancel out, we could consider to sum the squared of the difference instead, which is just $$\sum_i^n{ \left( \vec{w} \cdot \vec{x_i} - y_i \right)^2 }$$

Therefore, **our goal in linear regression is just to minize this sum of squared difference, which is known as Sum of Squared Residuals(SSR) or Residual Sum of Squares(RSS)**. And we're calling this as the Loss of our regression model. Just to rewrite the previous formula in terms of matrix multiplication, we have defined the loss of the linear regression. And our objective is just to find the weight vector $\vec{w}$ that minimize the loss. This method we're refering to is known as Ordinary Least Squares(OLS) 
$\begin{align*} Loss = ||X^T\vec{w} - Y||^2 \end{align*}$

## Solution

We want to find the weight vector $\vec{w}$ such that the loss function is minimized. We begin the derivation by expanding the Loss function

$$\begin{align*}
  Loss &= ||X^T\vec{w} - Y||^2 \\
  &= (X^T\vec{w} - Y)^T(X^T\vec{w} - Y) \\
  &= (\vec{w}^TX - Y^T)(X^T\vec{w} - Y) \\
  &= \vec{w}^TXX^T\vec{w} - \vec{w}^TXY - Y^TX^T\vec{w} + Y^TY
\end{align*}$$

To minimize the Loss function, which is dependent on the weight vector $\vec{w}$, we need to take the partial derivative of the Loss function with respect to $\vec{w}$.

$$\begin{align*}
  \frac{\partial Loss}{\partial \vec{w}} &= \frac{\partial}{\partial \vec{w}} (\vec{w}^TXX^T\vec{w}) - \frac{\partial}{\partial \vec{w}}(\vec{w}^TXY) - \frac{\partial}{\partial \vec{w}}(Y^TX^T\vec{w}) + \frac{\partial}{\partial \vec{w}}(Y^TY) \\
  &= XX^T\frac{\partial}{\partial \vec{w}} (\vec{w}^T\vec{w}) - XY\frac{\partial}{\partial \vec{w}}(\vec{w}^T) - XY\frac{\partial}{\partial \vec{w}}(\vec{w}) + 0 \\
  &= XX^T\frac{\partial}{\partial \vec{w}} (||\vec{w}||^2) - XY\frac{\partial}{\partial \vec{w}}(\vec{w}^T) - XY\frac{\partial}{\partial \vec{w}}(\vec{w}) \\
  &= 2XX^T\vec{w} - 2XY \\
\end{align*}$$

Let $\frac{\partial Loss}{\partial \vec{w}} = 0$, then we have

$$\begin{align*}
  2XX^T\vec{w} - 2XY &= 0 \\
  XX^T\vec{w} &= XY \\
  \vec{w} &= (XX^T)^{-1}XY
\end{align*}$$

Therefore, we have derived the solution to the objective of the Ordinary Least Squares(OLS), which is $\vec{w} = (XX^T)^{-1}XY$!

# Non-linear Relationship

Previously, we've derived the solution to the objective function of the OLS under the assumption that there's a linear relationship between features and labels. However, this assumption might not be useful in realistic. So what if we want to model some non-linear relationship between the features and labels?

For example, suppose we have some input data $X$ and label $Y$, where the relationship between each observation $(\vec{x_i}, y_i)$ is defined by $$y_i = w_1x_{i, 1} + w_2(x_{i, 2})^2 + w_3(x_{i, 3})^3 + \cdots + w_p(x_{i, p})^p$$

Noticed that previously under the linear relationship assumption, we could model $y_i = \vec{w} \vec{x_i}$ where $\vec{w} = (w_1, w_2, \dots, w_p)$ and $\vec{x_i} = (x_{i, 1}, x_{i, 2}, \dots , x_{i, p})$. However, obviously, the relationship is no longer linear now, but instead is polynomial. 

For this particular example, we could consider to define a new variable $$\vec{z_i} = [x_{i, 1}, (x_{i, 2})^2, (x_{i, 3})^3, \dots , (x_{i, p})^p]$$

Then we could model the linear relationship between $z_i$ and $y_i$ as $$y_i = \vec{w}\vec{z_i}$$

In fact, in general, we could always model any non-linear relationship between data $X$ and label $Y$ by first transforming the data $X = (\vec{x_1}, \vec{x_2}, \dots, \vec{x_p})$ into some projection $Z = (\vec{z_1}, \vec{z_2}, \dots, \vec{z_p})$ via some function $f_i$ such that $\vec{z_i} = f_i(\vec{x_i})$ for every feature $i$. We could then model a linear relationship between the propjection $Z$ and label $Y$ as $Y = Z^T \vec{w}$, and estimate the weight vector $\vec{w}$ by implementing the OLS as described in the previous section.

# Bias-Variance Tradeoff

In the previous section, we've discussed that we could model any non-linear relationship between data $X$ and label $Y$ by first mapping the data $X$ into some intermediate variable $Z$ via some function $f$, and then model the linear relationship between $Z$ and $Y$, and solve for $\vec{w}$ using OLS.

With this being said, it seems that we could model any polynomial relationship between $X$ and $Y$ via some transfor by finding some transformation $f$ so that $f(\vec{x_i})$ would have a linear relationship with $\vec{y_i}$. But how do we find such transformation $f$ that is suitable? And specifically, how could we know that if the relationship between $f(\vec{x_i})$ after transformation and $y_i$ is linear? This evolves the discussion about the bias and variance tradeoff.

## Experiment

Let's do a simulation experiment to try it out!

We sample 50 independent data sets based on an underlying function(some true relationship $f(x)$ ), then we fit the parameters for the 3 polynomial functions(of different degrees) individually. $g_1(x)$ representes a degree 1 linear function estimation, $g_3(x)$ representes a degree 3 polynomial function estimation, and $g_{10}(x)$ representes a degree 10 polynomial function estimation.

In the generated plots, the lightly-colored curves in each of the three plots below are an individual polynomial model fit $g_i(x)$ to one of the 50 sampled data sets. The darkly-colored curve in each plot is the average over the 50 individual fits $E[g_i(x)]$. The dark curve(black) is the true, underlying function $f(x)$.


```python
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean
np.random.seed(123)
MARKER_SIZE = 100
DATA_COLOR = 'black'
ERROR_COLOR = 'darkred'
POLYNOMIAL_FIT_COLORS = ['orange', 'royalblue', 'darkgreen']
LEGEND_FONTSIZE = 14
TITLE_FONTISIZE = 16
N_OBSERVATIONS = 10
NOISE_STD = 1.

polynomial_degrees = [1, 3, 10]

x = 2 * (np.random.rand(N_OBSERVATIONS) - .5)
x_grid = np.linspace(-1, 1, 100)
def f(x):
    """Base function"""
    return np.sin(x * np.pi)


def sample_fx_data(shape, noise_std=NOISE_STD):
    return f(x) + np.random.randn(*shape) * noise_std


def plot_fx_data(y=None):
    """Plot f(x) and noisy samples"""
    y = y if y is not None else sample_fx_data(x.shape)
    fig, axs = plt.subplots(figsize=(6, 6))
    plt.plot(x_grid, f(x_grid), color=DATA_COLOR, label='f(x)')
    plt.scatter(x, y, s=MARKER_SIZE, edgecolor=DATA_COLOR, facecolors='none', label='y')


n_simulations = 50
simulation_fits = defaultdict(list)
for sim in range(n_simulations):
    # Start from same samples
    y_simulation = sample_fx_data(x.shape)
    for degree in polynomial_degrees:
        # Note: we should get an overconditioned warning
        # for degree 10 because of extreme overfitting
        theta_tmp = np.polyfit(x, y_simulation, degree)
        simulation_fits[degree].append(np.polyval(theta_tmp, x_grid))


def error_function(pred, actual):
    return (pred - actual) ** 2

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ii, degree in enumerate(polynomial_degrees):
    plt.sca(axs[ii])
    for jj, fit in enumerate(simulation_fits[degree]):
        label = 'Single Simulation Model Fit' if jj == 0 else None
        plt.plot(x_grid, fit, color=POLYNOMIAL_FIT_COLORS[ii], alpha=.1, label=label)

    average_fit = np.array(simulation_fits[degree]).mean(0)
    squared_error = error_function(average_fit, f(x_grid))
    rms = np.sqrt(mean(squared_error))

    # Plot values and make the graph look good
    plt.plot(x_grid, average_fit, color=POLYNOMIAL_FIT_COLORS[ii], linewidth=3, label='Average Model')
    plt.plot(x_grid, squared_error, '--', color=ERROR_COLOR, label='Squared Error')
    plt.plot(x_grid, f(x_grid), color='black', label='f(x)')
    plt.yticks([])
    if ii == 1:
        plt.xlabel('x')
    elif ii == 0:
        plt.ylabel('y')
        plt.yticks([-2, 0, 2])

    plt.xlim([-1, 1])
    plt.ylim([-2, 2])
    plt.xticks([-1, 0, 1])
    plt.title(f"$g_{degree}(x)$ : RMS Error={np.round(rms, 1)}")
    plt.legend(loc='lower right')
plt.suptitle('Model Fits Given Random Samples Around f(x)', fontsize=TITLE_FONTISIZE)
plt.show()
```


    
![png](https://drive.google.com/uc?id=12kjQymTzznvoYkV-gShIiozLNhTygu7d)
    


**Looking at these three plots, what did you observe? Which degree polynomial function transformation $g(x)$ would you prefer? Why?**

We are going to address this question from two perspectives: Bias and Variance.

## Bias

Recall that bias is defined by $$\text{Bias}= E[g(x)]−f(x)$$

Bias describes how much the average estimator fit $E[g(x)]$ over many datasets deviates from the value of the actual underlying target function $f(x)$.

Looking at these three plots above, notice that

- Degree=1: the estimator $g_1(x)$ has the average estimator $E[g_1(x)]$
 (dark orange curve) deviates a lot from the true function $f(x)$, indicating that the estimator $g_1(x)$ has high bias.

- Degree=3: the estimator $g_3(x)$ has the average estimator $E[g_3(x)]$
 (dark blue curve) accurately approximates the true function $f(x)$, indicating that the estimator $g_3(x)$ has low bias.

- Degree=10: what is your guess??

## Variance

Recall that variance is defined by $$\text{Variance}=E[(g(x) −E[g(x)])^2]$$


Variance is the expected (i.e. average) squared difference between any single dataset-dependent estimate of $g(x)$ and the average value of $g(x)$ estimated over all datasets.

Looking at these three plots above, notice that

- estimator $g_1(x)$ (orange curves; right most plot) has low variance as each individual $g_1(x)$ is fairly similar across datasets.

- estimator $g_{10}(x)$ (green curves; right most plot) has high variance as each individual model fit varies dramatically from one data set to another.




## Overfitting and Underfitting



The problem of Underfitting occurs when an estimator (in our case ML model) is not flexible enough to capture the underlying trends in the observed data. Underfitting generally results when model is unable to learn through features. Few causes can be: bad optimizer, incorrect loss function, not enough training iterations. 
Underfitting usually occurs when a model has high bias.

The problem of Overfitting occurs when an estimator is too flexible, allowing it to capture illusory trends in the data. These illusory trends are often the result of the noise in the observations.
To identify overfitting, we can observe a very high accuracy/f-score on training dataset and low accuracy/f-score on test dataset. 
It can also be observed by plotting train and validation loss with iterations. In case of overfitting, Training loss will decrease with iterations and validation loss would start to increase after some iterations indicating that the model is overfitting.
Overfitting usually occurs when a model has high variance.

## Balance

Recall that previously we've introduced the loss in OLS as the Residual Sum of Squares in the model. However, if we take devide the Residual Sum of Squares (RSS) in the model by the degree of freedom(number of samples in the data substract number of features in the model), we would get another popular metrics to evaluate model performance, which is Mean Squared Error (MSE)
$$MSE = \frac{1}{n} \sum_i^n{ \left( \vec{w} \cdot \vec{x_i} - y_i \right)^2 }$$

With simple mathematical derivation, we could observe that

$$\begin{align*}
  MSE &= E[( \vec{w} \cdot \vec{x_i} - y_i )^2] \\
  &= Var(\vec{w} \cdot \vec{x_i} - y_i) - E[\vec{w} \cdot \vec{x_i} - y_i]^2 \\
  &= Variance + Bias^2
\end{align*}$$

This derivation has important implication, as it shows that the Mean Squared Error of a model is composed of both Bias and Variance. That is, both Bias and Variance would increase the loss of our model. Thus, in order to fulfill our objective which is to minimize the loss, we have to find a balance between Bias and Variance!

With the experiment above, we can identify that if our first model, $g_1(x)$, is too simple and has very few polynomial degrees then it may have high bias and low variance. On the other hand the third model, $g_{10}(x)$ has large polynomial degrees, and has high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data. This can be observed in the middle model and this is called Bias Variance Tradeoff.

<img src='https://ninjacatorg.files.wordpress.com/2022/09/84cd5015-f709-4d3f-8cf9-d5493d8d6e4d.jpeg'>

# Regularization

With that being said, we want to train our model so that it could leanred the trend and relationship within the observed data, which is to reduce the bias of our model and avoiding the underfitting problem. However, in the meantime, we also want to control for the model complexity so that the model won't over capture the illusory trends in the data, which is to control the variance and avoiding the overfitting problem. In order to achieve this purpose, it might be a good idea to consider implementing regularization.

Regularization techniques are used to improve generalization of a model by reducing its complexity. The main idea behind regularization is that simpler models are preferable, as it would possibly improve on both prediction accuracy and model interpretability. By using simpler model, our goal is to avoid the overfitting problem, reduce variance, and thus reduce the loss of our model.

Simpler models might improve prediction accuracy as it would eliminate uninformative predictors that can lead to lower variance, at the expense of slight increase in bias. Also, eliminating those uninformative predictors might be also helpful for you to find the true relationship between other predictors and the response.

## Ridge Regression

In Ordinary Least Squares, our objective is to minimize the loss function $||X^T\vec{w} - Y||^2$, and we've derived the solution, which is $\vec{w} = (XX^T)^{-1}XY$.

In Ridge Regression, since we want to control for the complexity of the model, we are introducing a shrinkage penalty $\lambda \geq 0$, such that we are ensuring the $L_2$ norm of the weight vector $||\vec{w}||^2 \leq c^2$ is less than some constant threshold $c$.

With this penalization term being added, we could rewrite the loss function in the Ridge Regression, also known as $L_2$ regularization loss, as $$Loss_{L_2} = ||X^T\vec{w} - Y||^2 + \lambda ||\vec{w}||^2$$

Similary to OLS, our objective here is to minimize the loss function. By applying similar derivation as in OLS, we could get the solution $$\vec{w} = (XX^T + \lambda I)^{-1}XY$$

Notice that different from OLS, here we've introduced $\lambda$ which is a hyperparameter we have to tune and choose for. As $\lambda$ increases, the penalization becomes harsher, and the flexibility of models decreases, which would increase the bias of the model, but decreases the variance of the model.
To tune the hyperparameter $\lambda$, we could utilize techniques such as cross-validation. 

Below is an example demonstrating how the magnitude of the coefficient for `Income` trends as $\lambda \rightarrow \infty$

<img src = 'https://drive.google.com/uc?id=17oIcgPMf-Cn0jPRZtPb-aDnUZXYhzqsO'>

Noticing that as $\lambda$ increases, the coefficient shrinks towards zero, but they could never actually reaches it.

## Lasso Regression

Ridge Regression has effectively control the weight from being deviate too much from 0, but add the $L_2$ regularization term into the loss function. However, Ridge Regression would still keep all the variables(features) in the model.

Different from Ridge Regression, Lasso Regression implements the $L_1$ regularization term into the loss function, to not only control the weight from being deviate too much from 0, but could even eliminate uninformative variables by shrinking their weight eventually becomes 0. Thus, Lasso Regression could control for the complexity of the model more efficiently, especially when we may believe there is a sparse solution of our task.

In Lasso Regression, we are ensuring the $L_1$ norm of the weight vector $||\vec{w}||1 \leq c$ is less than some constant threshold $c$.

With this penalization term being added, we could rewrite the loss function in the Lasso Regression, also known as $L_1$ regularization loss, as $$Loss_{L_1} = ||X^T\vec{w} - Y||^2 + \lambda ||\vec{w}||_1$$

Similar to Ridge Regression, we're introducing $\lambda$ which is a hyperparameter we have to tune and choose for. As $\lambda$ increases, the penalization becomes harsher, and the flexibility of models decreases, which would increase the bias of the model, but decreases the variance of the model.
To tune the hyperparameter $\lambda$, we could utilize techniques such as cross-validation. 

Below is an example demonstrating how the magnitude of the coefficient for `Income` trends as $\lambda \rightarrow \infty$

<img src = 'https://drive.google.com/uc?id=1OrTrpcxiH88IyIKZUmmmTdp8n1TZYa05'>

Noticing that as $\lambda$ increases, the coefficient shrinks towards and eventually equals zero at $\lambda \approx 1000$. This means if if we choose a larger $\lambda$, then `Income` would not be included as a feature in the leanred model!


<br />

<br />

<br />

# Exercise

Now let's practice implementing regularization techniques in Linear Regression in Python using the Scikit-learn library!

In `sklearn`, the `Ridge()` classifier is used for $L_2$(Ridge) Regularization and the `Lasso()` classifier is used for $L_1$(Lasso) Regularization.


```python
# loading the required libraries
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
```


```python
# Data loading and train-test split
dataset = datasets.load_diabetes()
train_x, test_x, train_y, test_y = train_test_split(dataset.data,dataset.target,test_size=0.2, shuffle = True, random_state=23)
```

Let's begin with using a Linear Regression model first. We will then use $L_1, L_2$ regularization and compare performances.


```python
from sklearn.linear_model import LinearRegression

# Simple Linear Regression
lr = LinearRegression()
lr.fit(train_x, train_y)
pred_lr = lr.predict(test_x)

mse_score = metrics.mean_squared_error(test_y, pred_lr)
print("MSE for Linear Regression: ", mse_score)
```

    MSE for Linear Regression:  3171.547252709534


In the following cell, we run through different values for alpha (in `sklearn`, $\alpha = \frac{1}{\lambda}$) and check the changes in MSE error for each. 

For L1 penalty, we have used Sklearn's Lasso model and set the maximum iterations to 10000.


```python

alpha_values = [1, 0.1, 0.01, 1e-3, 1e-5, 1e-7 ,1e-10]

print("Prediction Errors for LR with L1 penalty")

for alpha in alpha_values:
  # instantiate model
  lr_l1 = sklearn.linear_model.Lasso(alpha = alpha, max_iter = 10000)
  # fit data into model
  lr_l1.fit(train_x, train_y)
  
  pred_l1 = lr_l1.predict(test_x)
  # append the mse scores into the list
  mse_l1 = metrics.mean_squared_error(test_y, pred_l1)
  print("Alpha:", alpha,", MSE:", mse_l1)
```

    Prediction Errors for LR with L1 penalty
    Alpha: 1 , MSE: 3967.9867570632737
    Alpha: 0.1 , MSE: 3100.5002920850584
    Alpha: 0.01 , MSE: 3140.8859178042044
    Alpha: 0.001 , MSE: 3165.113066456015
    Alpha: 1e-05 , MSE: 3171.4799827209845
    Alpha: 1e-07 , MSE: 3171.546579685444
    Alpha: 1e-10 , MSE: 3171.547252036498


Use L2 regularization for Linear Regression and print the MSE error values for all the given values of alpha.


```python
alpha_values = [1, 0.1, 0.01, 1e-3, 1e-5, 1e-7 ,1e-10]

print("Prediction Errors for LR with L2 penalty")

# TO DO: Use sklearn.linear_model.Ridge to build  L2 regularization model, use max_iter = 10000

for alpha in alpha_values:
  # instantiate model
  lr_l2 = sklearn.linear_model.Lasso(alpha = alpha, max_iter = 10000)
  # fit data into model
  lr_l2.fit(train_x, train_y)
  
  pred_l2 = lr_l2.predict(test_x)
  # append the mse scores into the list
  mse_l2 = metrics.mean_squared_error(test_y, pred_l2)
  print("Alpha:", alpha,", MSE:", mse_l2)
```

    Prediction Errors for LR with L2 penalty
    Alpha: 1 , MSE: 3967.9867570632737
    Alpha: 0.1 , MSE: 3100.5002920850584
    Alpha: 0.01 , MSE: 3140.8859178042044
    Alpha: 0.001 , MSE: 3165.113066456015
    Alpha: 1e-05 , MSE: 3171.4799827209845
    Alpha: 1e-07 , MSE: 3171.546579685444
    Alpha: 1e-10 , MSE: 3171.547252036498

<br />
<br />
<br />

# References

- L. Pinto. (2022). CSCI-UA 473 Introduction to Machine Learning, Fall 2022. Campuswire. https://campuswire.com/c/G6C251796

- P. Dube. (2022). DS-UA 301 Advanced Topics in Data Science: Advanced Techniques in ML and Deep Learning, Fall 2022. NYU Brightspace. https://brightspace.nyu.edu/d2l/home/219804

- R. Yurko. (2022). The Summer Undergraduate Research Experience in Statistics, Summer 2022. CMU Department of Statistics & Data Science. https://www.stat.cmu.edu/cmsac/sure/2022/materials/index.html

- F. Pedregosa, et al. (2011). Scikit-learn. https://scikit-learn.org
