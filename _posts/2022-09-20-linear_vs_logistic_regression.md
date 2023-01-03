---
title: 'Linear vs. Logistic Regression'
date: 2022-09-20
permalink: /posts/2022/09/linear_vs_logistic_regression/
tags:
  - Machine Learning
  - Linear Regression
  - Logistic Regression
---

Linear Regression, Logistic Regression, and their difference

<br />

# Linear Regression

In linear regression, we assume a linear relationship between random variable $X$ and continuous random variable $Y$:

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_nX_n + \epsilon$$

where $Y$ is the response variable, $X$ is the predictor variable, $\beta$ is unknown constant parameter, and $\epsilon$ is the random noise which is assumed to be independent and identically distributed from the Normal distribution with mean 0 and some constant variance $\sigma^2$.

Then the conditional expectation for $Y$ given $X$: 
    
$$E[Y | X] = X^T\beta$$

By minimizing the residual sum of squares, the ordinary least squares estimate $\hat{\beta}$ for the model parameter $\beta$: 

$$\hat{\beta} = (X^TX)^{-1}X^TY$$

<br />

# Logistic Regression

However, in logistic regression, the response variable $Y$ is a discrete random variable with possible values 0 and 1. 
So $Y|x \sim Ber(p = E[Y|x])$

By calculating the log-odds ratio, the non-linear relationship between $X$ and $Y$ is defined by:

$$logit(E[Y|x]) = log(\frac{E[Y|x]}{1 - E[Y|x]}) = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_nX_n$$
    
where the logic function gives a probability of the response variable $Y$ between 0 and 1.

Then, the conditional expectation of $Y$ given $X$:
    
$$E[Y|x] = \frac{e^{X^T\beta}}{1 + e^{X^T\beta}} = \frac{1}{1 + e^{X^T\beta}}$$

Now, the model parameter $\beta$ could be estimated by the likelihood function:

$$L(\beta) = \prod^n_{i = 1} E[Y|x_i]^{y_i}(1 - E[Y|x_i])^{1-y_i}$$

<br />

# Differences

As described above, the main difference between linear and logistic regression is the type of response variable. In linear regression, the response variable is continuous, while in logistic regression, the response variable is a discrete Bernoulli random variable.

Observe that logistic regression is very similar to linear regression: instead of letting the conditional expectation for response variable be the dot product of predictor variable and parameter in linear regression, logistic regression, in order to give a response variable as a probability belongs to positive class, which is between 0 and 1, let the log-odds ratio of the expectation for response variable to be the dot product of predictor variable and parameter. Thus, the logistic regression model for a predictor variable and response variable is essentially a linear regression model for the predictor variable and the log-odds ratio of the conditional expectation of the response variable.

However, linear regression and logistic regression differ in the method of estimating the parameter of interest as well. Linear regression estimates the parameter through the method of least squares, which minimizes the residual sum of squares of the model, while logistic regression estimates the parameter through the method of maximum likelihood estimation. 

As a result of the different types of response variable the model outputs, linear regression is used to handle the regression problems, while logistic regression is used to handle the classification problems.

To conclude, linear regression and logistic regression take in different types of the response variable, utilize different estimation and optimization methods, and are suitable for different tasks.

