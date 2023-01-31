---
title: 'Causal Inference: An Example on Estimating ITT and LATE via Instrumental Variable'
date: 2023-01-31
permalink: /posts/2023/01/causal_inference_instrumental_variable_example/
tags:
  - Causal Inference
  - Instrumental Variable
  - Intention-to-treat (ITT)
  - Local Average Treatment Effect (LATE)
  - Public Policy
  - Public Health
---


An example estimating the Intention-to-treat (ITT) effect and the Local Average Treatment Effect (LATE) of a public health policy.

<br/>
<br/>

```python
import numpy as np
import pandas as pd
```

# 1. Introduction

Patients who get surgery, for example for orthopaedic reasons, are often advised by the doctors, subsequently to surgery, to get physiotherapy, that is, a series of exercises to help rehabilitation and more complete recovery. However, the costs of physiotherapy may often deter patients from following it. **It is therefore important to try to show the potential benefits of physiotherapy, so that more patients can become convinced to follow it.**

## 1.1 Setting

In the period of 4 years, three cooperating hospitals randomly assigned each of
the 537 eligible patients, who had gone through an orthopeadic operation, in one
of two groups: patients in the first group, ( $Z_i = 1$ ), were offered the opportunity to get physiotherapy at 50% reduced hospital fees; for patients assigned in the second group, physiotherapy was available at the standard cost.

For each patient, the recorded variables, in addition to assignment $Z_i$ , are: whether or not the patient got physiotherapy, $T^{obs}_i = 1$ for yes, 0 for no; an assessment of the patients recovery 3 months after surgery, $Y^{obs}_i = 1$ for satisfactory, 0 for unsatisfactory or poor. 

The assessment of this studys data was done by physicians blinded to both the assignment $Z_i$ and the taking (or not) of physiotherapy by the patient.

## 1.2 Data

The table below gives the counts, $n_{zty}$, of patients assigned $Z_i = z$ and with physiotherapy-taking status $T^{obs} = t$ and outcome $Y^{obs} = y$.


<div align="center">
  <table border="5" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>$Z$</th>
      <th>$T^{obs}$</th>
      <th>$Y^{obs}$</th>
      <th>$n$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>123</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>96</td>
    </tr>
  </tbody>
</table>
</div>

We could manually construct the dataframe in python according to the above table.


```python
n0 = np.repeat(np.array([[0, 0, 0]]), repeats=185, axis=0)
n1 = np.repeat(np.array([[0, 0, 1]]), repeats=123, axis=0)
n2 = np.repeat(np.array([[0, 1, 0]]), repeats=9, axis=0) 
n3 = np.repeat(np.array([[0, 1, 1]]), repeats=41, axis=0) 
n4 = np.repeat(np.array([[1, 0, 0]]), repeats=37, axis=0) 
n5 = np.repeat(np.array([[1, 0, 1]]), repeats=20, axis=0) 
n6 = np.repeat(np.array([[1, 1, 0]]), repeats=26, axis=0) 
n7 = np.repeat(np.array([[1, 1, 1]]), repeats=96, axis=0) 

df = pd.DataFrame(np.concatenate([n0, n1, n2, n3, n4, n5, n6, n7]),
                  columns = ['Z', 'T_obs', 'Y_obs'])
df.sample(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z</th>
      <th>T_obs</th>
      <th>Y_obs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>235</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>249</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>370</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>216</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>479</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>139</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>280</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

<br />

# 2. The Intention-to-treat (ITT) Effect

First, we estimate the intention-to-treat (ITT) effect of offering the discount on the improvement of recovery, $$E[Y (Z = 1)] − E[Y (Z = 0)]$$, using a difference-in-means estimator. We also estimate the standard error and the asymptotic 95% confidence interval. 



```python
import scipy.stats as sps

itt = df[df.Z == 1].Y_obs.mean() - df[df.Z == 0].Y_obs.mean()
itt_var = df[df.Z == 1].Y_obs.var() / len(df[df.Z == 1]) \
            + df[df.Z == 0].Y_obs.var() / len(df[df.Z == 0])

itt_se = np.sqrt(itt_var)

itt_ci95 = [itt - sps.norm.ppf(.975) * itt_se,
            itt + sps.norm.ppf(.975) * itt_se]

print('ITT estimate:', itt)
print('variance:', itt_var)
print('standard error:', itt_se)
print('95% confidence interval:', itt_ci95)
```

    ITT estimate: 0.18994413407821226
    variance: 0.0019767264386285955
    standard error: 0.04446039179571628
    95% confidence interval: [0.10280336742006825, 0.2770849007363563]


Notice that the ITT reflects a prescription of the treatment, but then the unit can actually take or not the treatment. The sets of units that take or not the treatment are different from the sets of units that are prescribed the treatment, which can result in different values for the contrast between observed outcomes.

<br />

# 3. Four Possible Strata

Thus, according to the instrument and the treatment values, we could define four possible strata in our setting as follows:

- **Compliers:** $T(Z = 1) = 1, T(Z = 0) = 0$: the patients take physiotherapy when they have the reduced fee, but not otherwise.

- **Defiers:** $T(Z = 1) = 0, T(Z = 0) = 1$: the patients take physiotherapy when the don’t have the reduced fee, but do not take it if they have the reduced fee.

- **Always-takers:** $T(Z = 1) = 1, T(Z = 0) = 1$: the patients take the physiotherapy whether or not they have the reduced fee or not. They could be very wealthy patients, or partients that value their health a lot.

- **Never-takers:** $T(Z = 1) = 0, T(Z = 0) = 0$: the patients do not take the physiotherapy whether or not they have the reduced fee or not. It could be patients with financial difficulties, or that do not really trust medicine.

In the following analysis, we will focus on estimating the Local Average Treatment Effect (LATE) among the **Compliers** strata, which is also called the Complier Average Causal Effect (CACE).

<br />

# 4. Checking Assumptions

However, in order to non-parametrically identify the LATE, we need to check the following assumptions for treatment assignment $Z$ to be an instrument.

## 4.1 Relevance

$$ \forall i, E[T_i(1)] \neq E[T_i(0)] $$

Treatment assignment $Z$ has a causal effect on the potential treatment $T$. In this setting, it seems likely that the decision to take physiotherapy is influenced by the price. In specific, we find the correlation between $Z$ and $T^{obs}$ is 0.55.


```python
df.corr()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Z</th>
      <th>T_obs</th>
      <th>Y_obs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Z</th>
      <td>1.000000</td>
      <td>0.547490</td>
      <td>0.179246</td>
    </tr>
    <tr>
      <th>T_obs</th>
      <td>0.547490</td>
      <td>1.000000</td>
      <td>0.378034</td>
    </tr>
    <tr>
      <th>Y_obs</th>
      <td>0.179246</td>
      <td>0.378034</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>



## 4.2 Exclusion Restriction

$$ \forall i, \forall t, Y_i(Z_i = 1, T_i = t) = Y_i(Z_i = 0, T_i = t) $$

The only effect of treatment assignment $Z$ on the potential outcome $Y$ is through potential treatment $T$. It seems also plausible that the price of the therapy just influences the decision to take therapy or not, but has no further effect on the patient’s recovery.

More specifically, notice that in the experiment below, when we are controlling for the potential treatment $T$, the causal effect of the treatment assignment $Z$ on the outcome $Y$ in both treated and control groups are no longer signifficant. This has shown that the causal effect of the treatment assignment $Z$ on the outcome $Y$ is fully mediated by the potential treatment $T$. Thus, the exclusion restriction assumption is plausible here!


```python
est_t = df[(df.Z == 1) & (df.T_obs == 1)].Y_obs.mean() \
          - df[(df.Z == 0) & (df.T_obs == 1)].Y_obs.mean()
est_c = df[(df.Z == 1) & (df.T_obs == 0)].Y_obs.mean() \
          - df[(df.Z == 0) & (df.T_obs == 0)].Y_obs.mean()

var_t = df[(df.Z == 1) & (df.T_obs == 1)].Y_obs.var() \
            / len(df[(df.Z == 1) & (df.T_obs == 1)]) \
          + df[(df.Z == 0) & (df.T_obs == 1)].Y_obs.var() \
            / len(df[(df.Z == 0) & (df.T_obs == 1)])
var_c = df[(df.Z == 1) & (df.T_obs == 1)].Y_obs.var() \
            / len(df[(df.Z == 1) & (df.T_obs == 1)]) \
          + df[(df.Z == 0) & (df.T_obs == 1)].Y_obs.var() \
            / len(df[(df.Z == 0) & (df.T_obs == 1)])

se_t = np.sqrt(var_t)
se_c = np.sqrt(var_c)

ci95_t = [est_t - sps.norm.ppf(.975) * se_t, est_t + sps.norm.ppf(.975) * se_t]
ci95_c = [est_c - sps.norm.ppf(.975) * se_c, est_c + sps.norm.ppf(.975) * se_c]

print('Difference-in-means among treated:', est_t)
print('var:', var_t, 'se:', se_t)
print('95% confidence interval:', ci95_t)
print()
print('Difference-in-means among control:', est_c)
print('var:', var_c, 'se:', se_c)
print('95% confidence interval:', ci95_c)
```

    Difference-in-means among treated: -0.03311475409836062
    var: 0.004398169325099314 se: 0.06631869514020397
    95% confidence interval: [-0.16309700807485192, 0.09686749987813068]
    
    Difference-in-means among control: -0.048473456368193224
    var: 0.004398169325099314 se: 0.06631869514020397
    95% confidence interval: [-0.17845571034468452, 0.08150879760829807]


## 4.3 Instrumental Unconfoundeness

$$ \forall i, Z_i \perp Y_i(0), Y_i(1), T_i(0), T_i(1) $$

No hidden confounder of the treatment assignment $Z$ and the outcome. This is plausible in our setting because the treatment assignment is randomized.

## 4.4 Monotonicity

$$\forall i, T_i(Z = 1) \geq T_i(Z = 0)$$

The treatment assignment $Z$ does not dissuade to get the potential treatment $T$. Here, we can think that a patient is not less likely to take the therapy because it is cheaper.

<br />

# 5. Local Average Treatment Effect (LATE)

Under assumptions discussed previously, we could estimate the LATE among the **Compliers** strata with the two stage least square estimator. Using the `IV2SLS` function in the `linearmodels` package, we also provide the standard error and a 95% confidence interval of the estimate.



```python
!pip install linearmodels
import linearmodels as lm
from linearmodels.iv import IV2SLS
```


```python
df = df.assign(const = 1)
IV2SLS(dependent = df.Y_obs,
       endog = df.T_obs,
       exog = df.const,
       instruments = df.Z
      ).fit(cov_type = "unadjusted")
```




<table class="simpletable">
<caption>IV-2SLS Estimation Summary</caption>
<tr>
  <th>Dep. Variable:</th>          <td>Y_obs</td>      <th>  R-squared:         </th> <td>0.1403</td> 
</tr>
<tr>
  <th>Estimator:</th>             <td>IV-2SLS</td>     <th>  Adj. R-squared:    </th> <td>0.1387</td> 
</tr>
<tr>
  <th>No. Observations:</th>        <td>537</td>       <th>  F-statistic:       </th> <td>20.070</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, Jan 30 2023</td> <th>  P-value (F-stat)   </th> <td>0.0000</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>21:26:13</td>     <th>  Distribution:      </th> <td>chi2(1)</td>
</tr>
<tr>
  <th>Cov. Estimator:</th>      <td>unadjusted</td>    <th>                     </th>    <td></td>    
</tr>
<tr>
  <th></th>                          <td></td>         <th>                     </th>    <td></td>    
</tr>
</table>
<table class="simpletable">
<caption>Parameter Estimates</caption>
<tr>
    <td></td>    <th>Parameter</th> <th>Std. Err.</th> <th>T-stat</th> <th>P-value</th> <th>Lower CI</th> <th>Upper CI</th>
</tr>
<tr>
  <th>const</th>  <td>0.4091</td>    <td>0.0321</td>   <td>12.764</td> <td>0.0000</td>   <td>0.3463</td>   <td>0.4720</td> 
</tr>
<tr>
  <th>T_obs</th>  <td>0.3505</td>    <td>0.0782</td>   <td>4.4800</td> <td>0.0000</td>   <td>0.1972</td>   <td>0.5039</td> 
</tr>
</table><br/><br/>Endogenous: T_obs<br/>Instruments: Z<br/>Unadjusted Covariance (Homoskedastic)<br/>Debiased: False<br/>id: 0x7fe22c75a1c0


<br />


# 6. Results

According to the results above, we've obtained a LATE estimate of 0.3505 with standard error 0.0782 and a 95% confidence interval from 0.1972 to 0.5039.

Notice that since our confidence interval does not include 0, and the p-value obtained is very small, we could conclude a positive causal effect of the treatment(taking physiotherapy) on the outcome(patients' satisfaction of the recovery 3 months after surgery) signifficant at level $\alpha = 0.05$.

<br />

# 7. Discussion

From a clinical viewpoint, the LATE estimation, found statistically significant indicates that physiotherapy is efficient to improve recovery among compliers. The ITT tells a slightly different story: the fee reduction to incentivize the patient adherence to the physiotherapy treatment enhances the recovery quality. Overall, encouraging patients to attend physiotherapy, through reduced fee (efficient) or other ways (to test) seems like a promising health policy. Improved recovery could even save future health costs.

In specific, the significant causal effect of the treatment(taking physiotherapy) on the outcome(patients' satisfaction of the recovery 3 months after surgery) further proven that taking physiotherapy could effectively help rehabilitation and would result in a more complete recovery. 
This finding has important implications on the clinical aspect as we've shown the significance beneficial effect of the physiotherapy on patiens' recovery. 

Moreover, our estimate of the intention-to-treat (ITT) effect of offering the discount on the improvement of recovery is significant at $\alpha = 0.05$ level as well, with an estimate of ITT at around 0.19 and a standard error at about 0.04. 
This implies that the encouragement of offering discount to the patients would effectively improve the recovery of the patients overall. 
Thus, from the policy perspective, in order to help more patients to get better rehabilation and more complete recovery, we could consider to make policy on reducing the cost of the physiotherapy, as we have shown that this encouragement is actually effective.

Besides, notice that under the four assumptions we've discussed, the LATE we've estimated is in fact the ITT effect we've estimated devided by the probability of compliance. 
Thus, by simple manipulation, we could estimate the proportion of the compliers in the entire sample, i.e. the probability of compliance, as $P(c) = \frac{ITT}{LATE} = \frac{0.1899}{0.3505} \approx 0.5418 $. 
This means that the compliers strata has about 54.18% population in the entire sample according to our estimation. 
In other words, there's a possibility of 54.18% that a given patients within our sample to become a complier of our encouragement(whether been offered a discount for the physiotherapy or not). 
Under all four assumptions we've discussed, this possibility of a given patient being a complier could be also interpret as the average effect of our instrument(offered a discount) on the treatment(taking the physiotherapy), which represents the effectiveness of our encouragement in this setting. 
Therefore, from a policy point of view, we want to make policy to help more patients to get better rehabilation and more complete recovery, while we also have to consider the effectiveness of the policy, as each policy would cause some amount of resources. 
Thus, this representation of the effectiveness on our encouragement provides us a way to evaluate certain policy, so that we could eventually make the policy that optimize the effectiveness under certain constratins on the resources.

<br />
<br />

# 8. Reference

- J. Abécassis. (2022). DS-UA 9201 Causal Inference, Fall 2022. NYU Brightspace. https://brightspace.nyu.edu/d2l/home/225088

