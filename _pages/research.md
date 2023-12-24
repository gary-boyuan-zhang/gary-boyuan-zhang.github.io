---
layout: archive
title: "Research"
permalink: /research/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

<br />

My research mainly focuses on discovering and exploiting causality in observational data. In particular, I'm interested in: 
1. developing methods for causal discovery and inference in observational data,
2. empowering machine learning models with learning causal representation, and
3. applications in healthcare, public policy, social science, and sports.

<br />

Works in Progress
------

<br />

**Testing for Individual Causal Effects in Observational Data**
(with [Wesley Tansey](https://wesleytansey.com/))

Many modern scientific datasets are derived from exploratory or observational studies, where controlled experiments are technologically, financially, or ethically infeasible. Often these datasets only capture intention-to-treat, where a treatment was assigned but compliance to treatment assignment may be ignored. For example, electronic health records track which patients receive prescriptions, but not whether the patient actually took the medication. We first show that, without strong parametric assumptions, the causal effect of a treatment is not identifiable when the compliance status is unobserved. Despite being unidentifiable, it turns out that it is possible to develop a nonparametric compliance testing procedure for each sample in the treatment-assigned population. We then turn to the finite-sample scenario, where we wish to test for individual causal effects with minimal modeling assumptions. We extend a recent empirical Bayes procedure for multiple hypothesis testing to the causal setting with heterogeneous treatment effects under an additive noise model. Finally, we propose a generalized machine learning approach, where flexible black box models are trained with a regularizer that constrains the posterior expectation of discovery to be conservative. Both proposed models are shown to control the false discovery rate (FDR) at the target level in well-specified synthetic simulations. On data from a large cancer drug study, models from the causal inference and multiple testing literature fail to control the target FDR, whereas the proposed generalized causal testing procedure succeeds.

<br />

**Automated Discovery of Difference-in-Differences**
(with [William Herlands](https://www.williamherlands.com/), [Edward McFowland](https://www.hbs.edu/faculty/Pages/profile.aspx?facId=772797), and [Daniel Neill](https://cs.nyu.edu/~neill/))

Difference-in-Differences (DDs) are a popular framework for causal inference in
temporal observational data. However, identifying DDs is a manual and serendipitous process. We develop the first automated DD discovery system to automatically
identify and validate DDs. Our two-stage approach identifies both heterogeneous
treatment discontinuities in arbitrary-dimensional categorical data and appropriate
control subsets. The system includes a rigorous testing procedure to ensure statistical and econometric validity. We demonstrate robust performance across a wide
set of synthetic, semi-synthetic, and real-world data, including two complex public
policy datasets where our approach provides novel policy-relevant insights.

<br />

**A Regularized Adjusted Expected Goal Plus-Minus Model for Soccer**
(with [Edvin Tran Hoac](https://www.edvintranhoac.com/) and [Phong Hoang](https://medium.com/@IwriteDSblog))

Evaluating the impact of individual players on their team’s performance is an important question to consider when analyzing team sports. Variants of the Adjusted Plus-Minus (APM) model have been viewed as all-in-one player’s performance evaluation matrices, and have been widely utilized in basketball and hockey. However, because of the low number of substitutions and scoring chances in soccer, the APM model has not been shown to be effective in identifying players' performance. In this work, we introduce a new kind of Regularized Adjusted Plus-Minus (RAPM) model, which incorporates priors generated from box score statistics into a regularized regression framework, performing point estimation on the player’s contribution to the expected goals per 90 minutes. In particular, using data from the 2021-2022 season of the English Premier League, we show that our RAPM model with box score prior has better predictability and interpretability than the APM model, RAPM model without priors, and RAPM model with FIFA ratings as prior. This model could be further utilized to evaluate the impact of player transfer, simulate teams' performance, and forecast players' market value.

<!---

{% for post in site.publications reversed %}
  {% include archive-single.html %}
 {% endfor %}

-->
