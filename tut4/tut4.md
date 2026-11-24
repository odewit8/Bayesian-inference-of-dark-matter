# Tutorial 4
## Bayesian straight line fitting
See these lecture notes [here](https://arxiv.org/pdf/1008.4686).

We see that for standard least-squares straight line fitting, if we have a lot of data points, then the standard uncertainty does not accurately reflect the case of outliers (Exercise 1 and 2 in the lecture notes).

What that example says is that there is a small region in the $(a,b)$, a circle with center the optimal $(a,b)$ and with radius $\alpha\times(\sigma_a,\sigma_b)$ (via Pythagorean adding of squares), such that in that region we say the fit is accurate with some confidence $\alpha$. But this conflicts with the outliers.

In the Bayesian setting, we make more assumptions on what kind of fit is a good fit. Moreover, Bayesian integrals can be estimated using Monte Carlo Markov Chain methods so that the posterior remains computable. 

In this case, MCMC, for example for the Bayesian fit, starts with a proposal
$$(a_1,b_1), \dots,(a_n,b_n)$$

and accepts a new $(a_{n+1},b_{n+1})$ based on the Bayesian rule and the priors and likelihood (the way this is determined varies across different MCMC methods, in the simple example we use the simple Metropolis-Hastings algorithm), so that the collection 
$$\{(a_n,b_n)\}_{n=1}^N$$
approximates the posterior distribution.

We consider data points $(x_i,y_i)$. We have a straight line model of the data $y=ax+b$. We want to find a probability distribution for how $a$ and $b$ are most likely distributed given the data.
Bayes' rule which says that for a given pair $(\hat{a},\hat{b})$ what the probability is of that pair being the true value,
$$\mathbb{P}\left((a,b)=(\hat{a},\hat{b})\Big|\mathrm{data}\right)=\frac{\mathbb{P}\left(\mathrm{data}\Big|(a,b)=(\hat{a},\hat{b})\right)\mathbb{P}((a,b)=(\hat{a},\hat{b}))}{\mathbb{P}(\mathrm{data})}.$$
This can also be thought of as 
$$\mathrm{posterior}=\mathrm{likelihood}\times\mathrm{prior}.$$

We want to estimate the errors of $a$ and $b$, given that there might be some (identically distributed) noise for each data point.
$$y_i=ax_i+b+\varepsilon_i, \quad \varepsilon\sim\mathcal{N}(0,\sigma^2).$$


Suppose we take a Gaussian modelling assumption. This means that, for fixing $a,b,\sigma$, that finding a point $(x_i,y_i)$ has probability, rewriting $\varepsilon_i=y_i-ax_i-b$,
$$\mathbb{P}\left((x,y)=(x_i,y_i)\Big|a,b,\sigma\right)=\frac{1}{(2\pi \sigma)^{1/2}}\exp\left(-\frac{1}{2\sigma^2}(y_i-ax_i-b)^2\right),$$
using, for Gaussian variables,
$$\mathbb{P}(\varepsilon_i=u)=\frac{1}{(2\pi \sigma)^{1/2}}\exp\left(-\frac{1}{2\sigma^2}u^2\right).$$
Priors: we need to assume how $a,b$ and $\sigma$ might be distributed.

Using the posterior we obtain via MCMC we can plot confidence contours for $a$ and $b$ to visualise how confident we are for a given found value for $a$ and $b$.

## Example 1: $\sigma$ is fixed and known
We let $\sigma=0.05$ be fixed and known. We assume Gaussian priors for $a$ and $b$, so that they have
priors $\mathcal{N}(\mu_i,\sigma_i^2)$ for $i=a,b$.
## Exercise: change the prior of $a$ and $b$ and see what happens to get a converging chain.
You can change ```proposal_width```, ```nsteps```, ```a_c```, `b_c`, `burn`
## Exercise: try the same for a quadratic function
$y=ax^2 + bx +c$.

