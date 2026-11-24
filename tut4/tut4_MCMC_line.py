import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm

xs = np.array([0.1,0.3,0.4,3.2,4.1,5.1,5.3])
a_true = 1.2
b_true = 0.2

def line_model(a,b,x):
    return a*x+b

ys_true = line_model(a_true,b_true,xs)
sigma_true = 0.1
ys_obs = ys_true + sigma_true*np.random.normal(size=np.size(xs))

# plt.plot(xs,ys_obs,linestyle='',marker='o',color='blue',label=r'$y_{\mathrm{obs}}$')
# plt.xlabel("x")
# plt.ylabel("x")
# plt.legend()
# plt.show()

# a prior is Gaussian with \mu_{a,0}, \sigma_{a,0}^2
# b prior is Gaussian with \mu_{b,0}, \sigma_{b,0}^2

# how do I sample a posterior distribution with these priors, 
# using the simplest Metropolis-Hasting (or any simple, first principles MCMC method, no packages)

# this fixes our priors, in principle we can change them and it affects the performance of the MCMC
mu_a = 2.5
sigma_a = 0.5

mu_b = 0.8
sigma_b = 0.2

# plot priors
as_dis = np.linspace(mu_a - 4*sigma_a, mu_a + 4*sigma_a, 500)
bs_dis = np.linspace(mu_b - 4*sigma_b, mu_b + 4*sigma_b, 500)

# plt.plot(as_dis, norm.pdf(as_dis, mu_a, sigma_a), label=r"$\mathrm{Prior}(a)$")
# plt.xlabel(r"$a$")
# plt.ylabel("Density")
# plt.legend()
# plt.show()

# simple MCMC chain
def log_prior(a, b):
    lp_a = -0.5*((a-mu_a)/sigma_a)**2
    lp_b = -0.5*((b-mu_b)/sigma_b)**2
    return lp_a + lp_b

sigma = sigma_true      # assume known

def log_likelihood(a, b):

    model = line_model(a, b, xs)

    return -0.5*np.sum(((ys_obs-model)/sigma)**2)

def log_posterior(a, b):

    return log_prior(a,b) + log_likelihood(a,b)

# starting the chain
a_c = 0.0
b_c = 0.0

samples = []

proposal_width = 0.1

nsteps = 10000

current_logpost = log_posterior(a_c,b_c)

for i in range(nsteps):

    # propose new point
    a_prop = a_c + proposal_width*np.random.randn()
    b_prop = b_c + proposal_width*np.random.randn()

    proposal_logpost = log_posterior(a_prop,b_prop)

    log_alpha = proposal_logpost - current_logpost

    if np.log(np.random.rand()) < log_alpha:

        a_c = a_prop
        b_c = b_prop
        current_logpost = proposal_logpost

    samples.append([a_c,b_c])

samples = np.array(samples)

# we forget about the first few parts of the chain 
# because the initial condition might have given us a bad start too far away from the posterior
burn = 5000

# this throws away #burn amount of initial samples
samples = samples[burn:]

plt.figure(figsize=(10,4))

plt.subplot(121)
plt.hist(samples[:,0], bins=10, density=True, label=r"$\mathrm{Posterior}(a)$")
plt.plot(as_dis, norm.pdf(as_dis, mu_a, sigma_a), label=r"$\mathrm{Prior}(a)$")
plt.axvline(a_true,color='r', label=r'$a_{\mathrm{true}}$')
plt.legend()
plt.xlabel("a")

plt.subplot(122)
plt.hist(samples[:,1], bins=10, density=True, label=r"$\mathrm{Posterior}(b)$")
plt.plot(bs_dis, norm.pdf(bs_dis, mu_b, sigma_b), label=r"$\mathrm{Prior}(b)$")

plt.axvline(b_true,color='r', label=r'$b_{\mathrm{true}}$')
plt.xlabel("b")
plt.legend()
plt.show()

from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

# Fixed limits
a_xlim = (as_dis.min(), as_dis.max())
b_xlim = (bs_dis.min(), bs_dis.max())

a_ylim = (0, 3.0*norm.pdf(mu_a, mu_a, sigma_a))
b_ylim = (0, 3.0*norm.pdf(mu_b, mu_b, sigma_b))

fig, ax = plt.subplots(1, 2, figsize=(10,4))
binnum = 20

def update(frame):

    ax[0].clear()
    ax[1].clear()

    s = samples[:frame]

    ax[0].hist(s[:,0], bins=binnum, density=True,
               label="Posterior")
    ax[0].plot(as_dis, norm.pdf(as_dis, mu_a, sigma_a),
               label="Prior")
    ax[0].axvline(a_true,color='r', label=r'$a_{\mathrm{true}}$')
    ax[0].set_xlabel("a")
    ax[0].set_ylabel("Density")
    ax[0].set_title(f"{frame} samples")
    ax[0].set_xlim(a_xlim)
    ax[0].set_ylim(a_ylim)
    ax[0].legend()

    ax[1].hist(s[:,1], bins=binnum, density=True,
               label="Posterior")
    ax[1].plot(bs_dis, norm.pdf(bs_dis, mu_b, sigma_b),
               label="Prior")
    ax[1].axvline(b_true,color='r', label=r'$b_{\mathrm{true}}$')
    ax[1].set_xlabel("b")
    ax[1].set_ylabel("Density")
    ax[1].set_xlim(b_xlim)
    ax[1].set_ylim(b_ylim)
    ax[1].legend()

anim = FuncAnimation(
    fig,
    update,
    frames=range(10, len(samples), 20),
    interval=50,
)

writer = FFMpegWriter(fps=20, bitrate=1800)
anim.save("posterior_animation.mp4", writer=writer)

# plt.show()
