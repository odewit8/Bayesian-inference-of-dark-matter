## packages, some #commented-out
import numpy as np
import random
# import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import betainc
# import arviz
# import corner

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

## Constants
G = 4.301e-9 # gravitational constant, in Mpc km^2 s^-2 Msun^-1
H = 70.      # Hubble's constant, in km s^-1 Mpc^-1
Dc= 200.     # critical overdensity

# Posti's physical formulas in Jax replaced by Dr Yasin

# _, _, r, vobs, e_vobs, vg, vd, vb, _, _ = np.genfromtxt('tut5/data/NGC2403_rotmod.dat', unpack=True)
# _, _, r, vobs, e_vobs, vg, vd, vb, _, _ = np.genfromtxt('tut5/data/DD0161_rotmod.dat', unpack=True)
_, _, r, vobs, e_vobs, vg, vd, vb, _, _ = np.genfromtxt('tut5/data/NGC2366_rotmod.dat', unpack=True)

# note here that r is not really the first column as in the SPARC set, Posti modified it first...

plt.errorbar(
    r, vobs,
    yerr=e_vobs,
    fmt='o',
    color='black',
    markersize=3,
    capsize=2,
    label='V_obs'
)
plt.legend()
plt.xlabel("r")
plt.ylabel("v")
plt.show()

# Dr Yasin's code
def fc(x): 
    """Accessory function for an NFW halo.""" 
    return jnp.log1p(x) - x / (1.0 + x)

def vvir(m_halo): 
    """Virial velocity for a halo of mass m_halo [Msun].""" 
    return jnp.sqrt((Dc * H**2 / 2.0) ** (1.0 / 3.0) * (G * m_halo) ** (2.0 / 3.0))

def rvir(m_halo): 
    """Virial radius [kpc] for a halo of mass m_halo [Msun].""" 
    rho_hat = (4.0 / 3.0) * jnp.pi * Dc * (3.0 * H**2 / (8.0 * jnp.pi * G)) 
    return 1.0e3 * (m_halo / rho_hat) ** (1.0 / 3.0)

def vhalo(params, radius): 
    """NFW dark-matter circular velocity [km/s].""" 
    m_halo = 10.0 ** params["log_mh"] 
    concentration = 10.0 ** params["log_c"] 
    rv = rvir(m_halo)
    return jnp.sqrt(
        vvir(m_halo) ** 2
        * rv / radius
        * fc(concentration * radius / rv)
        / fc(concentration)
    )

def dc14_alpha_beta_gamma(X):
    """DC14 alpha, beta, gamma as functions of log10(Mstar/Mhalo)."""

    alpha = (
        2.94
        - jnp.log10(
            (10.0 ** (X + 2.33)) ** (-1.08)
            + (10.0 ** (X + 2.33)) ** (2.29)
        )
    )

    beta = 4.23 + 1.34 * X + 0.26 * X**2

    gamma = (
        -0.06
        + jnp.log10(
            (10.0 ** (X + 2.56)) ** (-0.68)
            + 10.0 ** (X + 2.56)
        )
    )

    return alpha, beta, gamma

def dc14_mass_fraction(radius, rvir, rs, alpha, beta, gamma):
    """M(<r) / Mvir for a DC14 halo."""

    x = radius / rs
    xc = rvir / rs

    a = (3.0 - gamma) / alpha
    b = (beta - 3.0) / alpha

    z = x**alpha / (1.0 + x**alpha)
    zc = xc**alpha / (1.0 + xc**alpha)

    return betainc(a, b, z) / betainc(a, b, zc)

def vhalo_dc14(params, radius):
    """DC14 dark-matter circular velocity [km/s]."""

    m_halo = 10.0 ** params["log_mh"]
    m_star = 10.0 ** params["log_mstar"]

    # Stellar-to-halo mass ratio
    X = jnp.log10(m_star / m_halo)

    # DC14 shape parameters
    alpha, beta, gamma = dc14_alpha_beta_gamma(X)

    # Halo virial quantities
    rv = rvir(m_halo)
    vv = vvir(m_halo)

    # concentration
    concentration = 10.0 ** params["log_c"]

    # IMPORTANT:
    # Here concentration is interpreted as c = rvir / r_-2.
    r_minus2 = rv / concentration

    # Convert r_-2 to the DC14 scale radius
    rs = r_minus2 / (
        ((2.0 - gamma) / (beta - 2.0)) ** (1.0 / alpha)
    )

    # Enclosed mass fraction
    mass_fraction = dc14_mass_fraction(
        radius,
        rv,
        rs,
        alpha,
        beta,
        gamma,
    )

    return jnp.sqrt(
        vv**2
        * rv / radius
        * mass_fraction
    )

def vmodel(params, radius): 
    """ Total circular velocity: V^2 = V_DM^2 + V_gas^2 + (M/L)_disk V_disk^2 + (M/L)_bulge V_bulge^2 """ 
    baryon_v2 = jnp.interp( radius, params["r"], params["vg"] ** 2 
                           + 10.0 ** params["log_mld"] * params["vd"] ** 2 
                           + 10.0 ** params["log_mlb"] * params["vb"] ** 2, )

    return jnp.sqrt(vhalo(params, radius) ** 2 + baryon_v2)

def vmodel_dc14(params, radius):
    """
    Total circular velocity for DC14 halo.

    V^2 = V_DM^2 + V_gas^2
          + (M/L)_disk V_disk^2
          + (M/L)_bulge V_bulge^2
    """

    baryon_v2 = jnp.interp(
        radius,
        params["r"],
        (
            params["vg"]**2
            + 10.0**params["log_mld"] * params["vd"]**2
            + 10.0**params["log_mlb"] * params["vb"]**2
        ),
    )

    return jnp.sqrt(
        vhalo_dc14(params, radius)**2
        + baryon_v2
    )


def rotation_curve_model(radius, velocity_error, velocity_obs, model_data): 
    """Bayesian rotation-curve decomposition with independent Normal errors.""" 
    params = dict(model_data)

    params["log_mh"] = numpyro.sample("log_mh",dist.Uniform(8.0, 14.0),)

    params["log_c"] = numpyro.sample("log_c",
        dist.Normal(0.905 - 0.101 * (params["log_mh"] * 0.7 - 12.0),0.11,),)

    params["log_mld"] = numpyro.sample(
        "log_mld",
        dist.Normal(-0.3, 0.2),
    )

    params["log_mlb"] = numpyro.sample(
        "log_mlb",
        dist.Normal(-0.15, 0.2),
    )

    params["log_mstar"] = numpyro.sample(
    "log_mstar",
    dist.Uniform(7.0, 11.5)
    )   

    model_velocity = vmodel(params, radius)

    numpyro.sample(
        "y",
        dist.Normal(model_velocity, velocity_error),
        obs=velocity_obs,
    )

    numpyro.deterministic(
        "mu",
        vmodel(params, params["r_grid"]),
    )

def rotation_curve_model_dc14(
    radius,
    velocity_error,
    velocity_obs,
    model_data,
):
    """
    Bayesian rotation-curve decomposition using a DC14 halo.
    """

    params = dict(model_data)

    # Halo mass
    params["log_mh"] = numpyro.sample(
        "log_mh",
        dist.Uniform(8.0, 14.0),
    )

    # Concentration
    params["log_c"] = numpyro.sample(
        "log_c",
        dist.Normal(
            0.905 - 0.101 * (params["log_mh"] * 0.7 - 12.0),
            0.11,
        ),
    )

    # Stellar disk M/L
    params["log_mld"] = numpyro.sample(
        "log_mld",
        dist.Normal(-0.3, 0.2),
    )

    params["log_mlb"] = numpyro.sample(
            "log_mlb",
            dist.Normal(-0.15, 0.2),
        )
    

    # Explicit stellar mass parameter
    params["log_mstar"] = numpyro.sample(
        "log_mstar",
        dist.Uniform(7.0, 12.0),
    )


    # Model velocity
    model_velocity = vmodel_dc14(
        params,
        radius,
    )

    # Likelihood
    numpyro.sample(
        "y",
        dist.Normal(
            model_velocity,
            velocity_error,
        ),
        obs=velocity_obs,
    )

    # Smooth posterior-predictive curve
    numpyro.deterministic(
        "mu",
        vmodel_dc14(
            params,
            params["r_grid"],
        ),
    )


model_data = {
    "r": jnp.array(r),
    "vg": jnp.array(vg),
    "vd": jnp.array(vd),
    "vb": jnp.array(vb),
    "r_grid": jnp.linspace(
        r.min(),
        r.max(),
        200,
    ),
}

radius = jnp.array(r)
velocity_obs = jnp.array(vobs)
velocity_error = jnp.array(e_vobs)

radius = jnp.array(r)
velocity_obs = jnp.array(vobs)
velocity_error = jnp.array(e_vobs)

### Running the MCMC chain
import jax
from numpyro.infer import MCMC, NUTS

kernel = NUTS(rotation_curve_model_dc14)

mcmc = MCMC(
    kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    progress_bar=True,
)

mcmc.run(
    jax.random.PRNGKey(0),
    radius,
    velocity_error,
    velocity_obs,
    model_data,
)

mcmc.print_summary()

### Plotting
samples = mcmc.get_samples()

# Posteriors
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

parameters = [
    ("log_mh", r"$\log M_h$"),
    ("log_c", r"$\log c$"),
    ("log_mld", r"$\log \Upsilon_{\rm disk}$"),
    ("log_mlb", r"$\log \Upsilon_{\rm bulge}$"),
]

for ax, (name, label) in zip(axes.flat, parameters):
    ax.hist(samples[name], bins=50, density=True)
    ax.set_xlabel(label, fontsize=14)
    ax.set_ylabel("Density", fontsize=12)

plt.tight_layout()
plt.show()

## Correlations
import corner

data1 = np.column_stack([
    np.asarray(samples["log_mh"]),
    np.asarray(samples["log_c"]),
    np.asarray(samples["log_mld"]),
    np.asarray(samples["log_mlb"]),
])

print(data1.shape)

fig = corner.corner(
    data1,
    labels=[
        r"$\log M_h$",
        r"$\log c$",
        r"$\log \Upsilon_{\rm disk}$",
        r"$\log \Upsilon_{\rm bulge}$",
    ],
    show_titles=True,
)

plt.show()

### Confidence velocity curves

# Smooth radius grid for plotting
r_grid = np.linspace(r.min(), r.max(), 200)

# Number of posterior samples
n_samples = len(samples["log_mh"])

# Store model curves
velocity_samples = np.zeros((n_samples, len(r_grid)))

for i in range(n_samples):

    params_i = {
        "log_mh": samples["log_mh"][i],
        "log_c": samples["log_c"][i],
        "log_mld": samples["log_mld"][i],
        "log_mlb": samples["log_mlb"][i],
        "r": jnp.array(r),
        "vg": jnp.array(vg),
        "vd": jnp.array(vd),
        "vb": jnp.array(vb),
    }

    velocity_samples[i] = np.asarray(
        vmodel(params_i, r_grid)
    )

v16, v50, v84 = np.percentile(
    velocity_samples,
    [16, 50, 84],
    axis=0
)

v025, v975 = np.percentile(
    velocity_samples,
    [2.5, 97.5],
    axis=0
)

fig, ax = plt.subplots(figsize=(8, 5))

# 95% credible band
ax.fill_between(
    r_grid,
    v025,
    v975,
    alpha=0.15,
    label=r"$95\%$ credible interval"
)

# 68% credible band
ax.fill_between(
    r_grid,
    v16,
    v84,
    alpha=0.30,
    label=r"$68\%$ credible interval"
)

# Posterior median
ax.plot(
    r_grid,
    v50,
    lw=2,
    label=r"posterior median"
)

# Observed rotation curve
ax.errorbar(
    r,
    vobs,
    yerr=e_vobs,
    fmt=".",
    color="k",
    lw=0.5,
    label=r"data"
)

ax.set_xlabel(r"$\rm radius/kpc$", fontsize=18)
ax.set_ylabel(r"$\rm velocity/km\,s^{-1}$", fontsize=18)
# ax.set_title(f"{name1}", fontsize=20)
ax.legend(fontsize=12)
ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()