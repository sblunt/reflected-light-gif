import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u, constants as cst
from astropy.io import fits
from scipy.stats import norm

from matplotlib.animation import FuncAnimation
from orbitize.kepler import calc_orbit


def calculate_lambertian_phase_function(alpha, sma, inc, aop):
    """ 
    Calculate phase function g(alpha) for an object using the Lambertian disk 
    approximation, assuming e=0.

    Args:
        alpha (float): the angular phase (related to true anomaly). At 
            opposition/full reflectance, alpha=pi [radians]
        sma (float): semimajor axis of orbit [au]
        inc (float): inclination of orbit [radians]
        aop (float): argument of periastron of the **STAR'S** orbit [radians].
            Currently not used, since I'm assuming e=0.

    Returns:
        float: the Lambert phase function value
    """

    true_anomaly = alpha + 3 * np.pi / 2 # [radians]

    z = (
        - np.sin(inc) * np.sin(true_anomaly)
    )
    orbit_radius = sma
    z *= orbit_radius # line-of-sight distance between star and planet [au]

    proj_sep = np.sqrt(orbit_radius**2 - z**2)

    # angle between your eyeball, the planet, and the host star
    out_beta = np.arctan2(-proj_sep, z) + np.pi  

    # Lambert phase function
    phase_function = (
        np.sin(out_beta) + (np.pi - out_beta) * np.cos(out_beta)
    ) / np.pi 

    return phase_function

solar_spect = fits.open('phoenix_spectrum.fits')
solar_spect = solar_spect[0].data

wavelen = fits.open('wwavelengths.fits')
wavelen = wavelen[0].data / 10 # [nm]

solar_intens = solar_spect
flux_ratio = 0.1
jup_intens = flux_ratio * solar_intens

# compute the RVs of planet & star
n_frames = 100
epochs = np.linspace(0, 11.5, n_frames)
sma = 0.1
m0 = 1
per = np.sqrt(sma**3 / m0) * 365.25

# alpha=pi at full reflectance. Max brightness occurs when moving from
# a negative RV towards 0 RV.
alphas = (epochs - epochs[0]) * 2 * np.pi / per - np.pi / 2 
_, _, rvs = calc_orbit(
    epochs, sma, 0, 0.5 * np.pi, 0, 0, 0, 100, m0, mass_for_Kamp=1
)

def animate(i):
    dlamb_pl = wavelen * rvs[i] / 3e5

    # interpolate solar spectrum onto new wavelengths
    interpolated_solar_intens = np.interp(
        wavelen + dlamb_pl, wavelen, solar_intens
    )

    phase = calculate_lambertian_phase_function(alphas[i], 0.1, 0.5 * np.pi, 0)

    line.set_data(
        wavelen + dlamb_pl, 
        phase * jup_intens + interpolated_solar_intens
    )

    rvs4ccf = np.arange(-100, 100, 0.1)
    pl_gaussian = norm(rvs[i], 5)
    line2.set_data(
        rvs4ccf, 
        (
            star_gaussian.pdf(rvs4ccf) + 
            flux_ratio * phase * pl_gaussian.pdf(rvs4ccf)
        )
    )


fig, ax  = plt.subplots(1, 2, dpi=250, figsize=(12,6))
ax[0].set_xlabel('$\\lambda$ [nm]')
ax[0].set_xlim(500,501)
ax[0].set_yticks([])
ax[0].set_ylabel('flux')

# plot fake CCF
rvs4ccf = np.arange(-100, 100, 0.1)
star_gaussian = norm(0, 10)
pl_gaussian = norm(rvs[0], 5)
line2, = ax[1].plot(
    rvs4ccf, star_gaussian.pdf(rvs4ccf) + flux_ratio * pl_gaussian.pdf(rvs4ccf), 
    color='k'
)
ax[1].set_xlabel('RV [km/s]')
ax[1].set_yticks([])
ax[1].set_ylabel('CCF')

line, = ax[0].plot(wavelen, solar_intens + jup_intens, color='k')
anim = FuncAnimation(fig, animate, frames=n_frames, interval=50)

anim.save('wavelen_shift.gif', writer='imagemagick')


