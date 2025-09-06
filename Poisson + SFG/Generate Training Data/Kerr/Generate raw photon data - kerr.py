#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('reset', '')


# In[ ]:


import sys

sys.path.insert(0, '/home/users/ids29/DGRB')


# In[ ]:


import aegis
import numpy as np
import torch
import healpy as hp
import pickle as pk
from astropy import units
from astropy import constants as c
import matplotlib.pyplot as plt
from os import listdir
import os
from sbi.inference import SNLE, SNPE#, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
from sbi import analysis as analysis
# from sbi.inference.base import infer
from getdist import plots, MCSamples
from joblib import Parallel, delayed, parallel_backend
from scipy.integrate import quad, simpson
import pickle
from scipy.stats import norm

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


grains=1000
num_simulations = 240
num_workers = 48


# In[ ]:


parameter_range_aegis = [[], []]
abundance_luminosity_and_spectrum_list = []
source_class_list = []
parameter_names = []
energy_range = [1000, 100000] #MeV 
energy_range_gen = [energy_range[0]*0.5, energy_range[1]*18]
max_radius = 8.5 + 20*2 #kpc
exposure = 2000*10*0.2 #cm^2 yr
flux_cut = 1e-9 #photons/cm^2/s
angular_cut = np.pi #10*u.deg.to('rad') #degrees
angular_cut_gen = np.pi #angular_cut*1.5
lat_cut = 0 #2*u.deg.to('rad') #degrees
lat_cut_gen = lat_cut*0.5


# In[ ]:


my_cosmology = 'Planck18'
z_range = [0, 14]
luminosity_range = 10.0**np.array([37, 50]) # Minimum value set by considering Andromeda distance using Fermi as benchmark and receiving 0.1 photon at detector side
my_AEGIS = aegis.aegis(abundance_luminosity_and_spectrum_list, source_class_list, parameter_range_aegis, energy_range, luminosity_range, max_radius, exposure, angular_cut, lat_cut, flux_cut, energy_range_gen=energy_range_gen, cosmology = my_cosmology, z_range = z_range, verbose = False)
my_AEGIS.angular_cut_gen, my_AEGIS.lat_cut_gen = angular_cut_gen, lat_cut_gen


# In[ ]:


def spec_poisson(energy, params):

    Phi_Poisson =  params[0] 
    
    Gamma = 2.2
    Emin, Emax = energy_range[0], energy_range[1]
    exposure_det = exposure*units.yr.to('s') # cm^2 s
    expected_photons = 772_340 # Value such that it produces the twice number of photons (after mock_observe is applied) as detected by Fermi-LAT, under the condition that Phi_Poisson = 2
    num_photons_exposure_solidAngle = expected_photons / exposure_det / (4*np.pi) # photons/cm^2/sec/sr
    normalization =  (Emax**(1-Gamma) - Emin**(1-Gamma)) / (1-Gamma) 
    prop_const = num_photons_exposure_solidAngle / normalization
    return Phi_Poisson * prop_const * energy**(-Gamma)


# In[ ]:


Gamma_SFG = 2.2
gamma_energy_bounds = energy_range_gen  # in MeV
E_photon_GeV_SFG = ((-Gamma_SFG + 1) / (-Gamma_SFG + 2) *
                (gamma_energy_bounds[1]**(-Gamma_SFG + 2) - gamma_energy_bounds[0]**(-Gamma_SFG + 2)) /
                (gamma_energy_bounds[1]**(-Gamma_SFG + 1) - gamma_energy_bounds[0]**(-Gamma_SFG + 1))) # in MeV
E_photon_SFG = E_photon_GeV_SFG * 1.60218e-6  # erg

res = int(1e4)
log_LIRs = np.linspace(-5, 25, res)


# In[ ]:


def ZL_SFG1(z, l, params):


    Phi_star = params[1]

    l_erg = l * E_photon_SFG # erg/s
    LFs = np.zeros_like(l)

    def Phi_IR(log_LIR): #log_LIR = log_10(L_IR / solar_luminosity) # unitless

        # from Table 8 in Gruppioni et al.
        # Phi_star = 10**(-2.08) # Mpc^{-3} dex^{-1}
        Lstar = 10**(9.46) # Solar luminosity
        alpha = 1.00
        sigma = 0.50

        LIR = 10**log_LIR # solar luminosity

        Phi_IR = Phi_star * (LIR / Lstar)**(1 - alpha) * np.exp(-1 / (2 * sigma**2) * (np.log10(1 + LIR / Lstar))**2) # from Gruppioni paper eqn (3)  	

        return Phi_IR

    def PDF_log_Lgamma_given_log_LIR(log_LIR, log_Lgamma): #log_LIR = log_10(L_IR / solar_luminosity) # unitless
        LIR_solar_luminosity = 10**log_LIR # Solar luminosity
        L_IR_erg_second = LIR_solar_luminosity * 3.826e33 # erg/s

        a = 1.09
        g = 40.8
        sigma_SF = 0.202 

        mean = g + a * np.log10(L_IR_erg_second / 1e45)
        std = sigma_SF

        return norm.pdf(log_Lgamma, loc=mean, scale=std)

    def integrand(PhiIR_of_logLIRs, log_LIRs, log_Lgamma):
        return PhiIR_of_logLIRs * PDF_log_Lgamma_given_log_LIR(log_LIRs, log_Lgamma)

    PhiIR_of_logLIRs = Phi_IR(log_LIRs)

    for i in range(LFs.shape[0]):
        for j in range(LFs.shape[1]):
            LFs[i,j] = simpson(integrand(PhiIR_of_logLIRs, log_LIRs, np.log10(l_erg[i,j])), x=log_LIRs)
    return 1e-9 / np.log(10) / l * LFs # LF has spatial units of Mpc^{-3}. We need to convert this to kpc^{-3}. Hence the factor of 1e-9


def spec_SFG1(energy, params):
    Gamma = 2.2
    return energy**(-Gamma)


# In[ ]:


als_Poisson = [spec_poisson]
als_SFG1 = [ZL_SFG1, spec_SFG1]
my_AEGIS.abun_lum_spec = [als_Poisson, als_SFG1]
my_AEGIS.source_class_list = ['isotropic_diffuse', 'extragalactic_isotropic_faint_single_spectrum']


# In[ ]:


# a simple simulator with the total number of photons as the summary statistic
def simulator(params):

    input_params = params.numpy()

    source_info = my_AEGIS.create_sources(input_params, grains=grains, epsilon=1e-2)
    photon_info = my_AEGIS.generate_photons_from_sources(input_params, source_info, grains=grains) 
    obs_info = {'psf_fits_path': '/home/users/ids29/DGRB/FERMI_files/psf_P8R3_ULTRACLEANVETO_V2_PSF.fits', 'edisp_fits_path': '/home/users/ids29/DGRB/FERMI_files/edisp_P8R3_ULTRACLEANVETO_V2_PSF.fits', 'event_type': 'PSF3', 'exposure_map': None}
    obs_photon_info = my_AEGIS.mock_observe(photon_info, obs_info)
    
    return obs_photon_info


# In[ ]:


def manual_simulate_for_sbi(proposal, num_simulations=1000, num_workers=32):
    """
    Simulates the model in parallel using joblib.
    Each simulation call samples a parameter from the proposal and passes the index to the simulator.
    """
    def run_simulation(i):
        if i % 10 == 0:
            print(f"i= {i}")
        # Sample a parameter from the proposal (sbi.utils.BoxUniform has a .sample() method)
        theta_i = proposal.sample()
        photon_info = simulator(theta_i)

        with open(f'train_data_kerr_{i+920}.pkl', 'wb') as f:
            pickle.dump(photon_info, f)

        torch.save(theta_i, f'train_thetas_kerr_{i+920}.pt')


        # return theta_i , photon_info

    # Run simulations in parallel using joblib.
    # Switch to the threading backend
    with parallel_backend('threading', n_jobs=num_workers):
        Parallel(verbose=5, timeout=None)(delayed(run_simulation)(i) for i in range(num_simulations))

    # theta_list = zip(*results)

    # theta_tensor = torch.stack(theta_list, dim=0).to(torch.float32)
    
    
    # return theta_tensor #, photon_info_list


# In[ ]:


# Define the prior using sbi.utils.BoxUniform
Phi_Poisson_training_range = [0, 1.0944]
Phi_SFG_training_range = [0, 0.1]

prior_range = torch.tensor([[Phi_Poisson_training_range[0], Phi_SFG_training_range[0]],
                            [Phi_Poisson_training_range[1], Phi_SFG_training_range[1]]])

prior = utils.BoxUniform(low=prior_range[0], high=prior_range[1])

manual_simulate_for_sbi(prior, num_simulations=num_simulations, num_workers=num_workers)


# In[ ]:


# # 'photon_info_list' is a list of dictionaries

# # Save to file
# with open('train_data_Poisson_SFG_kerr_336.pkl', 'wb') as f:
#     pickle.dump(train_photon_info, f)

# # Save to file
# torch.save(train_thetas, 'train_thetas_Poisson_SFG_kerr_336.pt')
# torch.save(prior_range, 'prior_range_Poisson_SFG_kerr_336.pt')

