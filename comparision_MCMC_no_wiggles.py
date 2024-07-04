import camb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from getdist import *
import emcee
import getdist.plots as plots


# CONSTANTS
c = 299792.458  # Speed of light in km/s
kB = 1.380649e-23  # Boltzmann constant in m^2 kg s^-2 K^-1

# REDSHIFT VALUES
z_values = [0.65]  # Redshifts from 0 to 10, 10 points in total
fid_z = 0
z_values = np.array(z_values)

# Conversion factors
frequency_to_temperature = (c ** 2) / (2 * kB * 1e9)  # Convert km/s to GHz, then GHz to Kelvin

# Noise for the P_IM (from annex of https://arxiv.org/pdf/2210.05705.pdf) for SKAO IM survey

# sky fraction
f_sky = 0.48

# Observational time in hours
t_tot = 10000

# System temperature from https://arxiv.org/pdf/2009.06197.pdf to be changed
T_sys = 30

# Number of dishes, in the articles it was said N_d = 1, but in the annex N_d = 197
N_d = 197

# CALCULATIOM OF THE OBERSVED FREQUENCY OF HI EMISSION FOR DIFFERENT REDSHIFT VALUES
# Constants
nu_21cm_rest = 1420.40575  # Rest-frame frequency of the 21 cm line in MHz

# Calculate the observed frequency
nu_obs = nu_21cm_rest / (1 + z_values)
print("Observed frequency of the 21 cm line at z = 0.65:", nu_obs, "MHz")

# VALUES OF μ_Θ FROM THE ARTICLE https://arxiv.org/pdf/2210.05705.pdf
mi = 1

# values for the galaxy bias depending on the redshift
b_g = 1.5

# values from the HI bias from https://arxiv.org/pdf/2210.05705.pdf
b_HI = 0.3 * (1 + z_values) + 0.65
print(b_HI)

# Value for the error on the redshift from https://arxiv.org/pdf/2210.05705.pdf
sigma = 0.001

# Diameter of the telescope dish from https://arxiv.org/pdf/2210.05705.pdf
d = 15

# Wavelength of the observed redshift frequency
l = (1 + z_values) * 21 * 10 ** (-2)

# Full width at half maximum of the dish primary beam
theta_pb = 1.22 * l / d

#WE COMPUTE THE 'TRUE' POWER SPECTRUM, WHICH WILL ACT AS DATA IN OUR MCMC
# Define cosmological parameters
ombh2 = 0.0224  # Physical baryon density
omch2 = 0.17  # Physical cold dark matter density
H0 = 67  # Hubble constant in km/s/Mpc
As = 2.14e-9  # Amplitude of primordial scalaar perturbations
ns = 0.96  # Scalar spectral index
mnu = 0.06  # Sum of neutrino masses in eV
omk = 0  # Curvature parameter, set to 0 for flat universe


survey_area = 10000  # Square degrees
redshift = 0.65
#number_density = 0.01  # number density in galaxies per cubic Mpc

# Convert survey area to steradians
survey_area_sr = np.deg2rad(survey_area) ** 2

# Calculate comoving volume
volume = 4 / 3 * np.pi * (2478 - 2195) ** 3 * survey_area_sr


#k_min * h and k_max * h
delta_k = 2 * np.pi / (volume ** (1/3))
minkh_t = 0.01
maxkh_t = 0.2

log_spaced_k = np.geomspace(minkh_t, maxkh_t, 20)

print('the value of the fundamental frequency is:')
print(minkh_t)

ratio = maxkh_t / minkh_t
print('the ratio is:', ratio)

# Calculate the angular diameter distance at the reference cosmology
d_Aref = c / H0

# Ω_HI
Omega_HI = 4.0 * (1 + z_values) ** (0.6) * 10 ** (-4)

# Set parameters for matter power spectrum computation
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
pars.InitPower.set_params(As=As, ns=ns)

# Enable WantTransfer parameter
pars.WantTransfer = True

# Set parameters for matter power spectrum computation
pars.set_matter_power(redshifts=z_values, kmax=maxkh_t)  # Set redshifts and k range for matter power spectrum
n_points = 20
# Compute results
results = camb.get_results(pars)
k, z, pk_values = results.get_matter_power_spectrum(minkh=minkh_t, maxkh=maxkh_t,
                                                    npoints=n_points)  # Get matter power spectrum
###CAREFUL THE FUNCTION get_matter_power_spectrum IN CAMB OUTPUT THE k in h/Mpc and P(k) in (Mpc/h)^3 BUT IN OUR CASE WE WANT Mpc^-1 Mpc^3

print(f"Transfer.kmax set to: {pars.Transfer.kmax}")
print(f"h set to: {pars.h}")
print(f"maxkh * param.h = {pars.h * maxkh_t}")
# Step 3: Get the matter power interpolator
#PK = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=True, k_hunit=True, kmax=maxkh_t, zmax=2)

#PK = results.get_matter_power_interpolator(nonlinear=False, hubble_units=False, k_hunit=False)


# Use the interpolator for the specified k_array # Example redshift
#pk_values = PK.P(redshift, log_spaced_k)



# σ_8
sigma_8 = results.get_sigma8()
print(sigma_8)

# Compute Hubble parameter values at each redshift
H_z_values = results.hubble_parameter(z_values)
h_z_values = H_z_values / 100
h_0 = H0 / 100


### WE RESCALE THE K AND PS
pk_values = pk_values / h_z_values ** 3
#k = k * h_z_values

# Extract the angular diameter distance values
d_A_values = results.angular_diameter_distance(z_values)

# Hubble constant at the reference cosmology from https://arxiv.org/pdf/2210.05705.pdf
H0_fiducial = 67

# Compute H(z) at the reference cosmology
H_ref = H0_fiducial

# Reshape the array which was (1,200)
pk_values = pk_values.reshape(-1)

# To get the comoving radial distance
background = camb.get_background(pars)
r = background.comoving_radial_distance(z_values)


# Apply Savitzky-Golay filtering for dewiggling (The idea is to remove the wiggles due to the BAO)
window_length = 11  # Choose an appropriate window length
poly_order = 3  # Choose an appropriate polynomial order
pk_nw = savgol_filter(pk_values, window_length, poly_order)

# Compute growth rate using CAMB
growth_rate = results.get_redshift_evolution(z_values, maxkh_t, ['growth'])
# Extract growth rate values
growth_rate = growth_rate[0, :, 0]

# Compute the mean or median growth rate
growth_rate = np.mean(growth_rate)



# INTEGRATION FUNCTION
def integrate_function(f, minkh_t, maxkh_t, n=n_points-1):
    # Generate n equally spaced points between a and b
    x = np.linspace(minkh_t, maxkh_t, n + 1)

    # Compute the function values at the points x
    y = f(x)

    # Apply the trapezoidal rule
    integral = np.trapz(y, x)

    return integral


# Define function f(z)
def f(kh):
    # Your function definition goes here
    return growth_rate ** 2 * pk_values


# INTEGRATION FUNCTION
def trapezoidal_rule(y, x):
    """
    Approximate the integral of y(x) using the trapezoidal rule.

    Parameters:
        y (array_like): Array of function values.
        x (array_like): Array of x-values corresponding to y.

    Returns:
        float: Approximation of the integral.
    """
    integral = np.trapz(y, x)
    return integral


# Integrate power spectrum over k using the trapezoidal rule
pk_int = trapezoidal_rule(pk_values, k)

# from https://www.aanda.org/articles/aa/full_html/2020/10/aa38071-20/aa38071-20.html

# σ_v controls the strength of the non-linear damping of the BAO signal in all directions in three-dimensions
sigma_v = np.sqrt((1 / (6 * np.pi ** 2)) * pk_int)

# g_m non-linear damping factor of the BAO signal
g_m = sigma_v ** 2 * (1 - mi ** 2 + mi ** 2 * (1 + growth_rate) ** 2)

# “de-wiggled” power spectrum
pk_dw = pk_values

# Compute the integral of f(z) from a to b
result_int = integrate_function(f, minkh_t, maxkh_t)

# print("Integral of f(z) from", minkh_t, "to", maxkh_t, ":", result_int)

# pairwise velocity dispersion modulate the strength of FoG effect from https://arxiv.org/pdf/2210.05705.pdf
sigma_p_sqrd = (1 / (6 * np.pi ** 2)) * result_int



factor = np.exp(-(k ** 2 * mi ** 2 * sigma ** 2 * c ** 2 / (np.array(H_z_values) ** 2)))

# cross-correlation coefficient
r_HI_g = 1



def calculate_PS(d_Aref, H_z_values, H_ref, r_HI_g, k, mi, result_int, h_z_values, z_values, Omega_HI, pk_dw,
                     frequency_to_temperature, theta_pb, r, b_g, growth_rate, b_HI, H0, sigma_8, factor):
    # Compute the mean brightness temperature in mK from https://arxiv.org/pdf/2210.05705.pdf
    T_b_mean_mK = 189 * np.array(h_z_values) * (1 + np.array(z_values)) ** 2 * H0 * np.array(Omega_HI) / np.array(
        H_z_values) #* frequency_to_temperature






    # Alcock-Paczynksi effect
    AP = np.array(d_Aref) ** 2 * np.array(H_z_values) / (2295.9 ** 2 * np.array(H_ref))

    sigma_p_sqrd = (1 / (6 * np.pi ** 2)) * result_int

    # FoG effect
    FoG = 1 / (1 + k ** 2 * mi ** 2 * sigma_p_sqrd)

    # resolution of the map
    Beta = np.exp((-(k ** 2) * (1 - mi ** 2) * r ** 2 * theta_pb ** 2) / (16 * np.log(2)))

    # K_rsd effect
    K_rsd = b_g * sigma_8 + growth_rate * sigma_8 * mi ** 2

    # K_rsd effect for HI
    K_rsd_HI = b_HI * sigma_8 + growth_rate * sigma_8 * mi ** 2

    P_IM_g = r_HI_g * AP * FoG * T_b_mean_mK * K_rsd * K_rsd_HI * pk_dw / (sigma_8 ** 2) * Beta
    # the cross-correlation power spectrum of IM with a galaxy sample with bias bg and cross-correlation coefficient rHI,g / T_b_mean : to plot
    P_IM_g_wt_T = P_IM_g / T_b_mean_mK

    # redshift-space power spectrum
    P_dd_zs = FoG * K_rsd ** 2 * pk_dw / sigma_8 ** 2

    P_gg = AP * P_dd_zs * factor

    P_dd_zs_HI = FoG * K_rsd_HI ** 2 * pk_dw / sigma_8 ** 2

    P_IM = (T_b_mean_mK) ** 2 * AP * P_dd_zs_HI * Beta ** 2


    # NOISE
    P_IM_noise = 2 * np.pi * f_sky / (nu_obs * t_tot * N_d) * (1 + z_values) ** 2 * r ** 2 / H_z_values * (
            T_sys / T_b_mean_mK) ** 2

    P_IM_IM_wt_T = P_IM / (T_b_mean_mK) ** 2

    return P_IM_g, P_gg, P_IM, P_IM_noise, P_IM_g_wt_T, P_IM_IM_wt_T, T_b_mean_mK


P_IM_g, P_gg, P_IM, P_IM_noise, P_IM_g_wt_T, P_IM_IM_wt_T, T_b_mean_mK = calculate_PS(d_Aref, H_z_values, H_ref, r_HI_g, k, mi, result_int, h_z_values, z_values, Omega_HI, pk_dw,
                     frequency_to_temperature, theta_pb, r, b_g, growth_rate, b_HI, H0, sigma_8, factor)


# Covariance matrix
# ADD THE NOISE

def shot_noise(number_density):
    """
    Compute shot noise for a galaxy survey using CAMB.

    Parameters:
        survey_area (float): Survey area in square degrees.
        redshift (float): Redshift of the survey.
        number_density (float): Number density of galaxies in units of galaxies per cubic Mpc.

    Returns:
        shot_noise (float): Shot noise level for the survey.
    """

    # Calculate shot noise
    shot_noise = 1.0 / (number_density)

    return shot_noise


# Example usage:

number_density = 3.43 * 10 ** (-4)  # number density in galaxies per cubic Mpc

P_gg_N = shot_noise(number_density)
# print("Shot noise level for the survey:", P_gg_N)

# \tilde{P} = P + P_noise as in https://arxiv.org/pdf/2210.05705.pdf
P_t_g = P_gg + P_gg_N
P_t_IM_IM = P_IM + P_IM_noise
P_t_g_IM = P_IM_g # from https://arxiv.org/pdf/2210.05705.pdf \tilde{P}_IM_g = P_IM_g because of the Kronecker delta

# CHECK THE ERRORS AND VOLUME SURVEY from https://arxiv.org/pdf/2210.05705.pdf
# delta_k = 0.1493

delta_mi = 0.02889
V_survey = f_sky * (4 * np.pi/3) * r ** 3

# factor for the covariance matrix https://arxiv.org/pdf/2210.05705.pdf
fact = 4 * np.pi ** 2 / (k ** 2 * log_spaced_k * delta_mi * V_survey)

# Build the 3 covariance matrices for gg, IM and IM,g as in https://arxiv.org/pdf/2210.05705.pdf
C_gg = fact * P_t_g ** 2
C_IM = fact * P_t_IM_IM ** 2
C_g_IM = fact * (P_t_g * P_t_IM_IM + P_t_g_IM ** 2)

diagonal_C_gg = np.diag(C_gg)
diagonal_C_IM = np.diag(C_IM)
diagonal_C_g_IM = np.diag(C_g_IM)

# Signal for IM_g
Signal_IM_g = P_IM_g

Signal_IM_g_transpose = Signal_IM_g.T

Cov_IM_g = diagonal_C_g_IM
Cov_IM_g_inv = np.linalg.inv(Cov_IM_g)

dot_product1_IM_g = np.dot(Signal_IM_g_transpose, Cov_IM_g_inv)
dot_product2_IM_g = np.dot(dot_product1_IM_g, Signal_IM_g)

# SNR for IM,g
snr_IM_g = np.sqrt(dot_product2_IM_g)

print("SNR for IM_g:", snr_IM_g)

# Signal for gg
Signal_gg = P_gg

Signal_gg_transpose = Signal_gg.T

Cov_gg = diagonal_C_gg
Cov_gg_inv = np.linalg.inv(Cov_gg)

dot_product1_gg = np.dot(Signal_gg_transpose, Cov_gg_inv)
dot_product2_gg = np.dot(dot_product1_gg, Signal_gg)

snr_gg = np.sqrt(dot_product2_gg)

print("SNR for gg:", snr_gg)

# Signal for IM
Signal_IM = P_IM

Signal_IM_transpose = Signal_IM.T

Cov_IM = diagonal_C_IM

det_IM = np.linalg.det(Cov_IM)
Cov_IM_inv = np.linalg.inv(Cov_IM)

dot_product1_IM = np.dot(Signal_IM_transpose, Cov_IM_inv)
dot_product2_IM = np.dot(dot_product1_IM, Signal_IM)

snr_IM = np.sqrt(dot_product2_IM)

print("SNR for IM:", snr_IM)

# Extract the data from previous calculations
k_values = k.reshape(-1)  # Flatten the k array
P_gg_values = P_gg.reshape(-1)
P_IM_values = P_IM.reshape(-1)
P_gg_N_values = np.full_like(k_values, P_gg_N)  # Using the shot noise level as error
P_IM_noise_values = P_IM_noise.reshape(-1)

print('h is',h_z_values)

# Plotting the power spectrum with error bars
plt.figure(figsize=(10, 6))

# Plot P_gg with error bars
plt.errorbar(log_spaced_k * h_0, P_gg, yerr= np.sqrt(C_gg), fmt='.', capsize=0.01, label='P_gg', color='green')
# Plot P_gg with error bars
plt.errorbar(log_spaced_k * h_0, P_IM_g_wt_T, yerr= np.sqrt(C_g_IM)/T_b_mean_mK, fmt='.', capsize=0.01, label='P_IM_g', color='blue')
# Plot P_IM with error bars
plt.errorbar(log_spaced_k * h_0, P_IM_IM_wt_T, yerr= np.sqrt(C_IM)/(T_b_mean_mK**2), fmt='.', capsize=0.01, label='P_IM', color='magenta')
plt.xscale('log')
plt.yscale('log')

# Labels and legend
plt.xlabel('k (Mpc$^{-1}$)')
plt.ylabel('Power Spectrum (Mpc$^3$)')
plt.title('Power Spectrum as a function of k with Error Bars')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


def log_likelihood_IM_IM(theta, data):
    b_HI_e, H_0_e = theta


    # Set parameters for matter power spectrum computation
    pars1 = camb.CAMBparams()
    pars1.set_cosmology(H0=H_0_e, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    pars1.InitPower.set_params(As=As, ns=ns)

    # Enable WantTransfer parameter
    pars1.WantTransfer = True

    # Set parameters for matter power spectrum computation
    pars1.set_matter_power(redshifts=z_values, kmax=maxkh_t)  # Set redshifts and k range for matter power spectrum
    n_points = 20

    # Compute results
    results1 = camb.get_results(pars1)
    k1, z1, pk_values1 = results1.get_matter_power_spectrum(minkh=minkh_t, maxkh=maxkh_t,
                                                        npoints=n_points)  # Get matter power spectrum
    ###CAREFUL THE FUNCTION get_matter_power_spectrum IN CAMB OUTPUT THE k in h/Mpc and P(k) in (Mpc/h)^3 BUT IN OUR CASE WE WANT Mpc^-1 Mpc^3

    sigma_8 = results1.get_sigma8()

    # Compute Hubble parameter values at each redshift
    H_z_values = results1.hubble_parameter(z_values)
    h_z_values = H_z_values / 100

    ### WE RESCALE THE K AND PS
    pk_values1 = pk_values1 / h_z_values ** 3
    k1 = k1 * h_z_values

    # Extract the angular diameter distance values
    d_A_values = results1.angular_diameter_distance(z_values)

    # Hubble constant at the reference cosmology from https://arxiv.org/pdf/2210.05705.pdf
    H0_fiducial1 = H_0_e

    # Compute H(z) at the reference cosmology
    H_ref1 = H0_fiducial1

    # Reshape the array which was (1,200)
    pk_values1 = pk_values1.reshape(-1)

    # To get the comoving radial distance
    background1 = camb.get_background(pars1)
    r1 = background.comoving_radial_distance(z_values)

    # Apply Savitzky-Golay filtering for dewiggling (The idea is to remove the wiggles due to the BAO)
    window_length = 11  # Choose an appropriate window length
    poly_order = 3  # Choose an appropriate polynomial order
    pk_nw1 = savgol_filter(pk_values1, window_length, poly_order)


    # Compute growth rate using CAMB



    growth_rate1 = results1.get_redshift_evolution(z_values, maxkh_t, ['growth'])
    # Extract growth rate values
    growth_rate1 = growth_rate1[0, :, 0]

    # INTEGRATION FUNCTION
    def integrate_function(f, minkh_t, maxkh_t, n=n_points - 1):
        # Generate n equally spaced points between a and b
        x = np.linspace(minkh_t, maxkh_t, n + 1)

        # Compute the function values at the points x
        y = f(x)

        # Apply the trapezoidal rule
        integral = np.trapz(y, x)

        return integral

    # Define function f(z)
    def f(kh):
        # Your function definition goes here
        return growth_rate1 ** 2 * pk_values1

    # INTEGRATION FUNCTION
    def trapezoidal_rule(y, x):
        """
        Approximate the integral of y(x) using the trapezoidal rule.

        Parameters:
            y (array_like): Array of function values.
            x (array_like): Array of x-values corresponding to y.

        Returns:
            float: Approximation of the integral.
        """
        integral = np.trapz(y, x)
        return integral

    # Integrate power spectrum over k using the trapezoidal rule
    pk_int1 = trapezoidal_rule(pk_values1, k1)

    # from https://www.aanda.org/articles/aa/full_html/2020/10/aa38071-20/aa38071-20.html

    # σ_v controls the strength of the non-linear damping of the BAO signal in all directions in three-dimensions
    sigma_v1 = np.sqrt((1 / (6 * np.pi ** 2)) * pk_int1)

    # g_m non-linear damping factor of the BAO signal
    g_m1 = sigma_v1 ** 2 * (1 - mi ** 2 + mi ** 2 * (1 + growth_rate1) ** 2)

    # “de-wiggled” power spectrum
    pk_dw1 = pk_values1 * np.exp(- g_m1 * k1 ** 2) + pk_nw1 * (1 - np.exp(- g_m1 * k1 ** 2))

    # Compute the integral of f(z) from a to b
    result_int = integrate_function(f, minkh_t, maxkh_t)

    # print("Integral of f(z) from", minkh_t, "to", maxkh_t, ":", result_int)

    # pairwise velocity dispersion modulate the strength of FoG effect from https://arxiv.org/pdf/2210.05705.pdf

    _, _, model_IM_IM, _,_,_,_ = calculate_PS(d_Aref, H_z_values, H_ref1, r_HI_g, k1, mi, result_int, h_z_values, z_values,
                                           Omega_HI, pk_dw1,
                                           frequency_to_temperature, theta_pb, r1, b_g, growth_rate1, b_HI_e, H_0_e,
                                           sigma_8, factor)
    sigma = 0.1  # Assuming a constant observational uncertainty
    IM_IM_loglike = -0.5 * np.sum(np.dot((data - model_IM_IM), np.dot(Cov_IM_inv, (data - model_IM_IM)) / sigma ** 2))
    return IM_IM_loglike

def log_likelihood_IM_g(theta, data):
    b_HI_e, H_0_e = theta

    # Set parameters for matter power spectrum computation
    pars1 = camb.CAMBparams()
    pars1.set_cosmology(H0=H_0_e, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    pars1.InitPower.set_params(As=As, ns=ns)

    # Enable WantTransfer parameter
    pars1.WantTransfer = True

    # Set parameters for matter power spectrum computation
    pars1.set_matter_power(redshifts=z_values, kmax=maxkh_t)  # Set redshifts and k range for matter power spectrum
    n_points = 20
    # Compute results
    results1 = camb.get_results(pars1)
    k1, z1, pk_values1 = results1.get_matter_power_spectrum(minkh=minkh_t, maxkh=maxkh_t,
                                                        npoints=n_points)  # Get matter power spectrum
    ###CAREFUL THE FUNCTION get_matter_power_spectrum IN CAMB OUTPUT THE k in h/Mpc and P(k) in (Mpc/h)^3 BUT IN OUR CASE WE WANT Mpc^-1 Mpc^3

    sigma_8 = results1.get_sigma8()

    # Compute Hubble parameter values at each redshift
    H_z_values = results1.hubble_parameter(z_values)
    h_z_values = H_z_values / 100

    ### WE RESCALE THE K AND PS
    pk_values1 = pk_values1 / h_z_values ** 3
    k1 = k1 * h_z_values

    # Extract the angular diameter distance values
    d_A_values = results1.angular_diameter_distance(z_values)

    # Hubble constant at the reference cosmology from https://arxiv.org/pdf/2210.05705.pdf
    H0_fiducial1 = H_0_e

    # Compute H(z) at the reference cosmology
    H_ref1 = H0_fiducial1

    # Reshape the array which was (1,200)
    pk_values1 = pk_values1.reshape(-1)

    # To get the comoving radial distance
    background1 = camb.get_background(pars1)
    r1 = background.comoving_radial_distance(z_values)

    # Apply Savitzky-Golay filtering for dewiggling (The idea is to remove the wiggles due to the BAO)
    window_length = 11  # Choose an appropriate window length
    poly_order = 3  # Choose an appropriate polynomial order
    pk_nw1 = savgol_filter(pk_values1, window_length, poly_order)



    # Compute growth rate using CAMB
    growth_rate1 = results1.get_redshift_evolution(z_values, maxkh_t, ['growth'])
    # Extract growth rate values
    growth_rate1 = growth_rate1[0, :, 0]

    # INTEGRATION FUNCTION
    def integrate_function(f, minkh_t, maxkh_t, n=n_points - 1):
        # Generate n equally spaced points between a and b
        x = np.linspace(minkh_t, maxkh_t, n + 1)

        # Compute the function values at the points x
        y = f(x)

        # Apply the trapezoidal rule
        integral = np.trapz(y, x)

        return integral

    # Define function f(z)
    def f(kh):
        # Your function definition goes here
        return growth_rate1 ** 2 * pk_values1

    # INTEGRATION FUNCTION
    def trapezoidal_rule(y, x):
        """
        Approximate the integral of y(x) using the trapezoidal rule.

        Parameters:
            y (array_like): Array of function values.
            x (array_like): Array of x-values corresponding to y.

        Returns:
            float: Approximation of the integral.
        """
        integral = np.trapz(y, x)
        return integral

    # Integrate power spectrum over k using the trapezoidal rule
    pk_int1 = trapezoidal_rule(pk_values1, k1)









    # from https://www.aanda.org/articles/aa/full_html/2020/10/aa38071-20/aa38071-20.html

    # σ_v controls the strength of the non-linear damping of the BAO signal in all directions in three-dimensions
    sigma_v1 = np.sqrt((1 / (6 * np.pi ** 2)) * pk_int1)

    # g_m non-linear damping factor of the BAO signal
    g_m1 = sigma_v1 ** 2 * (1 - mi ** 2 + mi ** 2 * (1 + growth_rate1) ** 2)

    # “de-wiggled” power spectrum
    pk_dw1 = pk_values1 * np.exp(- g_m1 * k1 ** 2) + pk_nw1 * (1 - np.exp(- g_m1 * k1 ** 2))

    # Compute the integral of f(z) from a to b
    result_int = integrate_function(f, minkh_t, maxkh_t)

    # print("Integral of f(z) from", minkh_t, "to", maxkh_t, ":", result_int)

    # pairwise velocity dispersion modulate the strength of FoG effect from https://arxiv.org/pdf/2210.05705.pdf

    model_IM_g, _, _, _, _, _, _ = calculate_PS(d_Aref, H_z_values, H_ref1, r_HI_g, k1, mi, result_int, h_z_values, z_values,
                                           Omega_HI, pk_dw1,
                                           frequency_to_temperature, theta_pb, r1, b_g, growth_rate1, b_HI_e, H_0_e,
                                           sigma_8, factor)
    sigma = 0.1  # Assuming a constant observational uncertainty

    IM_g_loglike = -0.5 * np.sum(np.dot((data - model_IM_g), np.dot(Cov_IM_g_inv, (data - model_IM_g)) / sigma ** 2))

    return IM_g_loglike


# Define the log prior function with adjusted prior for b_HI
def log_prior_IM_IM(theta):
    b_HI_e, H_O_e = theta
    # Set prior ranges for b_HI and sigma8
    b_HI_min, b_HI_max = 0.2, 2  # Adjusted range for b_HI based on physical considerations
    H_0_min, H_0_max = 50, 110

    # Flat priors for b_HI and sigma8 within specified ranges
    if b_HI_min < theta[0] < b_HI_max and H_0_min < theta[1] < H_0_max:
        return 0.0  # Prior probability is uniform within the specified ranges
    return -np.inf  # Return negative infinity for points outside the specified ranges


def log_prior_IM_g(theta):
    b_HI_e, H_O_e = theta
    # Set prior ranges for b_HI and sigma8
    b_HI_min, b_HI_max = 0.2, 2  # Adjusted range for b_HI based on physical considerations
    H_0_min, H_0_max = 50, 110

    # Flat priors for b_HI and sigma8 within specified ranges
    if b_HI_min < theta[0] < b_HI_max and H_0_min < theta[1] < H_0_max:
        return 0.0  # Prior probability is uniform within the specified ranges
    return -np.inf  # Return negative infinity for points outside the specified ranges


# Define the log posterior function
def log_posterior_IM_IM(theta, data):
    lp = log_prior_IM_IM(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_IM_IM(theta, data)

def log_posterior_IM_g(theta, data):
    lp = log_prior_IM_g(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_IM_g(theta, data)





# Generate synthetic data (you should replace this with your actual data)
np.random.seed(42)

true_params = np.array([b_HI[0], H0])
print(true_params)


P_IM_g, P_gg, P_IM, P_IM_noise, P_IM_g_wt_T, P_IM_IM_wt_T,_= calculate_PS(d_Aref, H_z_values, H_ref, r_HI_g, k, mi, result_int, h_z_values, z_values, Omega_HI, pk_dw,
                     frequency_to_temperature, theta_pb, r, b_g, growth_rate, true_params[0], true_params[1], sigma_8, factor)
data_IM_IM = P_IM + P_IM_noise
data_IM_g = P_IM_g + np.sqrt(P_IM_noise + shot_noise(number_density))


# Setup the getdist sampler
ndim = 2  # Number of parameters
nwalkers = 4  # Number of walkers
nsteps = 200  # Number of steps

# Initialize walkers in a small Gaussian ball around the initial guess
initial_guess = [0.5, 60]
# Adjust step sizes
step_sizes = [1e-3, 1e-2]  # Initial step sizes for b_HI and sigma8 respectively

# Reduce step size for b_HI
#step_sizes[0] = 1e-4  # Adjust the step size for b_HI


# Setup the getdist sampler with adjusted step sizes
pos = initial_guess + step_sizes * np.random.randn(nwalkers, ndim)




sampler_IM_IM = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_IM_IM, args=(data_IM_IM,))
#sampler_IM_IM.run_mcmc(initial_state=pos, nsteps=nsteps, progress=True)

sampler_IM_g = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_IM_g, args=(data_IM_g,))
#sampler_IM_g.run_mcmc(initial_state=pos, nsteps=nsteps, progress=True)


burnin = int(0.3 * nsteps)

max_n = 20000
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

max_n = 20000
# Now we'll sample for up to max_n steps
for sample in sampler_IM_IM.sample(pos, iterations=max_n, progress=True):
# Only check convergence every 100 steps
    if sampler_IM_IM.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler_IM_IM.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler_IM_IM.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        samples_IM_IM = sampler_IM_IM.get_chain(discard=burnin, flat=True)
        break
    old_tau = tau

# Now we'll sample for up to max_n steps
for sample in sampler_IM_g.sample(pos, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler_IM_g.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler_IM_g.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler_IM_g.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        samples_IM_g = sampler_IM_g.get_chain(discard=burnin, flat=True)
        break
    old_tau = tau









# Adjust bandwidth manually for b_HI, and H_0
bandwidth_b_HI = 1e-4
bandwidth_H_0 = 1e-4



# Define the thinning factor
thinning_factor = 2  # Keep every second sample


# Apply thinning to your MCMC samples
thinned_samples_chain_IM_IM = samples_IM_IM[::thinning_factor]
thinned_samples_chain_IM_g = samples_IM_g[::thinning_factor]

# Create MCSamples object with adjusted fine_bins and bandwidths
getdist_samples_IM_IM = MCSamples(samples=thinned_samples_chain_IM_IM, names=['b_HI', 'H_0'], labels=[r'b_{HI}', r'H_0'], settings={'fine_bins': 2000, 'smooth_scale_1D': bandwidth_b_HI})
getdist_samples_IM_g = MCSamples(samples=thinned_samples_chain_IM_g, names=['b_HI', 'H_0'], labels=[r'b_{HI}', r'H_0'], settings={'fine_bins': 2000, 'smooth_scale_1D': bandwidth_b_HI})


#save data to file




# Plot
g = plots.getSubplotPlotter()
g.settings.num_plot_contours = 5  # Increase the number of contour levels more

# Assuming getdist_samples_IM_IM and getdist_samples_IM_g are properly defined
g.triangle_plot([getdist_samples_IM_g, getdist_samples_IM_IM], filled=True, contour_colors=['b', 'm'], alpha=[0.1, 0.5], legend_labels=['IM_g', 'IM_IM'])



# Save the plot to a file
plt.savefig('new_getdist_triangle_plot_comparison.pdf')

# Show the plot
plt.show()



dictionary = {'thinned_samples_chain_IM_IM': thinned_samples_chain_IM_IM, 'thinned_samples_chain_IM_g':thinned_samples_chain_IM_g}
np.savez('smaples_MCMC.npz', **dictionary)




# Calculation of Gelman Rubin coefficient
def gelman_rubin(chains):
    if len(chains) == 0:
        return np.nan  # Return NaN if there are no chains

    num_chains = len(chains)
    num_samples = chains[0].shape[0]

    # Check if there are any samples
    if num_samples == 0:
        return np.nan  # Return NaN if there are no samples

    # Calculate the within-chain variances
    W = np.mean([np.var(chain, axis=0, ddof=1) for chain in chains], axis=0)

    # Calculate the between-chain variances if there are multiple chains
    if num_chains > 1:
        means = np.mean(chains, axis=1)
        B = num_samples * np.var(means, axis=0, ddof=1)
    else:
        B = np.nan  # Set B to NaN if there is only one chain

    # Estimate the variance of the pooled posterior
    V_hat = ((num_samples - 1) / num_samples) * W + (1 / num_samples) * B

    # Calculate the potential scale reduction factor
    R_hat = np.sqrt(V_hat / W)

    return R_hat



# Get the samples array from the samples object
samples_array_IM_IM = getdist_samples_IM_IM
samples_array_IM_g = getdist_samples_IM_g


# Calculate Gelman-Rubin statistic for samples after burn-in
R_hat_IM_IM = gelman_rubin(thinned_samples_chain_IM_IM)
R_hat_IM_g = gelman_rubin(thinned_samples_chain_IM_g)

# Check convergence
converged = np.all(np.abs(R_hat_IM_IM - 1) < 0.01)
print("Convergence status for IM_IM:", converged)
print("Gelman-Rubin statistic (R-hat) for IM_IM:", R_hat_IM_IM)

converged = np.all(np.abs(R_hat_IM_g - 1) < 0.01)
print("Convergence status for IM_g:", converged)
print("Gelman-Rubin statistic (R-hat) for IM_g:", R_hat_IM_g)


# Concatenate the parameter chains from all walkers
parameter_chains_IM_IM = np.concatenate((samples_IM_IM[:, 0], samples_IM_IM[:, 1]))
parameter_chains_IM_g = np.concatenate((samples_IM_g[:, 0], samples_IM_g[:, 1]))


# Reshape the parameter chains to have shape (num_steps, num_parameters)
parameter_chains_IM_IM = parameter_chains_IM_IM.reshape(-1, 2)
parameter_chains_IM_g = parameter_chains_IM_g.reshape(-1, 2)

# Calculate the correlation matrix
correlation_matrix_IM_IM = np.corrcoef(parameter_chains_IM_IM.T)
correlation_matrix_IM_g = np.corrcoef(parameter_chains_IM_g.T)

print("Correlation matrix for IM_IM:")
print(correlation_matrix_IM_IM)

print("Correlation matrix for IM_g:")
print(correlation_matrix_IM_g)


# Plot the MCMC samples
fig_IM_IM, axes_IM_IM = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ['b_HI', 'H_0']

fig_IM_g, axes_IM_g = plt.subplots(ndim, figsize=(10, 7), sharex=True)
labels = ['b_HI', 'H_0']

for i in range(ndim):
    ax_IM_IM = axes_IM_IM[i]
    for j in range(nwalkers):  # Iterate over each walker
        ax_IM_IM.plot(samples_IM_IM[:, i], color="k", alpha=0.3)  # Plot trajectory for each walker
    ax_IM_IM.set_ylabel(labels[i])
    ax_IM_IM.yaxis.set_label_coords(-0.1, 0.5)

for ax_IM_g in range(ndim):
    ax_IM_g = axes_IM_g[i]
    for j in range(nwalkers):  # Iterate over each walker
        ax_IM_g.plot(samples_IM_g[:, i], color="k", alpha=0.3)  # Plot trajectory for each walker
    ax_IM_g.set_ylabel(labels[i])
    ax_IM_g.yaxis.set_label_coords(-0.1, 0.5)

axes_IM_IM[-1].set_xlabel("Step number for IM_IM")
plt.tight_layout()
plt.show()

axes_IM_g[-1].set_xlabel("Step number for IM_g")
plt.tight_layout()
plt.show()


# Set up the backend
# Don't forget to clear it in case the file already exists
