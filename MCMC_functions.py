import camb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from getdist import *
import emcee
import getdist.plots as plots
import time
from multiprocessing import Pool


class MCMC_functions:

    def __init__(self, c, kB, z_values, fid_z, ombh2, omch2, H_0, As, ns, mnu, omk, f_sky, t_tot, T_sys, N_d, nu_21cm_rest, mi, b_g, sigma, survey_area, min_k, max_k, n_points, H0_fiducial,r_HI_g, d, number_density, delta_mi):
        self.c = c  # Speed of light in km/s
        self.kB = kB
        self.z_values = z_values
        self.fid_z = fid_z
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.H_0 = H_0
        self.As = As
        self.ns = ns
        self.mnu = mnu
        self.omk = omk
        self.f_sky = f_sky
        self.t_tot = t_tot
        self.T_sys = T_sys
        self.N_d = N_d
        self.nu_21cm_rest = nu_21cm_rest
        self.mi = mi
        self.b_g = b_g
        self.sigma = sigma
        self.survey_area = survey_area
        self.min_k = min_k
        self.max_k = max_k
        self.n_points = n_points
        self.H0_fiducial = H0_fiducial
        self.r_HI_g = r_HI_g
        self.d = d
        self.number_density = delta_mi



    def frequency_to_temperature(self):
        frequency_to_temperature = (c ** 2) / (2 * kB * 1e9)
        return frequency_to_temperature

    def calculate_nu_obs(self):
        nu_obs = nu_21cm_rest / (1 + self.z_values)
        return nu_obs

    def calculate_b_HI(self):
        b_HI = 0.3 * (1 + self.z_values) + 0.65
        return b_HI

    def calculate_lambda(self):
        # Wavelength of the observed redshift frequency
        l = (1 + self.z_values) * 21 * 10 ** (-2)
        return l

    def calculate_theta_pb(self, l):
        # Full width at half maximum of the dish primary beam
        theta_pb = 1.22 * l / self.d
        return theta_pb

    def calculate_survey_area_sr(self):
        survey_area_sr = np.deg2rad(self.survey_area) ** 2
        return survey_area_sr

    def calculate_comoving_volume(self, survey_area_sr):
        volume = 4 / 3 * np.pi * (2478 - 2195) ** 3 * survey_area_sr
        return volume

    def calculate_survey_volume(self, r):
        V_survey = f_sky * (4 * np.pi / 3) * r ** 3
        return V_survey

    def calculate_cov_matrix_factor(self, log_spaced_k, V_survey):
        # factor for the covariance matrix https://arxiv.org/pdf/2210.05705.pdf
        fact = 4 * np.pi ** 2 / (k ** 2 * log_spaced_k * self.delta_mi * V_survey)
        return fact

    def calculate_covariance_matrices(self, fact, P_t_g, P_t_IM_IM, P_t_g_IM):
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

        # Signal for gg
        Signal_gg = P_gg

        Signal_gg_transpose = Signal_gg.T

        Cov_gg = diagonal_C_gg
        Cov_gg_inv = np.linalg.inv(Cov_gg)

        dot_product1_gg = np.dot(Signal_gg_transpose, Cov_gg_inv)
        dot_product2_gg = np.dot(dot_product1_gg, Signal_gg)

        snr_gg = np.sqrt(dot_product2_gg)

        # Signal for IM
        Signal_IM = P_IM

        Signal_IM_transpose = Signal_IM.T

        Cov_IM = diagonal_C_IM

        det_IM = np.linalg.det(Cov_IM)
        Cov_IM_inv = np.linalg.inv(Cov_IM)

        dot_product1_IM = np.dot(Signal_IM_transpose, Cov_IM_inv)
        dot_product2_IM = np.dot(dot_product1_IM, Signal_IM)

        snr_IM = np.sqrt(dot_product2_IM)

        return Cov_gg_inv, Cov_IM_inv, Cov_IM_g_inv, snr_gg, snr_IM, snr_IM_g

    def calculate_delta_k(self, volume):
        delta_k = 2 * np.pi / (volume ** (1 / 3))
        return delta_k

    def calculate_d_Aref(self):
        d_Aref = self.c / self.H_0
        return d_Aref

    def calculate_Omega_HI(self):
        Omega_HI = 4.0 * (1 + self.z_values) ** (0.6) * 10 ** (-4)
        return Omega_HI

    def CAMB_output(self):
        # Set parameters for matter power spectrum computation
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H_0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk)
        pars.InitPower.set_params(As=self.As, ns=self.ns)

        # Enable WantTransfer parameter
        pars.WantTransfer = True

        # Set parameters for matter power spectrum computation
        pars.set_matter_power(redshifts=self.z_values, kmax=self.max_k)  # Set redshifts and k range for matter power spectrum
        # Compute results
        results = camb.get_results(pars)
        k, z, pk_values = results.get_matter_power_spectrum(minkh=self.min_k, maxkh=self.max_k,
                                                            npoints=self.n_points)  # Get matter power spectrum
        ###CAREFUL THE FUNCTION get_matter_power_spectrum IN CAMB OUTPUT THE k in h/Mpc and P(k) in (Mpc/h)^3 BUT IN OUR CASE WE WANT Mpc^-1 Mpc^3


        return k, z, pk_values, results, pars


    def calculate_sigma_8(self, results):
        sigma_8 = results.get_sigma8()
        return  sigma_8

    def calculate_H(self, results):
        # Compute Hubble parameter values at each redshift
        H_z_values = results.hubble_parameter(z_values)
        h_z_values = H_z_values / 100
        h_0 = self.H_0 / 100
        return H_z_values, h_z_values, h_0

    def calculate_pk_rescaled(self, pk_values, h_z_values):
        pk_values = pk_values / h_z_values ** 3
        pk_values = pk_values.reshape(-1)
        return pk_values

    def calculate_diameter_distance(self, results):
        d_A_values = results.angular_diameter_distance(self.z_values)
        return d_A_values

    def calculate_comoving_radial_distance(self, pars):
        background = camb.get_background(pars)
        r = background.comoving_radial_distance(self.z_values)
        return r

    def dewiggling_PS(self, pk_values):
        # Apply Savitzky-Golay filtering for dewiggling (The idea is to remove the wiggles due to the BAO)
        window_length = 11  # Choose an appropriate window length
        poly_order = 3  # Choose an appropriate polynomial order
        pk_nw = savgol_filter(pk_values, window_length, poly_order)
        return pk_nw

    def calculate_growth_rate(self, results):
        growth_rate = results.get_redshift_evolution(self.z_values, self.max_k, ['growth'])
        # Extract growth rate values
        growth_rate = growth_rate[0, :, 0]
        # Compute the mean or median growth rate
        growth_rate = np.mean(growth_rate)
        return growth_rate

    def integrate_function(self, f, growth_rate, pk_values):
        # Generate n equally spaced points between a and b
        n = self.n_points - 1
        x = np.linspace(self.min_k, self.max_k, n + 1)

        # Compute the function values at the points x
        y = f(self, growth_rate, pk_values)

        # Apply the trapezoidal rule
        integral = np.trapz(y, x)

        return integral

    # Define function f(z)
    def calculatge_f(self, growth_rate, pk_values):
        # Your function definition goes here
        result = growth_rate ** 2 * pk_values
        return result

    def trapezoidal_rule(self, y, x):
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

    def calculate_pk_int(self, trapezoidal_rule, pk_values, k):
        pk_int = trapezoidal_rule(pk_values, k)
        return pk_int

    def calculate_sigma_v(self, pk_int):
        # σ_v controls the strength of the non-linear damping of the BAO signal in all directions in three-dimensions
        sigma_v = np.sqrt((1 / (6 * np.pi ** 2)) * pk_int)
        return sigma_v

    def calculate_g_m(self, sigma_v, growth_rate):
        # g_m non-linear damping factor of the BAO signal
        g_m = sigma_v ** 2 * (1 - self.mi ** 2 + self.mi ** 2 * (1 + growth_rate) ** 2)
        return g_m

    def calculate_sigma_p_squared(self, integrate_function, f):
        # Compute the integral of f(z) from a to b
        result_int = integrate_function(f, self.min_k, self.max_k)
        # pairwise velocity dispersion modulate the strength of FoG effect from https://arxiv.org/pdf/2210.05705.pdf
        sigma_p_sqrd = (1 / (6 * np.pi ** 2)) * result_int
        return sigma_p_sqrd

    def calculate_factor(self, k, H_z_values):
        factor = np.exp(-(k ** 2 * self.mi ** 2 * sigma ** 2 * self.c ** 2 / (np.array(H_z_values) ** 2)))
        return factor


    def calculate_PS(self, d_Aref, H_z_values, k, result_int, h_z_values, Omega_HI, pk_dw,
                     frequency_to_temperature, theta_pb, r, growth_rate, b_HI, H_0, sigma_8, factor, nu_obs):

        H_ref = self.H0_fiducial
        # Compute the mean brightness temperature in mK from https://arxiv.org/pdf/2210.05705.pdf
        T_b_mean_mK = 189 * np.array(h_z_values) * (1 + np.array(z_values)) ** 2 * H_0 * np.array(Omega_HI) / np.array(
            H_z_values)  # * frequency_to_temperature

        # Alcock-Paczynksi effect
        AP = np.array(d_Aref) ** 2 * np.array(H_z_values) / (2295.9 ** 2 * np.array(H_ref))

        sigma_p_sqrd = (1 / (6 * np.pi ** 2)) * result_int

        # FoG effect
        FoG = 1 / (1 + k ** 2 * self.mi ** 2 * sigma_p_sqrd)

        # resolution of the map
        Beta = np.exp((-(k ** 2) * (1 - self.mi ** 2) * r ** 2 * theta_pb ** 2) / (16 * np.log(2)))

        # K_rsd effect
        K_rsd = self.b_g * sigma_8 + growth_rate * sigma_8 * self.mi ** 2

        # K_rsd effect for HI
        K_rsd_HI = b_HI * sigma_8 + growth_rate * sigma_8 * self.mi ** 2

        P_IM_g = self.r_HI_g * AP * FoG * T_b_mean_mK * K_rsd * K_rsd_HI * pk_dw / (sigma_8 ** 2) * Beta
        # the cross-correlation power spectrum of IM with a galaxy sample with bias bg and cross-correlation coefficient rHI,g / T_b_mean : to plot
        P_IM_g_wt_T = P_IM_g / T_b_mean_mK

        # redshift-space power spectrum
        P_dd_zs = FoG * K_rsd ** 2 * pk_dw / sigma_8 ** 2

        P_gg = AP * P_dd_zs * factor

        P_dd_zs_HI = FoG * K_rsd_HI ** 2 * pk_dw / sigma_8 ** 2

        P_IM = (T_b_mean_mK) ** 2 * AP * P_dd_zs_HI * Beta ** 2

        # NOISE
        P_IM_noise = 2 * np.pi * self.f_sky / (nu_obs * self.t_tot * self.N_d) * (1 + self.z_values) ** 2 * r ** 2 / H_z_values * (
                self.T_sys / T_b_mean_mK) ** 2

        P_IM_IM_wt_T = P_IM / (T_b_mean_mK) ** 2

        return P_IM_g, P_gg, P_IM, P_IM_noise, P_IM_g_wt_T, P_IM_IM_wt_T, T_b_mean_mK

    def shot_noise(self):

        # Calculate shot noise
        shot_noise = 1.0 / (self.number_density)

        return shot_noise

    def P_tilde(self, shot_noise, P_gg, P_IM, P_IM_g, P_IM_noise):
        P_t_g = P_gg + shot_noise
        P_t_IM_IM = P_IM + P_IM_noise
        P_t_g_IM = P_IM_g
        return P_t_g, P_t_IM_IM, P_t_g_IM

    def log_spaced_k(self):
        log_spaced_k = np.geomspace(self.min_k, self.max_k, self.n_points)
        return log_spaced_k

    def log_likelihood_IM_IM(self, theta, data, calculate_PS, Cov_IM_inv):
        b_HI_e, H_0_e = theta

        # Set parameters for matter power spectrum computation
        pars1 = camb.CAMBparams()
        pars1.set_cosmology(H0=H_0_e, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk)
        pars1.InitPower.set_params(As=self.As, ns=self.ns)

        # Enable WantTransfer parameter
        pars1.WantTransfer = True

        # Set parameters for matter power spectrum computation
        pars1.set_matter_power(redshifts=self.z_values, kmax=self.max_k)  # Set redshifts and k range for matter power spectrum

        # Compute results
        results1 = camb.get_results(pars1)
        k1, z1, pk_values1 = results1.get_matter_power_spectrum(minkh=self.min_k, maxkh=self.max_k,
                                                                npoints=self.n_points)  # Get matter power spectrum
        ###CAREFUL THE FUNCTION get_matter_power_spectrum IN CAMB OUTPUT THE k in h/Mpc and P(k) in (Mpc/h)^3 BUT IN OUR CASE WE WANT Mpc^-1 Mpc^3

        sigma_8 = results1.get_sigma8()

        # Compute Hubble parameter values at each redshift
        H_z_values = results1.hubble_parameter(self.z_values)
        h_z_values = H_z_values / 100

        ### WE RESCALE THE K AND PS
        pk_values1 = pk_values1 / h_z_values ** 3
        k1 = k1 * h_z_values

        # Extract the angular diameter distance values
        d_A_values = results1.angular_diameter_distance(self.z_values)

        # Hubble constant at the reference cosmology from https://arxiv.org/pdf/2210.05705.pdf
        H0_fiducial1 = H_0_e

        # Compute H(z) at the reference cosmology
        H_ref1 = H0_fiducial1

        # Reshape the array which was (1,200)
        pk_values1 = pk_values1.reshape(-1)

        # To get the comoving radial distance
        background1 = camb.get_background(pars1)
        r1 = background1.comoving_radial_distance(self.z_values)

        # Apply Savitzky-Golay filtering for dewiggling (The idea is to remove the wiggles due to the BAO)
        window_length = 11  # Choose an appropriate window length
        poly_order = 3  # Choose an appropriate polynomial order
        pk_nw1 = savgol_filter(pk_values1, window_length, poly_order)

        # Compute growth rate using CAMB

        growth_rate1 = results1.get_redshift_evolution(self.z_values, self.max_k, ['growth'])
        # Extract growth rate values
        growth_rate1 = growth_rate1[0, :, 0]

        # INTEGRATION FUNCTION
        def integrate_function1(self, f):
            # Generate n equally spaced points between a and b
            n = self.n_points - 1
            x = np.linspace(self.min_k, self.max_k, n + 1)

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
        def trapezoidal_rule1(y, x):
            integral = np.trapz(y, x)
            return integral

        # Integrate power spectrum over k using the trapezoidal rule
        pk_int1 = trapezoidal_rule1(pk_values1, k1)

        # from https://www.aanda.org/articles/aa/full_html/2020/10/aa38071-20/aa38071-20.html

        # σ_v controls the strength of the non-linear damping of the BAO signal in all directions in three-dimensions
        sigma_v1 = np.sqrt((1 / (6 * np.pi ** 2)) * pk_int1)

        # g_m non-linear damping factor of the BAO signal
        g_m1 = sigma_v1 ** 2 * (1 - self.mi ** 2 + self.mi ** 2 * (1 + growth_rate1) ** 2)

        # “de-wiggled” power spectrum
        pk_dw1 = pk_values1 * np.exp(- g_m1 * k1 ** 2) + pk_nw1 * (1 - np.exp(- g_m1 * k1 ** 2))

        # Compute the integral of f(z) from a to b
        result_int = integrate_function1(self, f)

        # print("Integral of f(z) from", minkh_t, "to", maxkh_t, ":", result_int)



        # pairwise velocity dispersion modulate the strength of FoG effect from https://arxiv.org/pdf/2210.05705.pdf

        _, _, model_IM_IM, _, _, _, _ = calculate_PS(d_Aref, H_z_values, k1, result_int, h_z_values, Omega_HI, pk_dw1,
                     frequency_to_temperature, theta_pb, r1, growth_rate1, b_HI_e, H_0_e, sigma_8, factor, nu_obs)

        sigma = 0.1  # Assuming a constant observational uncertainty
        IM_IM_loglike = -0.5 * np.sum(
            np.dot((data - model_IM_IM), np.dot(Cov_IM_inv, (data - model_IM_IM)) / sigma ** 2))
        return IM_IM_loglike

    def log_likelihood_IM_g(self, theta, data, calculate_PS, Cov_IM_g_inv):
        b_HI_e, H_0_e = theta

        # Set parameters for matter power spectrum computation
        pars1 = camb.CAMBparams()
        pars1.set_cosmology(H0=H_0_e, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, omk=self.omk)
        pars1.InitPower.set_params(As=self.As, ns=self.ns)

        # Enable WantTransfer parameter
        pars1.WantTransfer = True

        # Set parameters for matter power spectrum computation
        pars1.set_matter_power(redshifts=self.z_values, kmax=self.max_k)  # Set redshifts and k range for matter power spectrum
        # Compute results
        results1 = camb.get_results(pars1)
        k1, z1, pk_values1 = results1.get_matter_power_spectrum(minkh=self.min_k, maxkh=self.max_k,
                                                                npoints=self.n_points)  # Get matter power spectrum
        ###CAREFUL THE FUNCTION get_matter_power_spectrum IN CAMB OUTPUT THE k in h/Mpc and P(k) in (Mpc/h)^3 BUT IN OUR CASE WE WANT Mpc^-1 Mpc^3

        sigma_8 = results1.get_sigma8()

        # Compute Hubble parameter values at each redshift
        H_z_values = results1.hubble_parameter(self.z_values)
        h_z_values = H_z_values / 100

        ### WE RESCALE THE K AND PS
        pk_values1 = pk_values1 / h_z_values ** 3
        k1 = k1 * h_z_values

        # Extract the angular diameter distance values
        d_A_values = results1.angular_diameter_distance(self.z_values)

        # Hubble constant at the reference cosmology from https://arxiv.org/pdf/2210.05705.pdf
        H0_fiducial1 = H_0_e

        # Compute H(z) at the reference cosmology
        H_ref1 = H0_fiducial1

        # Reshape the array which was (1,200)
        pk_values1 = pk_values1.reshape(-1)

        # To get the comoving radial distance
        background1 = camb.get_background(pars1)
        r1 = background1.comoving_radial_distance(self.z_values)

        # Apply Savitzky-Golay filtering for dewiggling (The idea is to remove the wiggles due to the BAO)
        window_length = 11  # Choose an appropriate window length
        poly_order = 3  # Choose an appropriate polynomial order
        pk_nw1 = savgol_filter(pk_values1, window_length, poly_order)

        # Compute growth rate using CAMB
        growth_rate1 = results1.get_redshift_evolution(self.z_values, self.max_k, ['growth'])
        # Extract growth rate values
        growth_rate1 = growth_rate1[0, :, 0]



        # INTEGRATION FUNCTION
        def integrate_function1(self, f):
            # Generate n equally spaced points between a and b
            n = self.n_points - 1
            x = np.linspace(self.min_k, self.max_k, n + 1)

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
        def trapezoidal_rule1(y, x):
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
        pk_int1 = trapezoidal_rule1(pk_values1, k1)

        # from https://www.aanda.org/articles/aa/full_html/2020/10/aa38071-20/aa38071-20.html

        # σ_v controls the strength of the non-linear damping of the BAO signal in all directions in three-dimensions
        sigma_v1 = np.sqrt((1 / (6 * np.pi ** 2)) * pk_int1)

        # g_m non-linear damping factor of the BAO signal
        g_m1 = sigma_v1 ** 2 * (1 - self.mi ** 2 + self.mi ** 2 * (1 + growth_rate1) ** 2)

        # “de-wiggled” power spectrum
        pk_dw1 = pk_values1 * np.exp(- g_m1 * k1 ** 2) + pk_nw1 * (1 - np.exp(- g_m1 * k1 ** 2))

        # Compute the integral of f(z) from a to b
        result_int = integrate_function1(self, f)

        # print("Integral of f(z) from", minkh_t, "to", maxkh_t, ":", result_int)

        # pairwise velocity dispersion modulate the strength of FoG effect from https://arxiv.org/pdf/2210.05705.pdf

        model_IM_g, _, _, _, _, _, _ = calculate_PS(d_Aref, H_z_values, k1, result_int, h_z_values, Omega_HI, pk_dw1,
                     frequency_to_temperature, theta_pb, r1, growth_rate1, b_HI_e, H_0_e, sigma_8, factor, nu_obs)
        sigma = 0.1  # Assuming a constant observational uncertainty

        IM_g_loglike = -0.5 * np.sum(
            np.dot((data - model_IM_g), np.dot(Cov_IM_g_inv, (data - model_IM_g)) / sigma ** 2))

        return IM_g_loglike

    def log_prior_IM_IM(self, theta):
        b_HI_e, H_O_e = theta
        # Set prior ranges for b_HI and sigma8
        b_HI_min, b_HI_max = 0.2, 2  # Adjusted range for b_HI based on physical considerations
        H_0_min, H_0_max = 50, 110

        # Flat priors for b_HI and sigma8 within specified ranges
        if b_HI_min < theta[0] < b_HI_max and H_0_min < theta[1] < H_0_max:
            return 0.0  # Prior probability is uniform within the specified ranges
        return -np.inf  # Return negative infinity for points outside the specified ranges

    def log_prior_IM_g(self, theta):
        b_HI_e, H_O_e = theta
        # Set prior ranges for b_HI and sigma8
        b_HI_min, b_HI_max = 0.2, 2  # Adjusted range for b_HI based on physical considerations
        H_0_min, H_0_max = 50, 110

        # Flat priors for b_HI and sigma8 within specified ranges
        if b_HI_min < theta[0] < b_HI_max and H_0_min < theta[1] < H_0_max:
            return 0.0  # Prior probability is uniform within the specified ranges
        return -np.inf  # Return negative infinity for points outside the specified ranges

    # Define the log posterior function
    def log_posterior_IM_IM(self, theta, data, log_prior_IM_IM, log_likelihood_IM_IM):
        lp = log_prior_IM_IM(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_IM_IM(theta, data)

    def log_posterior_IM_g(self, theta, data, log_prior_IM_g, log_likelihood_IM_g):
        lp = log_prior_IM_g(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_IM_g(theta, data)

    def run_mcmc_IM_IM(self, initial_guess, n_walkers, n_steps, n_burn, step_size, log_posterior_IM_IM, data_IM_IM):
        ndim = len(initial_guess)
        pos = initial_guess + step_size * np.random.randn(n_walkers, ndim)
        with Pool() as pool:
            sampler_IM_IM = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_IM_IM, args=(data_IM_IM,), backend=backend1, pool=pool)
            start = time.time()
            sampler_IM_IM.run_mcmc(initial_state=pos, nsteps=n_steps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        samples = sampler_IM_IM.get_chain(discard=n_burn, thin=15, flat=True)
        return samples


    def run_mcmc_IM_g(self, initial_guess, n_walkers, n_steps, n_burn, step_size, log_posterior_IM_g, data_IM_g):
        ndim = len(initial_guess)
        pos = initial_guess + step_size * np.random.randn(n_walkers, ndim)
        with Pool() as pool:
            sampler_IM_IM = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_IM_g, args=(data_IM_g,), backend=backend2, pool=pool)
            start = time.time()
            sampler_IM_IM.run_mcmc(initial_state=pos, nsteps=n_steps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        samples = sampler_IM_IM.get_chain(discard=n_burn, thin=15, flat=True)
        return samples


    def plot_corner(self, samples, labels):
        g = plots.get_subplot_plotter()
        g.triangle_plot(samples, labels, filled=True)


if __name__ == '__main__':
    c = 299792.458  # Speed of light in km/s
    kB = 1.380649e-23
    fid_z = 0  # sky fraction
    f_sky = 0.48
    # Observational time in hours
    t_tot = 10000
    # System temperature from https://arxiv.org/pdf/2009.06197.pdf to be changed
    T_sys = 30
    # Number of dishes, in the articles it was said N_d = 1, but in the annex N_d = 197
    N_d = 197
    # Rest-frame frequency of the 21 cm line in MHz
    nu_21cm_rest = 1420.40575
    z_values = [0.65]
    z_values = np.array(z_values)
    # VALUES OF μ_Θ FROM THE ARTICLE https://arxiv.org/pdf/2210.05705.pdf
    mi = 1
    # values for the galaxy bias depending on the redshift
    b_g = 1.5
    # Value for the error on the redshift from https://arxiv.org/pdf/2210.05705.pdf
    sigma = 0.001
    # Diameter of the telescope dish from https://arxiv.org/pdf/2210.05705.pdf
    d = 15

    # WE COMPUTE THE 'TRUE' POWER SPECTRUM, WHICH WILL ACT AS DATA IN OUR MCMC
    # Define cosmological parameters
    ombh2 = 0.0224  # Physical baryon density
    omch2 = 0.17  # Physical cold dark matter density
    H_0 = 67  # Hubble constant in km/s/Mpc
    As = 2.14e-9  # Amplitude of primordial scalaar perturbations
    ns = 0.96  # Scalar spectral index
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0  # Curvature parameter, set to 0 for flat universe
    survey_area = 10000  # Square degrees
    min_k = 0.01
    max_k = 0.2

    n_points = 20

    H0_fiducial = 67
    # cross-correlation coefficient
    r_HI_g = 1
    number_density = 3.43 * 10 ** (-4)  # number density in galaxies per cubic Mpc
    delta_mi = 0.02889


    my_MCMC = MCMC_functions(c, kB, z_values, fid_z, ombh2, omch2, H_0, As, ns, mnu, omk, f_sky, t_tot, T_sys, N_d, nu_21cm_rest, mi, b_g, sigma, survey_area, min_k, max_k, n_points, H0_fiducial,r_HI_g, d, number_density, delta_mi)

    d_Aref = my_MCMC.calculate_d_Aref()

    k, z, pk_values, results, pars = my_MCMC.CAMB_output()

    H_z_values, h_z_values, h_0 = my_MCMC.calculate_H(results)
    pk_values = my_MCMC.calculate_pk_rescaled(pk_values, h_z_values)
    pk_dw = my_MCMC.dewiggling_PS(pk_values)

    growth_rate = my_MCMC.calculate_growth_rate(results)
    #f = my_MCMC.calculatge_f(growth_rate, pk_values)
    f=1
    #result_int = my_MCMC.integrate_function(f, growth_rate, pk_values)
    result_int = 1

    Omega_HI = my_MCMC.calculate_Omega_HI()
    frequency_to_temperature = my_MCMC.frequency_to_temperature()
    l = my_MCMC.calculate_lambda()
    theta_pb = my_MCMC.calculate_theta_pb(l)
    r = my_MCMC.calculate_comoving_radial_distance(pars)
    b_HI = my_MCMC.calculate_b_HI()
    sigma_8 = my_MCMC.calculate_sigma_8(results)
    factor = my_MCMC.calculate_factor(k, H_z_values)
    nu_obs = my_MCMC.calculate_nu_obs()



    P_IM_g, P_gg, P_IM, P_IM_noise, P_IM_g_wt_T, P_IM_IM_wt_T, T_b_mean_mK = my_MCMC.calculate_PS(d_Aref, H_z_values, k, result_int, h_z_values, Omega_HI, pk_dw,
                     frequency_to_temperature, theta_pb, r, growth_rate, b_HI, H_0, sigma_8, factor, nu_obs)

    shot_noise = my_MCMC.shot_noise()

    np.random.seed(42)

    true_params = np.array([b_HI[0], H_0])
    print(true_params)

    ndim = 2  # Number of parameters
    n_walkers = 20  # Number of walkers
    n_steps = 2000  # Number of steps
    n_burn = 50
    initial_guess = [0.5, 60]
    step_size = [1e-3, 1e-2]

    directory = "chains"
    if not os.path.exists(directory):
        os.makedirs(directory)

    chains1 = os.path.join(directory, "CHAIN_1.h5")
    backend1 = emcee.backends.HDFBackend(chains1)
    backend1.reset(n_walkers, ndim)

    chains2 = os.path.join(directory, "CHAIN_2.h5")
    backend2 = emcee.backends.HDFBackend(chains2)
    backend2.reset(n_walkers, ndim)

    os.environ["OMP_NUM_THREADS"] = "1"

    log_spaced_k = my_MCMC.log_spaced_k()

    V_survey = my_MCMC.calculate_survey_volume(r)

    P_IM_g, P_gg, P_IM, P_IM_noise, P_IM_g_wt_T, P_IM_IM_wt_T, T_b_mean_mK = my_MCMC.dewiggling_PS(pk_values)

    fact = my_MCMC.calculate_cov_matrix_factor(log_spaced_k, V_survey)

    P_t_g, P_t_IM_IM, P_t_g_IM = my_MCMC.P_tilde(shot_noise, P_gg, P_IM, P_IM_g, P_IM_noise)

    Cov_gg_inv, Cov_IM_inv, Cov_IM_g_inv, snr_gg, snr_IM, snr_IM_g = my_MCMC.calculate_covariance_matrices(fact, P_t_g, P_t_IM_IM, P_t_g_IM)

    log_likelihood_IM_IM = my_MCMC.log_likelihood_IM_IM(theta, data = P_IM, calculate_PS, Cov_IM_inv)
    log_prior_IM_IM = my_MCMC.log_prior_IM_IM(theta)
    log_posterior_IM_IM = my_MCMC.log_posterior_IM_IM(theta, data = P_IM, log_prior_IM_IM, log_likelihood_IM_IM)


    samples_IM_IM = my_MCMC.run_mcmc_IM_IM(initial_guess, n_walkers, n_steps, n_burn, step_size, log_posterior_IM_IM, data_IM_IM)
