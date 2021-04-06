import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

dir_base = '/Users/hrmd_work/Documents/research/stoneley/'
dir_input = os.path.join(dir_base, 'input', 'Magrathea', 'llsvp')

def remove_micro_discon(r, x, tolerance = 1.0E-10, thresh = 2.0E-1):

    tolerance = 1.0E-10
    abs_diff_r =    np.abs(np.diff(r))
    abs_diff_x =  np.abs(np.diff(x))
    condition_discon_r   = (abs_diff_r < tolerance)
    condition_discon_x = (abs_diff_x > tolerance)
    i_discon = np.where((condition_discon_r) & (condition_discon_x))[0] 
    #
    i_micro_discon = i_discon[np.where(abs_diff_x[i_discon] < thresh)[0]]

    #
    for i in i_micro_discon:
        

        x_lower = x[i]
        x_upper = x[i + 1]

        x_mean = (x_lower + x_upper)/2.0

        x[i] = x_mean
        x[i + 1] = x_mean

    return x

def load_radial_model(dir_input, name = 'prem_noocean', period = None, remove_crust = False):

    # Load the model.
    path_model = os.path.join(dir_input, '{:}.txt'.format(name))
    data = np.loadtxt(path_model, skiprows = 3)

    # Unpack the data (all in SI units).
    r       = data[:, 0] # Radial coordinate (m).
    rho     = data[:, 1] # Density (kg/m3).
    v_pv    = data[:, 2] # Vertically-polarised P-wave speed (m/s).
    v_sv    = data[:, 3] # Vertically-polarised S-wave speed (m/s).
    q_ka    = data[:, 4] # Inverse Q-factor for kappa (dimensionless).
    q_mu    = data[:, 5] # Inverse Q-factor for mu (dimensionless).
    v_ph    = data[:, 6] # Horizontally-polarised P-wave speed (m/s).
    v_sh    = data[:, 7] # Horizontally-polarised S-wave speed (m/s).
    eta     = data[:, 8] # Anisotropy parameter (dimensionless).

    # Calculate Q-factors.
    Q_ka = 1.0/q_ka
    Q_mu = 1.0/q_mu

    # Calculate the Love (1927) transverse anisotropy parameters. 
    A       = rho*(v_ph**2.0)
    C       = rho*(v_pv**2.0)
    N       = rho*(v_sh**2.0)
    L       = rho*(v_sv**2.0)
    F       = (A - (2.0*L))*eta

    # Calculate the Voigt average bulk and shear moduli (in SI units: Pa).
    ka = ((4.0*A) + C + (4.0*F) - (4.0*N))/9.0
    mu = (A + C - (2.0*F) + (5.0*N) + (6.0*L))/15.0

    # Convert to Voigt average v_p and v_s (in SI units: m/s).
    v_p = np.sqrt((ka + ((4.0/3.0)*mu))/rho)
    v_s = np.sqrt(mu/rho)

    # Remove micro-discontinuities in PREM P- and S-wave speed.
    v_p = remove_micro_discon(r, v_p, thresh = 5.0E-1)
    v_s = remove_micro_discon(r, v_s)

    if remove_crust:
        
        n_pts = len(r)
        i_crust = list(range(157, 185))
        n_pts_crust = len(i_crust)
        n_pts_not_crust = n_pts - n_pts_crust
        n_pts_without_crust = n_pts_not_crust + 1

        r_new = np.zeros(n_pts_without_crust)
        v_s_new = np.zeros(n_pts_without_crust)
        v_p_new = np.zeros(n_pts_without_crust)
        rho_new = np.zeros(n_pts_without_crust)
        q_ka_new = np.zeros(n_pts_without_crust)
        q_mu_new = np.zeros(n_pts_without_crust)
        
        r_new  [0 : n_pts_not_crust] =   r[0 : n_pts_not_crust] 
        v_s_new[0 : n_pts_not_crust] = v_s[0 : n_pts_not_crust] 
        v_p_new[0 : n_pts_not_crust] = v_p[0 : n_pts_not_crust] 
        rho_new[0 : n_pts_not_crust] = rho[0 : n_pts_not_crust] 
        q_ka_new[0 : n_pts_not_crust] = q_ka[0 : n_pts_not_crust] 
        q_mu_new[0 : n_pts_not_crust] = q_mu[0 : n_pts_not_crust] 

        r_new[-1]   = r[-1]
        v_s_new[-1] = v_s_new[-2]
        v_p_new[-1] = v_p_new[-2]
        rho_new[-1] = rho_new[-2]
        q_ka_new[-1] = q_ka_new[-2]
        q_mu_new[-1] = q_mu_new[-2]

        r   = r_new
        v_s = v_s_new
        v_p = v_p_new
        rho = rho_new
        q_ka = q_ka_new
        q_mu = q_mu_new

    # Find fluid regions.
    cond_fluid = (v_s < 1.0E-11)
    i_fluid = np.where(cond_fluid)[0]
    i_solid = np.where(~cond_fluid)[0]

    # Fit quadratic to the upper part of the Q-mu model.
    q_220 = q_mu[-10] 
    q_0     = q_mu[-1]
    q_fit_func = lambda r, k: q_220 + (q_0 - q_220)*k*(((r - (6371.0 - 220.0))/220.0)**2.0)

    k_fit, _ = curve_fit(q_fit_func, 1.0E-3*r[-10:], q_mu[-10:], p0 = 1.0)
    
    q_mu[-10:] = q_fit_func(1.0E-3*r[-10:], k_fit)
    Q_mu = 1.0/q_mu

    # Apply a frequency correction.
    if period is not None:
        
        E =  (4.0/3.0)*((v_s/v_p)**2.0)

        v_p[i_fluid] = v_p[i_fluid]*(1.0 - ((1.0/np.pi)*(np.log(period))*(((1.0 - E[i_fluid])*Q_ka[i_fluid]))))
        v_p[i_solid] = v_p[i_solid]*(1.0 - ((1.0/np.pi)*(np.log(period))*(((1.0 - E[i_solid])*Q_ka[i_solid]) + (E[i_solid]*Q_mu[i_solid])))) 
        
        v_s[i_fluid] = 0.0
        v_s[i_solid] = v_s[i_solid]*(1.0 - ((1.0/np.pi)*(np.log(period))*Q_mu[i_solid]))



    # Convert to units used in NormalModes.
    r       = 1.0E-3*r
    rho     = 1.0E-3*rho
    #v_pv    = 1.0E-3*v_pv
    #v_sv    = 1.0E-3*v_sv
    #v_ph    = 1.0E-3*v_ph
    #v_sh    = 1.0E-3*v_sh
    v_p     = 1.0E-3*v_p
    v_s     = 1.0E-3*v_s

    # Find fluid regions.
    cond_fluid = (v_s < 1.0E-11)
    i_fluid = np.where(cond_fluid)[0]
    i_solid = np.where(~cond_fluid)[0]

    # Find the indices of the fluid outer core.
    i_cmb = i_fluid[-1] + 1 # Index of lowermost layer in mantle.
    i_icb = i_fluid[0] # Index of lowermost layer in outer core.
    #
    n_layers = len(r)
    i_inner_core = np.array(list(range(0, i_icb)), dtype = np.int)
    i_outer_core = np.array(list(range(i_icb, i_cmb)), dtype = np.int)
    i_mantle = np.array(list(range(i_cmb, n_layers)), dtype = np.int)

    # Store in a dictionary.
    model = dict()
    model['r']      = r
    model['rho']    = rho
    #model['v_pv']    = v_pv
    #model['v_sv']    = v_sv
    #model['v_ph']    = v_ph
    #model['v_sh']    = v_sh
    model['v_p']    = v_p
    model['v_s']    = v_s
    model['q_ka']   = q_ka
    model['q_mu']   = q_mu

    #return r, v_p, v_s, rho, i_icb, i_cmb, i_inner_core, i_outer_core, i_mantle
    return model

def main():
    
    #f_min = 0.1
    #f_max = 1.0
    #f_mid = (f_min + f_max)/2.0
    #f_mid_Hz = 1.0E-3*f_mid
    #period = 1.0/f_mid_Hz

    remove_crust = True
    if remove_crust:

        #crust_str = 'no_crust'
        crust_str = 'no_80km'


    else:

        crust_str = 'with_crust'

    f0 = 1000.0 # mHz 
    T0 = 1.0E3/f0 # s
    model_0 = load_radial_model(dir_input, period = T0, remove_crust = remove_crust)

    f1 = 3.0 # mHz 
    T1 = 1.0E3/f1 # s
    model_1 = load_radial_model(dir_input, period = T1, remove_crust = remove_crust)
    
    # Save the array.

    out_arr = np.array([model_1['r'], model_1['rho'], model_1['v_p'], model_1['v_s']])
    name_out = 'prem_{:}_{:>04.1f}.txt'.format(crust_str, f1)
    path_out = os.path.join(dir_input, name_out) 
    #print('Saving to {:}'.format(path_out))
    #np.savetxt(path_out, out_arr.T) 

    keys = ['v_p', 'v_s', 'rho', 'q_ka', 'q_mu']
    i_discon = dict()
    r_diffs = np.abs(np.diff(model_0['r']))
    tolerance = 1.0E-10
    condition_discon_r = (r_diffs < tolerance)
    #key_t = 'v_p'
    #for key in [key_t]:
    for key in keys:
        
        abs_diff = np.abs(np.diff(model_0[key]))
        condition_discon_key = (abs_diff > tolerance)
        
        i_discon[key] = np.where((condition_discon_r) & (condition_discon_key))[0] 

    i_discon_merge = []
    for key in keys:

        i_discon_merge = i_discon_merge + list(i_discon[key])

    i_discon_merge = np.sort(np.unique(i_discon_merge)) 

    frac_diffs = dict()
    for key in ['v_p', 'v_s']:

        frac_diffs[key] = (model_1[key] - model_0[key])/model_0[key]

    frac_diffs['v_s'][model_0['v_s'] < 1.0E-10] = 0.0
    
    i_not_crust = np.where(model_0['r'] < (6371.0 - 30.0))
    i_crust = np.where(model_0['r'] > (6371.0 - 30.0))
    for key in ['v_p', 'v_s']:
        
        frac_diffs[key][i_crust] = 0
        abs_frac_diffs = np.abs(frac_diffs[key])
        i_max_frac_diff = np.argmax(abs_frac_diffs)
        print(abs_frac_diffs[i_max_frac_diff])

    upper_mantle = True 

    fig, ax_arr = plt.subplots(1, 5, sharey = True, figsize = (11.0, 8.5))
    
    labels = [  '$\\alpha_{iso}$ (km s$^{-1}$)',
                '$\\beta_{iso}$ (km s$^{-1}$)',
                '$\\rho$ (kg m$^{-3})$',
                '1/Q$_\\kappa$',
                '1/Q$_\\mu$']
    
    if upper_mantle:
        upper_lims = [12.0, 6.8, 5.0, 61000.0, 700.0]
        y_lims = [800.0, model_0['r'][0]] # Inverted axis.
        fig_name = 'model_upper_mantle_{:}'.format(crust_str)
    else:
        upper_lims = [14.0, 7.5, 14.0, 61000.0, 700.0]
        y_lims = [model_0['r'][-1], model_0['r'][0]] # Inverted axis.
        fig_name = 'model_whole_earth_{:}'.format(crust_str)

    colour_ref = 'b'
    colour_new = 'r'
    alpha = 0.8
    #linestyle_ref = ':'
    #linestyle_new = '-'
    linestyle = '-'
    font_size_label = 12
    plot_kwargs = {'linestyle' : linestyle, 'alpha' : alpha, 'marker' : '.'}
    hline_kwargs = {'color' : 'k', 'linestyle' : '-', 'alpha' : 0.5}
    for i in range(len(keys)):

        ax = ax_arr[i]
        
        key = keys[i]
        ax.plot(model_0[key], 6371.0 - model_0['r'], c = colour_ref, **plot_kwargs)
        ax.plot(model_1[key], 6371.0 - model_1['r'], c = colour_new, **plot_kwargs)
        
        for j in i_discon_merge:

            #ax.axhline(model_0['r'][j], **hline_kwargs)
            ax.axhline(6371.0 - model_0['r'][j], **hline_kwargs)

        ax.set_xlabel(labels[i], fontsize = font_size_label)
        ax.set_xlim([0.0, upper_lims[i]])

    ax_arr[-1].plot([80.0, 80.0, 600.0, 600.0], [220.0, 80.0, 80.0, 0.0]) 

    ax = ax_arr[0]
    ax.set_ylabel('Radius (km)', fontsize = font_size_label)
    #ax.set_ylim([model_0['r'][0], model_0['r'][-1]])
    ax.set_ylim(y_lims)
    ax.plot([], [], label = 'PREM\n(1 Hz)', c = colour_ref, **plot_kwargs)
    ax.plot([], [], label = 'New model\n(3 mHz)', c = colour_new, **plot_kwargs)
    ax.legend(loc= 'lower left')

    plt.tight_layout()
    
    path_fig = '{:}.png'.format(fig_name)
    print('Saving to {:}'.format(path_fig))
    plt.savefig(path_fig, dpi = 300)

    plt.show()

    #fig = plt.figure(figsize = (11.0, 8.5))
    #ax = plt.gca()
    
    #T0_string = '{:>.1f}'.format(T0)
    #ax.plot(model_0['v_p'], model_0['r'], c = 'r', label = 'Period = {:} s, $\\alpha$ (km s$^{{-1}}$)'.format(T0_string))
    #ax.plot(model_0['v_s'], model_0['r'], c = 'g', label = 'Period = {:} s, $\\beta$ (km s$^{{-1}}$)'.format(T0_string))
    #ax.plot(model_0['rho'], model_0['r'], c = 'b', label = 'Period = {:} s, $\\rho$ (kg m$^{{-3}}$)'.format(T0_string))

    #T1_string = '{:>.1f}'.format(T1)
    #ax.plot(model_1['v_p'], model_1['r'], c = 'r', ls = ':', label = 'Period = {:} s, $\\alpha$ (km s$^{{-1}}$)'.format(T1_string))
    #ax.plot(model_1['v_s'], model_1['r'], c = 'g', ls = ':', label = 'Period = {:} s, $\\beta$ (km s$^{{-1}}$)'.format(T1_string))
    #ax.plot(model_1['rho'], model_1['r'], c = 'b', ls = ':', label = 'Period = {:} s, $\\rho$ (kg m$^{{-3}}$)'.format(T1_string))

    #ax.legend()

    #plt.show()

    return

if __name__ == '__main__':

    main()
