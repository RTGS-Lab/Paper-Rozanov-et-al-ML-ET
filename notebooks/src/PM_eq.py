import numpy as np

def penman_monteith(inputs, target_idx, fluxes, mode='ground'):
    """
    Compute the Penman-Monteith equation.
    """
    
    ro_a = 1.225 #air density
    Cp = 1.013 * 10**-3  # Specific heat capacity of air at constant pressure (J/kg/K)
    epsilon = 0.622  # Ratio of molecular weight of water vapor to dry air
    gamma_constant = 0.665 # Psychrometric constant (kPa/K)
    r_s = 70 # Estimate surface resistance (r_s), assuming well-watered vegetation (~70 s/m)
    
    if mode == 'era5':
        temp = inputs[:, :, 0] - 273.15   # Temperature at 2m (K -> C)
        Tdew = inputs[:, :, 1] - 273.15  # Dewpoint temperature at 2m (K -> C)
        Rn = inputs[:, :, 4] / (60*60*24)  # Net solar radiation (W/m²)
        P = inputs[:, :, 6]/1000 # Surface pressure (Pa -> kPa)
        u10 = inputs[:, :, 2]  # Wind speed u-component (m/s)
        v10 = inputs[:, :, 3]  # Wind speed v-component (m/s)
        G = 0.1*Rn
        
        wind = np.sqrt(u10**2 + v10**2)

        e_s = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3 + 1e-9))
        e_a = 0.6108 * np.exp((17.27 * Tdew) / (Tdew + 237.3 + 1e-9))

    elif mode == 'ground':
        
        temp = fluxes.loc[target_idx, 'TA_F']   # Temperature at 2m (C)
        vpd = fluxes.loc[target_idx, 'VPD_F']*0.1 #Vapor Pressure Deficit (hPa -> kPa)
        Rn = fluxes.loc[target_idx, 'NETRAD']  # Net solar radiation (W/m²)
        G = fluxes.loc[target_idx, 'G_F_MDS']  #Soil heat flux (W/m²)
        P = fluxes.loc[target_idx, 'PA_F'] # Surface pressure (kPa)
        wind = fluxes.loc[target_idx, 'WS_F'] # Wind speed (m/s)
        
        e_s = 0.6108 * np.exp((17.27 * temp ) / (temp + 237.3 + 1e-9))  # Saturation vapor pressure (kPa)
        e_a = e_s - vpd  # Actual vapor pressure (Pa) from VPD (Vapor Pressure Deficit)

    
    delta = (4098 * e_s) / (temp + 273.15) ** 2 # Calculate the slope (Δ) of the saturation vapor pressure curve
        
    Lv = 2.501 - 0.00237 * temp #latent heat of vaporization # MJ/kg

    gamma = (P * Cp) / (Lv * epsilon) # Psychrometric constant (γ) kPa/K

    # Calculate the latent heat flux using Penman-Monteith equation
    numerator = 0.408 * delta * (Rn - G) + gamma * (
        900 / (temp + 273.0)
    ) * wind * (e_s - e_a)
    denominator = delta + gamma * (1 + 0.34 * wind)
    ET0 = numerator / denominator

    LE = ET0 * Lv * 1e6/(24*60*60)
    return LE 