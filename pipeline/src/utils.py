import numpy as np
import pandas as pd
from src.PM_eq import penman_monteith
import src.config as config

def compute_features(arrays, era_names, mod_names, var_names, doy, year, lats, lons):
    lons, lats = np.meshgrid(lons, lats, indexing='xy')
    lat, lon = lats.flatten(), lons.flatten()
    
    flattened = arrays.reshape((arrays.shape[0], arrays.shape[1], arrays.shape[2]*arrays.shape[3]))
    arrays_reshaped = np.swapaxes(np.swapaxes(flattened,0,2), 1,2)
    
    pm = penman_monteith(arrays_reshaped[:, :, :len(era_names)])
    
    Red, Blue, Green, NIR = arrays_reshaped[:,:,-7], arrays_reshaped[:,:,-5], arrays_reshaped[:,:,-4], arrays_reshaped[:,:,-6]
    Red, Blue, Green, NIR = Red*1e-4, Blue*1e-4, Green*1e-4, NIR*1e-4
    NDVI = (NIR - Red) / (NIR + Red)
    EVI = (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    GNDVI = (NIR - Green) / (NIR + Green)
    SAVI = (NIR - Red) / (NIR + Red) / (NIR + Red + 0.5)*1.5
    ARVI = (NIR - 2*Red + Blue) / (NIR + 2*Red + Blue)

    NDVI[np.isinf(NDVI)] = np.nan
    EVI[np.isinf(EVI)] = np.nan
    GNDVI[np.isinf(GNDVI)] = np.nan
    SAVI[np.isinf(SAVI)] = np.nan
    ARVI[np.isinf(ARVI)] = np.nan
    
    features = np.concatenate([arrays_reshaped, pm[:, :, np.newaxis],
                           NDVI[:,:, np.newaxis], EVI[:,:, np.newaxis], GNDVI[:,:, np.newaxis],
                           SAVI[:,:, np.newaxis], ARVI[:,:, np.newaxis]], axis=2)

    df = pd.DataFrame(data=features[:,-1,:], columns=var_names + ['LE_PM'] + ['NDVI', 'EVI', "GNDVI", 'SAVI', 'ARVI']) 
    samples, timesteps, variables = features.shape

    f = features.transpose(2, 0, 1) # (variables, samples, timesteps)

    results = {}
    for idx, col in enumerate(era_names+['LE_PM']):
        x = f[idx]  # (samples, timesteps)
        if col in ['surface_net_solar_radiation_sum','total_evaporation_sum','total_precipitation_sum']:
            rol_30 = np.apply_along_axis(lambda m: np.sum(m[-30:]), 1, x)
            rol_7 = np.apply_along_axis(lambda m: np.sum(m[-7:]), 1, x)
        else:
            rol_30 = np.apply_along_axis(lambda m: np.mean(m[-30:]), 1, x)
            rol_7 = np.apply_along_axis(lambda m: np.mean(m[-7:]), 1, x)
        results[col+'_rol_30'] = rol_30
        results[col+'_rol_7'] = rol_7
        results[col+'_min'] = x.min(axis=1)
        results[col+'_max'] = x.max(axis=1)
        results[col+'_std'] = x.std(axis=1)
    df = pd.DataFrame({**dict(zip(df.columns, df.values.T)), **results})
    df['doy'] = doy
    df['year'] = year
    df['lat'] = lat
    df['lon'] = lon
    df['IGBP'] = df.IGBP.astype('category')

    df = df.reindex(columns=config.ref)
    return df

def interpolate(arrays):
    result = arrays.copy()
    timesteps, vars_count, height, width = arrays.shape
    reshaped = result.reshape(timesteps, vars_count, -1)

    for var_idx in range(vars_count):
        data = reshaped[:, var_idx]  # (timesteps, locations)
        for loc_idx in range(data.shape[1]):
            series = data[:, loc_idx]
            if np.any(np.isnan(series)):
                valid = ~np.isnan(series)
                if np.sum(valid) >= 1:
                    indices = np.arange(timesteps)
                    if np.sum(valid) == 1:
                        reshaped[:, var_idx, loc_idx] = series[valid][0]
                    else:
                        reshaped[:, var_idx, loc_idx] = np.interp(
                               indices, indices[valid], series[valid], 
                               left=series[valid][0], right=series[valid][-1]
                           )

    return result.reshape(timesteps, vars_count, height, width)