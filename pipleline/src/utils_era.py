import numpy as np
import pandas as pd
from .PM_eq import penman_monteith
import ee

ee.Authenticate()
ee.Initialize(project='ee-demo-soil')

bands = [
    'temperature_2m',
    'dewpoint_temperature_2m',
    'u_component_of_wind_10m',
    'v_component_of_wind_10m',
    'surface_net_solar_radiation_sum',
    'total_evaporation_sum',
    'surface_pressure',
    'total_precipitation_sum',
]
def extract_era_features(features):
    pm_era = penman_monteith(features, None, None, mode='era5') 
    features = np.concatenate([features, pm_era[:, :, np.newaxis]], axis=2)
    df = pd.DataFrame(data=features[:,-1,:], columns=bands + ['LE_PM']) 
    for idx, col in enumerate(bands+['LE_PM']): # TS are only for the ERA5-derived variables
        if col in ['surface_net_solar_radiation_sum','total_evaporation_sum','total_precipitation_sum',]:
            ewm = pd.DataFrame(features[:,:,idx]).T.rolling(window=30).sum().iloc[[-1]].values
            df[col+'_rol_30'] = ewm.T
            ewm_7 = pd.DataFrame(features[:,:,idx]).T.rolling(window=7).sum().iloc[[-1]].values
            df[col+'_rol_7'] = ewm_7.T
        else:
          ewm = pd.DataFrame(features[:,:,idx]).T.rolling(window=30, min_periods=1).mean().iloc[[-1]].values#.ewm(span=3, adjust=False).mean()
          df[col+'_rol_30'] = ewm.T
          ewm_7 = pd.DataFrame(features[:,:,idx]).T.rolling(window=7, min_periods=1).mean().iloc[[-1]].values#.ewm(span=3, adjust=False).mean()
          df[col+'_rol_7'] = ewm_7.T
        Min = pd.DataFrame(features[:,:,idx]).T.min(axis=0).values#.ewm(span=3, adjust=False).mean()
        df[col+'_min'] = Min
        Max = pd.DataFrame(features[:,:,idx]).T.max(axis=0).values#.ewm(span=3, adjust=False).mean()
        df[col+'_max'] = Max
        std = pd.DataFrame(features[:,:,idx]).T.std(axis=0).values#.ewm(span=3, adjust=False).mean()
        df[col+'_std'] = std
    return df
    
def extract_location_data(feature):
        lon = feature.geometry().coordinates().get(0)
        lat = feature.geometry().coordinates().get(1)
        date = ee.Date(feature.get('date'))
        
        roi = ee.Geometry.Point([lon, lat])
        start_date = date.advance(-29, 'day')
        end_date = date.advance(1, 'day')
        
        data = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .select(bands)

        def reduce_image(image):
            values = image.reduceRegion(
                reducer=ee.Reducer.first(), 
                geometry=roi.buffer(11132), 
                scale=11132
            )
            return ee.Feature(None, values).set({
                'index': feature.get('index'),
                'image_date': image.date().format('YYYY-MM-dd')
            })
        reduced = data.map(reduce_image)
        return reduced

def convert_to_numpy(features_info):
        feature_values, feature_dates = [], []
        ts_features, ts_dates = [], []
        old_idx = 0
        for feature in features_info['features']:
            props = feature['properties']
            new_idx = props.get('index')
                
            values = [
                props.get(band) for band in bands 
                if band in props and isinstance(props.get(band), (int, float))
            ] 
            
            if values and len(values)>0:
                if new_idx==old_idx:
                    ts_features.append(values)
                    ts_dates.append(props['image_date'])
                else:
                    if len(ts_features)>0:
                        feature_values.append(ts_features)
                        feature_dates.append(ts_dates)
                    
                    ts_features, ts_dates = [], [] #lists to store 30,features arrays
                    ts_features.append(values)
                    ts_dates.append(props['image_date'])
            old_idx = new_idx
        feature_values.append(ts_features)
        feature_dates.append(ts_dates)
        feature_values = np.array(feature_values)
        feature_dates = np.array(feature_dates)
        return feature_values, feature_dates
    
def extract_era5_chunk(grid_chunck):
    features = []
    for idx, row in grid_chunck.iterrows():
        point = ee.Geometry.Point([row['lon'], row['lat']])
        feature = ee.Feature(point, {
            'date': row['date'],
            'index': idx
        })
        features.append(feature)
    
    locations = ee.FeatureCollection(features)
    
    nested_collections = locations.map(extract_location_data)
    flattened = ee.FeatureCollection(nested_collections).flatten()

    result = flattened.getInfo()
    result_array, dates = convert_to_numpy(result)
    final_result = extract_era_features(result_array)
    return final_result, dates