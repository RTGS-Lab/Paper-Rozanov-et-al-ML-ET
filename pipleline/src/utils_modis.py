import numpy as np
import pandas as pd
from PM_eq import penman_monteith
import ee

ee.Authenticate()
ee.Initialize(project='ee-demo-soil')

bands = ['SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth',
         'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04',
         'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'state_1km']
mod_names = [
          'SensorZenith', 'SensorAzimuth', 'SolarZenith', 'SolarAzimuth',
         'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04',
         'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'Clouds'
]

def extract_modis_features(mod_features):
    df = pd.DataFrame(data=mod_features, columns=mod_names) 
    df['NDVI'] = (mod_features[:,1] - mod_features[:,0])/(mod_features[:,1] + mod_features[:,0])
    df['EVI'] = (mod_features[:,1] - mod_features[:,0])/(mod_features[:,1] + 6*mod_features[:,0] - 7.5*mod_features[:,2] + 1)
    df['GNDVI'] = (mod_features[:,1] - mod_features[:,3])/(mod_features[:,1] + mod_features[:,3])
    df['SAVI'] = (mod_features[:,1] - mod_features[:,0])/(mod_features[:,1] + mod_features[:,0] + 0.5)*1.5
    df['ARVI'] = (mod_features[:,1] + mod_features[:,2] - 2*mod_features[:,0])/(mod_features[:,1] - mod_features[:,2] + 2*mod_features[:,0])
    
    return df
    
def extract_location_data(feature):
        lon = feature.geometry().coordinates().get(0)
        lat = feature.geometry().coordinates().get(1)
        date = ee.Date(feature.get('date'))
        
        roi = ee.Geometry.Point([lon, lat])
        start_date = date
        end_date = date.advance(1, 'day')
        
        data = ee.ImageCollection("MODIS/061/MOD09GA") \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .select(bands)
        def reduce_image(image):
            qa = image.select(bands[-1])
            cloud = (qa.bitwiseAnd(1 << 0).neq(0)
                     .Or(qa.bitwiseAnd(1 << 1).neq(0))
                     .Or(qa.bitwiseAnd(1 << 2).neq(0)))
            image = image.addBands(cloud.rename(bands[-1]), overwrite=True)
            
            values = image.reduceRegion(
                reducer=ee.Reducer.first(), 
                geometry=roi.buffer(500), 
                scale=500
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
        for feature in features_info['features']:
            props = feature['properties']
            idx = props.get('index')
                
            values = [
                props.get(band) for band in bands 
                if band in props and isinstance(props.get(band), (int, float))
            ] 
            
            if values and len(values)==12:
                feature_values.append(values)
                feature_dates.append(props['image_date'])

        feature_values, feature_dates = np.array(feature_values), np.array(feature_dates)

        return feature_values, feature_dates
    
def extract_MODIS_chunk(chunk_fluxes):
    features = []
    for idx, row in chunk_fluxes.iterrows():
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
    final_result = extract_modis_features(result_array)
    return final_result, dates


