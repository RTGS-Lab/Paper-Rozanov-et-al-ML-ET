def NDVI():
    return
def EVI():
    return
def NBR():
    return
def ARVI():
    return
def SAVI():
    return

features['NDVI'] = (features['NIR']-features['Red'])/(features['NIR']+features['Red'])
features['EVI'] = (features['NIR']-features['Red'])/(features['NIR']+6*features['Red']-7.5*features['Blue']+1)
features['GNDVI'] = (features['NIR']-features['Green'])/(features['NIR']+features['Green'])
features['SAVI'] = (features['NIR']-features['Red'])/(features['NIR']+features['Red']+0.5)*1.5