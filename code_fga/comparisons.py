import pandas as pd
# val, cross 
jpeg_people = {'alkane': [0.90, 0.26], 
               'methyl': [0.84, 0.36], 
               'alkene': [0.68, 0.74], 
               'alkyne': [0.80, 0.33], 
               'alcohols': [0.84, 0.33], 
               'amines': [0.80, 0.49],
               'nitriles': [0.65, 0.67], 
               'aromatics': [0.89, 0.26], 
               'alkyl halides': [0.73, 0.53], 
               'esters': [0.83, 0.32], 
               'ketones': [0.76, 0.50],
               'aldehydes': [0.80, 0.39], 
               'carboxylic acids': [0.98, 0.08], 
               'ether': [0.81, 0.44], 
               'acyl halides': [0.98, 0.14], 
               'amides': [0.70, 0.9],
               'nitro': [0.89, 0.67]}

jpeg_people = pd.DataFrame.from_dict(jpeg_people, orient='index', columns=['accuracy', 'cross_entropy'])

# print(jpeg_people)

# F1 Metrics

# Molecular perfection score 

