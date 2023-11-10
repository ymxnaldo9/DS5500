import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
crash_data = pd.read_csv("imputed_Data.csv")
# Some basic statistical characteristics
'''
injury_distribution = crash_data['Injury Severity'].value_counts()
data = {'Injury Severity': ['NO APPARENT INJURY', 'POSSIBLE INJURY', 'SUSPECTED MINOR INJURY',
                            'SUSPECTED SERIOUS INJURY', 'FATAL INJURY', 'NONE DETECTED',
                            'MEDICATION PRESENT', 'UNKNOWN', 'ALCOHOL CONTRIBUTED'],
        'Count': [137253, 17011, 11633, 1382, 151, 46, 12, 2, 1]}
crash_data = pd.DataFrame(data)
value_counts = crash_data['Count']
labels = crash_data['Injury Severity']
plt.figure(figsize = (10, 6))  # 设置图形大小
plt.bar(labels, value_counts, color = 'blue')  # 使用 matplotlib 绘制直方图
plt.title('Distribution of Injury Severity')
plt.xlabel('Injury Severity')
plt.ylabel('Count')
plt.xticks(rotation=45)  # 旋转 x 轴标签以提高可读性
plt.show()
'''
'''
values_to_remove = ['NONE DETECTED', 'MEDICATION PRESENT', 'UNKNOWN', 'ALCOHOL CONTRIBUTED']
mask = ~crash_data['Injury Severity'].isin(values_to_remove)
crash_data = crash_data[mask]
crash_data = crash_data[crash_data['Driverless Vehicle'] == 'No']
crash_data = crash_data[crash_data['Parked Vehicle'] == 'No']
crash_data['Cross-Street Type'].replace("unknown", "County")
crash_data['Collision Type'].replace("UNKNOWN", "SAME DIR REAR END")
crash_data['Weather'].replace("UNKNOWN", "CLEAR")
crash_data['Surface Condition'].replace("UNKNOWN", "DRY")
crash_data['Light'].replace("UNKNOWN", "DAYLIGHT")
crash_data['Traffic Control'].replace("UNKNOWN", "NO CONTROLS")
crash_data['Driver Substance Abuse'].replace("UNKNOWN", "NONE DETECTED")
random_choices = np.random.choice(['Yes', 'No'], size = len(crash_data))
crash_data['Driver At Fault'] = np.where(crash_data['Driver At Fault'] == 'Unknown', random_choices, crash_data['Driver At Fault'])
crash_data['Driver Distracted By'].replace("UNKNOWN", "NOT DISTRACTED")
crash_data['Vehicle Damage Extent'].replace("UNKNOWN", "DISABLING")
clock = {1:"ONE OCLOCK", 2:"TWO OCLOCK", 3:"THREE OCLOCK", 4:"FOUR OCLOCK", 5:"FIVE OCLOCK", 6:"SIX OCLOCK", 7:"SEVEN OCLOCK", 8:"EIGHT OCLOCK", 9:"NINE OCLOCK", 10:"TEN OCLOCK", 11:"ELEVEN OCLOCK", 12:"TWELVE OCLOCK"}
valid_locations = [
    'TWELVE OCLOCK',
    'SIX OCLOCK',
    'ELEVEN OCLOCK',
    'TWO OCLOCK',
    'TEN OCLOCK',
    'FIVE OCLOCK',
    'FOUR OCLOCK',
    'SEVEN OCLOCK',
    'EIGHT OCLOCK',
    'THREE OCLOCK',
    'NINE OCLOCK'
]
for index, location in crash_data['Vehicle First Impact Location'].items():
    if location not in valid_locations:
        random_value = random.randint(1, 12)
        crash_data.at[index, 'Vehicle First Impact Location'] = clock[random_value]
for index, location in crash_data['Vehicle Second Impact Location'].items():
    if location not in valid_locations:
        random_value = random.randint(1, 12)
        crash_data.at[index, 'Vehicle Second Impact Location'] = clock[random_value]
crash_data['Vehicle Body Type'].replace("UNKNOWN", "PASSENGER CAR")
crash_data['Vehicle Movement'].replace("UNKNOWN", "MOVING CONSTANT SPEED")
features = [
    'Cross-Street Type',
    'Collision Type',
    'Weather',
    'Surface Condition',
    'Light',
    'Traffic Control',
    'Driver Substance Abuse',
    'Driver At Fault',
    'Driver Distracted By',
    'Vehicle Damage Extent',
    'Vehicle First Impact Location',
    'Vehicle Second Impact Location',
    'Vehicle Body Type',
    'Vehicle Movement',
    'Speed Limit',
    'Injury Severity'
]
subset = crash_data[features]
label_encoder = LabelEncoder()
for column in subset.columns:
    subset[column] = label_encoder.fit_transform(subset[column])
subset.to_csv('crash_data_for_modeling.csv', index = False)
'''











