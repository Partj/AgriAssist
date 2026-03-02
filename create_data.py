import pandas as pd
import random

# 1. We define the columns we need for our farming data
columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Rainfall', 'Crop', 'Yield_kg_per_acre']
data = []

# 2. We define four basic crops for this initial test
crops = ['Wheat', 'Rice', 'Maize', 'Sugarcane']

# 3. We generate 1,000 rows of random but realistic farming data
for _ in range(1000):
    crop = random.choice(crops)
    nitrogen = random.randint(20, 120)
    phosphorus = random.randint(10, 60)
    potassium = random.randint(10, 60)
    temperature = round(random.uniform(15.0, 35.0), 1)
    rainfall = round(random.uniform(50.0, 300.0), 1)
    yield_kg = round(random.uniform(1000.0, 4500.0), 1)
    
    data.append([nitrogen, phosphorus, potassium, temperature, rainfall, crop, yield_kg])

# 4. We use pandas to convert this list into a table and save it as a CSV file
df = pd.DataFrame(data, columns=columns)
df.to_csv('crop_data.csv', index=False)

print("Success: 'crop_data.csv' has been created!")