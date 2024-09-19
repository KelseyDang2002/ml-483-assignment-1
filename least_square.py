import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('wine_data.csv', header=[0,1])

df.head()

print(df)
