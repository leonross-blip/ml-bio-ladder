import pandas as pd
import numpy as np

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(URL)
print(df.shape)
print(df.head())

small = df.dropna(subset=["species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"])
species = small["species"]
X = small[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]].to_numpy()
Z = (X - X.mean(axis=0))/X.std(axis=0)
print(Z.shape)
print(Z.mean(axis=0).round(3))
print(Z.std(axis=0).round(3))

