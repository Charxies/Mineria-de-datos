import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset.csv",nrows=1000)
df_limp = pd.DataFrame.from_dict(df.loc[:,
                                 ['popularity', 'danceability', 'track_name', 'artists', 'album_name', 'explicit',
                                  'duration_ms', 'key', 'loudness', 'track_genre', 'valence', 'tempo']])


# Se crearan graficas de bell para representar los datos

# se genera una funcion para districucion de probabilidad

def distribprob(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
    return y_out


# generamos grafica para bailabilidad
distribucionX = np.arange(0, 1000, 1)
distribucionY = df_limp["valence"]
x_fill = np.arange(0, 1, 0.1)
y_fill = (x_fill)
plt.figure(figsize=(6, 6))
plt.plot(distribucionX, distribucionY, color='black')
plt.scatter(distribucionX, distribucionY, marker='o', s=25, color='blue')
plt.fill_between(x_fill, y_fill, 0, alpha=0.2, color='blue')
plt.show()
