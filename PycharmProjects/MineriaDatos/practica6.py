#6.- Regresion Lineal

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("dataset.csv",nrows=1000)
df_limp = pd.DataFrame.from_dict(df.loc[:,['popularity','track_name','artists','album_name','explicit','duration_ms','key','loudness','track_genre','valence']])
print(df_limp)

X = df['valence'].values.reshape(-1,1)
Y = df['popularity']

linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('valor de valencia')
plt.ylabel('popularidad')
plt.title('Regresion lineal')
plt.show()
plt.close()

df_lr = pd.DataFrame({'valence': df['valence'], 'popularity': df['popularity']})
df_lr_plot_scatter = df_lr.plot.scatter(x = 'valence', y = 'popularity')
fig = df_lr_plot_scatter.get_figure()
plt.xlabel('valor de valencia')
plt.ylabel('popularidad')
plt.title('valores')
plt.show()
plt.close()