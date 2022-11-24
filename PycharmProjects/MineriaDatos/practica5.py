#5. prueba de hipotesis
from this import d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

df = pd.read_csv("dataset.csv",nrows=1000)

print("\n\tPruebas Estadísticas\n\n")
#print(housingPrices.head())

D = df['danceability']
V = df['valence']


print("Un valor de curtosis y/o coeficiente de asimetría entre -1 y 1, es generalmente considerada una ligera desviación de la normalidad")
print("Entre -2 y 2 hay una evidente desviación de la normal pero no extrema.")
print("\nRealizamos una curtosis al area de vivienda por venta")
print("\tbailabilidad: " + str(stats.kurtosis(D)))
print("\tvalencia: " + str(stats.kurtosis(V)))

print("\nRealizamos una asimetría a las distribuciones devalores")

print("\tbailabilidad: " + str(stats.skew(D)))
print("\tValencia: " + str(stats.skew(V)))

fig, [ax1, ax2] = plt.subplots(1,2)
ax1.violinplot(D)
ax2.violinplot(V)
fig.suptitle("Distribuciones de valor de valencia y bailabilidad")
ax1.set_xlabel('bailabilidad')
ax2.set_xlabel('valencia')
fig.savefig("Practica5.png")
plt.close()

