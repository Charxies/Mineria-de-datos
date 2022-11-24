import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

df= pd.read_csv("dataset.csv",nrows=1000)


#   Formula de distancia entre dos puntos, como el eje 'y' es muy pequeño, redondeamos a el rango de 'x'
def ClassificationFunc(position):
    radioA = 150
    radioB = 250

    distancia = np.sqrt(df['danceability'][position] ** 2 + df['energy'][position] ** 2)

    if distancia <= radioA:
        return "Grupo A"
    elif distancia <= radioB and distancia > radioA:
        return "Grupo B"
    elif distancia > radioB:
        return "Grupo C"

    return "GrupoC"


#   Grafica sin clasificación
plt.scatter(df['danceability'],df['energy'])
plt.title("Relationship between danceability and energy indexes")
plt.xlabel("indice de bailabilidad")
plt.ylabel("indice de energia")
plt.savefig("p8_sin_clasif.png")
plt.close()

df = df.reset_index()
df['group'] = df['index'].transform(ClassificationFunc)


# print(housingPrices)

#   Grafica con clasificación
def scatterClassification(file_path, df, x_column, y_column, label_column):
    colors = ["blue", "orange", "red"]
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])

    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=colors[i])

    ax.set_title("Relationship between danceability and energy indexes")
    ax.set_xlabel("indice de bailabilidad")
    ax.set_ylabel("indice de energia")
    plt.savefig(file_path)
    plt.close()


def euclidean_distance(p_1, p_2) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neightbors(points, labels, input_data, k):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    return [
        mode([labels[index] for index in point_nearest])
        for point_nearest in points_k_nearest
    ]

scatterClassification("Relationship dance-energy with Classification", df, "danceability", "energy", "group")

daf = pd.DataFrame()
daf['x'] = df['danceability']
daf['y'] = df['energy']
daf['label'] = df['group']
# print(df)


list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]

points = [point for point, _ in list_t]
labels = [label for _, label in list_t]

kn = k_nearest_neightbors(
    points,
    labels,
    [np.array([5, 100]), np.array([8, 200]), np.array([10, 300]), np.array([12, 400])],
    5,
)
print(kn)