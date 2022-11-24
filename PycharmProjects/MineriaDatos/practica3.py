import pandas as pd
#3. analisis de datos

df = pd.read_csv("dataset.csv")
df_limp = pd.DataFrame.from_dict(df.loc[:,['popularity','track_name','artists','album_name','explicit','duration_ms','key','loudness','track_genre','valence']])

#se consigue promedio,media, moda y mediana de duration_ms
mean_df = df_limp['duration_ms'].mean()
mean_df_float=float(mean_df)
mediana=float(df_limp["duration_ms"].median())
moda=float(df_limp["duration_ms"].mode())
print("""
    promedio de duracion de canciones: %d min (%s ms)
    mediana de duracion de canciones: %d min
    moda de duracion de canciones: %d min
    \n
 """ % (((mean_df_float/1000)/60),mean_df,((mediana/1000)/60),((moda/1000)/60) ))


#se consigue promedio, media, moda y mediana del valor de valence
mean_df = df_limp['valence'].mean()
mediana=df_limp["valence"].median()
moda=df_limp["valence"].mode()
print("""
    promedio de valor de valencia: %s
    mediana de valor de valencia: %s
    moda de valor de valencia: %s
    \n
 """ % (mean_df,mediana,moda))


