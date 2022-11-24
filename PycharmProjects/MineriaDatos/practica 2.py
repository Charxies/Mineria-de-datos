import pandas as pd
#2. limpieza de csv
df = pd.read_csv("dataset.csv")
df_limp = pd.DataFrame.from_dict(df.loc[:,['popularity','track_name','artists','album_name','explicit','duration_ms','key','loudness','track_genre','valence']])
print(df_limp)