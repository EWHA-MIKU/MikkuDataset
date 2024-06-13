import pandas as pd

df = pd.read_csv("/mnt/aix21006/data/figure_extract/data/extract_figure_final/extracted_figures_real.csv", encoding="utf-8")

df = df[['ko', 'per_list_real']]

df['per_list_real'] = df['per_list_real'].str.split(', ')
df_exploded = df.explode('per_list_real')

df_exploded.columns = ['ko', 'figure']

df_exploded = df_exploded.dropna(subset=['figure'])

df_exploded=df_exploded.reset_index(drop=True)

print(df_exploded)

df_exploded.to_csv("prepro_gender_labeling.csv", index=False, encoding="utf-8")
