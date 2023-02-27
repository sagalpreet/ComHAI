import pandas as pd

df1 = pd.read_csv('raw/human_counts.csv', index_col=False)
df1.columns = [f'{column}_x' if column != 'filenames' else 'filenames' for column in df1.columns]

df2 = pd.read_csv('raw/model_probs.csv', index_col=False)
df2.columns = [f'{column}_y' if column != 'filenames' else 'filenames' for column in df2.columns]

df = pd.merge(df1, df2, on='filenames')

l = ['true_labels_y', '0_x', '1_x', '2_x', '3_x', '4_x', '5_x', '6_x', '7_x', '8_x', '9_x', '0_y', '1_y', '2_y', '3_y', '4_y', '5_y', '6_y', '7_y', '8_y', '9_y']

for column in l[::-1]:
    df.insert(0, column, df.pop(column))

for column in (set(df.columns) - set(l)):
    df = df.drop(column, axis=1)

df.to_csv('cnn_data.csv', index=False, header=False)