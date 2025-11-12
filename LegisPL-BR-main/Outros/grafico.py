import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_excel('analisados.xlsx')

# Convert the 'data' column to datetime objects
df['data'] = pd.to_datetime(df['data'])

# Define the columns containing the number of deputies
deputy_cols = ['qtd_esquerda', 'qtd_centrao', 'qtd_direita']

# Calculate the total number of deputies for each month
df['total_deputados'] = df[deputy_cols].sum(axis=1)

# Calculate the percentage for each bloc
for col in deputy_cols:
    df[col] = (df[col] / df['total_deputados']) * 100

# Melt the DataFrame to have one row per observation for easy plotting
df_melted = df.melt(id_vars=['data'], value_vars=deputy_cols, var_name='bloco_partidario', value_name='percentual_deputados')

# Map the original column names to the desired bloc names in English
bloc_mapping = {
    'qtd_esquerda': 'Left',
    'qtd_centrao': 'Big Center',
    'qtd_direita': 'Right'
}
df_melted['bloco_partidario'] = df_melted['bloco_partidario'].map(bloc_mapping)

# Define the color mapping for the plot
colors = {'Left': 'red', 'Big Center': 'blue', 'Right': 'green'}

# Create the line plot
plt.figure(figsize=(12, 8))
for bloc, color in colors.items():
    bloc_data = df_melted[df_melted['bloco_partidario'] == bloc]
    plt.plot(bloc_data['data'], bloc_data['percentual_deputados'], color=color, label=bloc, linewidth=2)

# Set the font sizes for the labels, ticks, and legend
font_size_labels = 20
font_size_ticks = 16
font_size_legend_title = 20
font_size_legend_labels = 18

plt.xlabel('Date', fontsize=font_size_labels)
plt.ylabel('Percentage of Deputies (%)', fontsize=font_size_labels)
plt.xticks(fontsize=font_size_ticks)
plt.yticks(fontsize=font_size_ticks)
plt.legend(title='Party Bloc', title_fontsize=font_size_legend_title, fontsize=font_size_legend_labels)

# Improve layout and save the figure
plt.grid(True)
plt.tight_layout()
plt.savefig('evolution_percentage_deputies_blocs_english_updated.png')

print("Gráfico gerado com sucesso e salvo como 'evolution_percentage_deputies_blocs_english_updated.png'.")