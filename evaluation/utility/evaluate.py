import os
import pandas as pd
import json
from sdmetrics.reports.single_table import QualityReport
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Load metadata from a JSON file
with open('', 'r') as metadata_file:
    metadata = json.load(metadata_file)

# Load real dataset
real_data = pd.read_csv('')

# Load synthetic datasets
synthetic_datasets = {
    "privbayes": pd.read_csv(''),
    "mst": pd.read_csv(''),
    "mwem+pgm": pd.read_csv(''),
    "fem": pd.read_csv(''),
    "aim": pd.read_csv(''),
    "dpctgan": pd.read_csv(''),
    "dpgan": pd.read_csv(''),
    "patectgan": pd.read_csv(''),
    "pategan": pd.read_csv('')
}

# Define the output directory and file path
output_dir = r''
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'score.txt')

data = {
    "Generator": [],
    "Overall Score": [],
    "Column Shapes Score": [],
    "Column Pair Trends Score": []
}

# Open the text file for writing
with open(output_file, 'w') as file:
    file.write("Synthetic Dataset Scores:\n")
    file.write("=" * 50 + "\n")

    # Process each dataset and write scores
    for name, synthetic_data in synthetic_datasets.items():
        report = QualityReport()
        report.generate(real_data, synthetic_data, metadata)

        # Get scores and round to 2 decimal places
        overall_score = round(report.get_score(), 2)
        column_shapes_score = round(report.get_details(property_name='Column Shapes')['Score'].mean(), 2)
        column_pair_trends_score = round(report.get_details(property_name='Column Pair Trends')['Score'].mean(), 2)

        # Append scores to the data dictionary
        data["Generator"].append(name)
        data["Overall Score"].append(overall_score)
        data["Column Shapes Score"].append(column_shapes_score)
        data["Column Pair Trends Score"].append(column_pair_trends_score)

        # Write to the text file
        file.write(f"Dataset: {name}\n")
        file.write(f"  Overall Score: {overall_score:.2f}\n")
        file.write(f"  Column Shapes Score: {column_shapes_score:.2f}\n")
        file.write(f"  Column Pair Trends Score: {column_pair_trends_score:.2f}\n")
        file.write("-" * 50 + "\n")

# Create a DataFrame from the scores
df = pd.DataFrame(data)

# Transpose the DataFrame so that each score is a row, and generators are columns
df_transposed = df.set_index('Generator').transpose()

# Define output directory for images
tables_dir = os.path.join(output_dir, "tables")
os.makedirs(tables_dir, exist_ok=True)

def save_heatmap_table(df, output_path):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 5)) 
    ax.axis('off')  # Turn off the axis

    # Normalize values for coloring
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = LinearSegmentedColormap.from_list(
        "extended_score_cmap", 
        ["darkblue", "blue", "lightblue", "white", "yellow", "orange", "red"] 
    )

    # Create the table with cell coloring
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index, 
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    for (i, j), cell in table.get_celld().items():
        if i == 0: 
            cell.set_text_props(fontsize=12)
        else:
            cell.set_text_props(fontsize=12)

    # Apply heatmap-like coloring to the cells with transparency
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        if key[0] > 0:
            value = df.values[key[0] - 1, key[1]]
            try:
                value = float(value)
            except ValueError:
                value = 0.0

            color = cmap(norm(value))
            transparency = 0.6 
            color = (color[0], color[1], color[2], transparency)
            cell.set_facecolor(color)

        cell.set_edgecolor("black")
        cell.set_linewidth(1)

    for i in range(len(df.index)):
        row_label_cell = table[(i + 1, -1)]
        row_label_cell.set_facecolor("white")

    # Save the figure as an image
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



# Sort each row by score (from highest to lowest)
sorted_overall_score = df_transposed.loc[['Overall Score']].sort_values(by='Overall Score', axis=1, ascending=False)
sorted_column_shapes_score = df_transposed.loc[['Column Shapes Score']].sort_values(by='Column Shapes Score', axis=1, ascending=False)
sorted_column_pair_trends_score = df_transposed.loc[['Column Pair Trends Score']].sort_values(by='Column Pair Trends Score', axis=1, ascending=False)

# Save the sorted tables
save_heatmap_table(
    sorted_overall_score,
    os.path.join(tables_dir, "overall_score.png")
)

save_heatmap_table(
    sorted_column_shapes_score,
    os.path.join(tables_dir, "column_shapes_score.png")
)

save_heatmap_table(
    sorted_column_pair_trends_score, 
    os.path.join(tables_dir, "column_pair_trends.png")
)

# Create DataFrame
combined_df = pd.DataFrame(data)
sorted_combined_df = combined_df.sort_values(by='Overall Score', ascending=False)
sorted_combined_df_transposed = sorted_combined_df.set_index('Generator').transpose()
combined_table_output_path = os.path.join(tables_dir, "combined_scores.png")
save_heatmap_table
(
    sorted_combined_df_transposed, 
    combined_table_output_path
)
