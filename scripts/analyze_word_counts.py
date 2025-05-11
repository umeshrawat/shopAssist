import pandas as pd
import numpy as np

# Read the parquet file
df = pd.read_parquet('data/inScopeMetadata_flattened.parquet')

# Function to count words in a cell
def count_words(text):
    if isinstance(text, (list, np.ndarray)):
        text = ' '.join(str(x) for x in text if not (isinstance(x, float) and np.isnan(x)))
        return len(str(text).split())
    if pd.isna(text):
        return 0
    return len(str(text).split())

output_lines = []
output_lines.append("\nWord count analysis per column:")
output_lines.append("=" * 50)
print("\nWord count analysis per column:")
print("=" * 50)

for column in df.columns:
    # Calculate word counts
    word_counts = df[column].apply(count_words)
    
    # Calculate statistics
    stats = {
        'mean': word_counts.mean(),
        'median': word_counts.median(),
        'min': word_counts.min(),
        'max': word_counts.max(),
        'std': word_counts.std(),
        'total_words': word_counts.sum(),
        'non_zero_cells': (word_counts > 0).sum(),
        'total_cells': len(word_counts)
    }
    
    col_stats = (
        f"\nColumn: {column}\n"
        f"Mean words per cell: {stats['mean']:.2f}\n"
        f"Median words per cell: {stats['median']:.2f}\n"
        f"Min words: {stats['min']}\n"
        f"Max words: {stats['max']}\n"
        f"Standard deviation: {stats['std']:.2f}\n"
        f"Total words in column: {stats['total_words']}\n"
        f"Cells with words: {stats['non_zero_cells']} out of {stats['total_cells']} ({(stats['non_zero_cells']/stats['total_cells']*100):.1f}%)\n"
    )
    print(col_stats)
    output_lines.append(col_stats)

# Print and save overall statistics
output_lines.append("\nOverall Statistics:")
output_lines.append("=" * 50)
print("\nOverall Statistics:")
print("=" * 50)
total_words = sum(df[col].apply(count_words).sum() for col in df.columns)
total_cells = df.size
overall_stats = (
    f"Total words across all columns: {total_words}\n"
    f"Total cells: {total_cells}\n"
    f"Average words per cell across all columns: {total_words/total_cells:.2f}\n"
)
print(overall_stats)
output_lines.append(overall_stats)

with open('word_count_stats.txt', 'w') as f:
    f.writelines(line if line.endswith('\n') else line + '\n' for line in output_lines) 