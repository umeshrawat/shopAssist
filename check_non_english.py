import pandas as pd
from langdetect import detect, LangDetectException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the parquet file
df = pd.read_parquet('data/inScopeMetadata_with_embeddings.parquet')

# Columns used for embeddings
columns = ['item_name_flat', 'brand_flat', 'product_type_flat', 'bullet_point_flat', 'item_keywords_flat', 'node_flat']

# Function to detect language
def detect_language(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return 'unknown'

# Check each column for non-English text
for column in columns:
    if column in df.columns:
        logging.info(f"Processing column: {column}")
        df[f'{column}_language'] = df[column].apply(detect_language)
        non_english = df[df[f'{column}_language'] != 'en']
        if not non_english.empty:
            logging.info(f"Non-English entries in {column}:")
            print(non_english[[column, f'{column}_language']].head())
    else:
        logging.warning(f"Column {column} not found in the DataFrame.") 