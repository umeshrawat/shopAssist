import pandas as pd
import random
from sentence_transformers import InputExample
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from retriever.generate_embeddings import add_embedding_input_column

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(parquet_path: str = 'data/inScopeMetadata_flattened.parquet') -> pd.DataFrame:
    """Load the flattened parquet file."""
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} products from flattened metadata")
        
        # Only create embedding_input if it doesn't exist
        if 'embedding_input' not in df.columns:
            df = add_embedding_input_column(df)
            logger.info("Created embedding_input column using add_embedding_input_column function")
        else:
            logger.info("embedding_input column already exists in the DataFrame")
        return df
    except Exception as e:
        logger.error(f"Error loading parquet file: {e}")
        return None

def generate_triplets(df, label_col='product_type_flat', text_col='embedding_input', max_triplets=10000):
    """
    Generate triplets for training using flattened metadata.
    
    Args:
        df: DataFrame with flattened product data
        label_col: Column to use for grouping products (default: 'product_type_flat')
        text_col: Column containing text to compare (default: 'embedding_input')
        max_triplets: Maximum number of triplets to generate (default: 10000)
    """
    # Filter out rows with empty product types
    df = df[df[label_col].notna() & (df[label_col] != '')]
    logger.info(f"Found {len(df)} products with valid product types")
    
    label_to_indices = {}
    for idx, row in df.iterrows():
        label = row[label_col]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    triplets = []
    all_labels = list(label_to_indices.keys())
    logger.info(f"Found {len(all_labels)} unique product types")

    for anchor_label in all_labels:
        anchor_indices = label_to_indices[anchor_label]
        if len(anchor_indices) < 2:
            continue
        for anchor_idx in anchor_indices:
            positive_idx = random.choice([i for i in anchor_indices if i != anchor_idx])
            negative_label = random.choice([l for l in all_labels if l != anchor_label])
            negative_idx = random.choice(label_to_indices[negative_label])

            anchor_text = df.loc[anchor_idx, text_col]
            positive_text = df.loc[positive_idx, text_col]
            negative_text = df.loc[negative_idx, text_col]

            # Skip if any text is empty
            if not all([anchor_text, positive_text, negative_text]):
                continue

            triplets.append(InputExample(texts=[anchor_text, positive_text, negative_text]))

            if len(triplets) >= max_triplets:
                break
        if len(triplets) >= max_triplets:
            break

    logger.info(f"Generated {len(triplets)} triplets")
    return triplets

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Generate triplets
    triplets = generate_triplets(
        df,
        label_col='product_type_flat',
        text_col='embedding_input',
        max_triplets=10000
    )
    
    # Print some example triplets
    logger.info("\nExample triplets:")
    for i, triplet in enumerate(triplets[:3]):
        logger.info(f"\nTriplet {i+1}:")
        logger.info(f"Anchor: {triplet.texts[0][:100]}...")
        logger.info(f"Positive: {triplet.texts[1][:100]}...")
        logger.info(f"Negative: {triplet.texts[2][:100]}...")

if __name__ == "__main__":
    main()
