import urllib.request
import xml.etree.ElementTree as ET
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fetch_arxiv_papers_api(query='cat:cs.AI', max_results=3000):
    """
    Fetch papers directly from arXiv API.
    
    Parameters:
    - query: arXiv query string
        Examples:
        'cat:cs.AI' = AI papers
        'cat:cs.LG' = Machine Learning
        'cat:cs.CV' = Computer Vision
        'cat:cs.CL' = Computation and Language (NLP)
        'cat:cs.AI OR cat:cs.LG' = AI or ML papers
    - max_results: How many papers to fetch
    
    Returns:
    - DataFrame with papers
    """
    base_url = 'http://export.arxiv.org/api/query?'
    
    papers = []
    start = 0
    per_request = 100  # API allows max 100 per request
    
    print(f"Fetching {max_results} papers from arXiv...")
    print(f"Query: {query}")
    print("-" * 50)
    
    while len(papers) < max_results:
        # Build query URL
        query_url = f"{base_url}search_query={query}&start={start}&max_results={per_request}&sortBy=submittedDate&sortOrder=descending"
        
        try:
            print(f"Requesting papers {start} to {start + per_request}...", end=' ')
            
            # Fetch data
            with urllib.request.urlopen(query_url) as response:
                xml_data = response.read()
            
            # Parse XML response
            root = ET.fromstring(xml_data)
            
            # Find all entry elements (each is a paper)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            
            if not entries:
                print("No more results!")
                break
            
            print(f"Got {len(entries)} papers")
            
            # Extract paper information
            for entry in entries:
                try:
                    # Get paper ID
                    paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text
                    paper_id = paper_id.split('/')[-1]  # Extract just the ID
                    
                    # Get title
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text
                    title = ' '.join(title.split())  # Clean whitespace
                    
                    # Get abstract
                    abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
                    abstract = ' '.join(abstract.split())  # Clean whitespace
                    
                    # Get authors
                    author_elements = entry.findall('{http://www.w3.org/2005/Atom}author')
                    authors = ', '.join([
                        author.find('{http://www.w3.org/2005/Atom}name').text 
                        for author in author_elements
                    ])
                    
                    # Get publication year
                    published = entry.find('{http://www.w3.org/2005/Atom}published').text
                    year = published[:4]
                    
                    # Get categories
                    primary_cat = entry.find('{http://arxiv.org/schemas/atom}primary_category')
                    if primary_cat is not None:
                        categories = primary_cat.attrib['term']
                    else:
                        categories = 'unknown'
                    
                    # Add to list
                    papers.append({
                        'id': paper_id,
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'categories': categories,
                        'year': year
                    })
                    
                except Exception as e:
                    print(f"\n  Warning: Couldn't parse a paper: {e}")
                    continue
            
            start += per_request
            
            # IMPORTANT: Sleep to respect arXiv API rate limits
            # ArXiv requires 3 second delay between requests
            if len(papers) < max_results:
                print("  Waiting 3 seconds (API rate limit)...")
                time.sleep(3)
            
        except Exception as e:
            print(f"\n Error fetching data: {e}")
            print("Continuing with papers collected so far...")
            break
    
    print("-" * 50)
    print(f" Successfully fetched {len(papers)} papers!")
    
    # Convert to DataFrame
    df = pd.DataFrame(papers)
    #print(df)
    
    # Clean the data
    print("\nCleaning data...")
    original_len = len(df)
    
    # Remove papers without abstracts
    df = df.dropna(subset=['abstract'])
    
    # Remove papers with very short abstracts (likely errors)
    df = df[df['abstract'].str.len() > 100]
    
    # Remove duplicates (just in case)
    df = df.drop_duplicates(subset=['id'])
    
    print(f"  Removed {original_len - len(df)} low-quality papers")
    print(f"  Final dataset: {len(df)} papers")
    
    return df

# 1 plot - shows abstract length distribution
def plot_abstract_lengths(df, save_path='figures/abstract_lengths.png'):
    """
    Shows that abstracts have enough content for text analysis.
    This demonstrates that a TF-IDF approach will have meaningful data.
    """
    #  Compute word counts for each abstract
    word_counts = df['abstract'].apply(lambda x: len(x.split()))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(
        word_counts, 
        bins=50,
        edgecolor='black',
        alpha=0.7,
        color='steelblue'
    )

    # Add median and mean lines
    median_val = word_counts.median()
    mean_val = word_counts.mean()

    ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.0f} words')
    ax.axvline(mean_val, color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.0f} words')

    # Labels and title
    ax.set_xlabel('Number of Words in Abstract', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Abstract Lengths', fontsize=14, fontweight='bold')

    # Legend + grid
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()

    # Interpretation
    print(f"\nInterpretation:")
    print(f"Median abstract length: {median_val:.0f} words")
    print(f"Mean abstract length: {mean_val:.0f} words")
    print("These lengths suggest abstracts are long enough for TF-IDF analysis.")


# plot 2
def plot_papers_over_time(df, save_path='figures/papers_over_time.png'):
    """
    Shows temporal distribution of papers.
    Proves your dataset covers recent research.
    """
    # Count papers per year
    year_counts = df['year'].value_counts().sort_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar chart
    bars = ax.bar(
        year_counts.index, 
        year_counts.values,
        color='steelblue',
        edgecolor='black',
        alpha=0.7
    )
    
    # Highlight most recent years
    current_year = '2024'
    for i, (year, bar) in enumerate(zip(year_counts.index, bars)):
        if int(year) >= 2023:
            bar.set_color('orange')
    
    # Labels
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Papers Over Time', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels if many years
    plt.xticks(rotation=45, ha='right')
    
    # Grid
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Saved: {save_path}")
    plt.show()
    
    # Interpretation
    recent_papers = df[df['year'].astype(int) >= 2022].shape[0]
    pct_recent = recent_papers / len(df) * 100
    print(f"\n Interpretation:")
    print(f"{pct_recent:.1f}% of papers are from 2022 or later,")
    print(f"demonstrating the dataset covers recent research.")


 # tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_tfidf_vectorizer(df):
    """
    Convert all paper abstracts into TF-IDF vectors.
    
    Think about:
    - Should we use just abstracts, or title + abstract?
    - What parameters matter?
    """
    
    # Combine title and abstract (titles are important)
    # Titles  contain the key terms
    documents = df['title'] + ' ' + df['abstract']
    
    print("Creating TF-IDF vectorizer...")
    print(f"Processing {len(documents)} documents...")
    
    # Create vectorizer with reasonable parameters
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Keep top 5000 most important words
        stop_words='english',   # Remove common words like "the", "is", "at"
        ngram_range=(1, 2),     # Include single words AND pairs (e.g., "neural network")
        min_df=2,               # Ignore words appearing in only 1 document
        max_df=0.8              # Ignore words appearing in >80% of documents
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f" Created TF-IDF matrix: {tfidf_matrix.shape}")
    print(f"  - {tfidf_matrix.shape[0]} papers")
    print(f"  - {tfidf_matrix.shape[1]} unique terms")
    print(f"  - {tfidf_matrix.nnz:,} non-zero values")
    print(f"  - Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")
    
    return tfidf_matrix, vectorizer


def explore_tfidf_matrix(tfidf_matrix, vectorizer, df, paper_idx=0):
    """
    Look inside the TF-IDF matrix to understand what it contains.
    helps to understand what the model "sees".
    """
    print("\n" + "="*60)
    print("EXPLORING TF-IDF MATRIX")
    print("="*60)
    
    # Get feature names (the vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\nVocabulary size: {len(feature_names)}")
    print(f"Sample terms: {list(feature_names[:20])}")
    
    # Look at one paper's representation
    print(f"\n" + "-"*60)
    print(f"Example: Paper {paper_idx}")
    print(f"-"*60)
    print(f"Title: {df.iloc[paper_idx]['title']}")
    print(f"\nTop 15 terms by TF-IDF score:")
    
    # Get this paper's TF-IDF vector
    paper_vector = tfidf_matrix[paper_idx].toarray().flatten()
    
    # Get top scoring terms
    top_indices = paper_vector.argsort()[-15:][::-1]
    
    for rank, idx in enumerate(top_indices, 1):
        term = feature_names[idx]
        score = paper_vector[idx]
        if score > 0:  # Only show non-zero
            print(f"  {rank:2d}. {term:20s} → {score:.4f}")
    
    print("\n These are the most terms for this paper")
    print("  They appear frequently here but rarely across all papers.")


def recommend_papers(seed_idx, tfidf_matrix, df, top_k=10):
    """
    Given a seed paper, find the most similar papers.
    
    How it works:
    1. Get the seed paper's TF-IDF vector
    2. Compare it to ALL other papers using cosine similarity
    3. Sort by similarity and return top K
    
    Parameters:
    - seed_idx: Index of the seed paper in dataframe
    - tfidf_matrix: The TF-IDF matrix we created
    - df: The dataframe with paper metadata
    - top_k: How many recommendations to return
    """
    
    # Get seed paper info
    seed_paper = df.iloc[seed_idx]
    
    print("\n" + "="*80)
    print("GENERATING RECOMMENDATIONS")
    print("="*80)
    print(f"\n SEED PAPER:")
    print(f"   Title: {seed_paper['title']}")
    print(f"   Year: {seed_paper['year']}")
    print(f"   Category: {seed_paper['categories']}")
    print(f"   Abstract: {seed_paper['abstract'][:200]}...")
    
    # Get seed paper's TF-IDF vector
    seed_vector = tfidf_matrix[seed_idx]
    
    # Compute cosine similarity with ALL papers
    # Cosine similarity measures angle between vectors (0 to 1)
    # 1 = identical, 0 = completely different
    similarities = cosine_similarity(seed_vector, tfidf_matrix).flatten()
    
    # Get indices of most similar papers (excluding seed itself)
    # argsort gives indices that would sort the array
    # [::-1] reverses to get descending order
    # [1:top_k+1] skips the seed (always most similar to itself)
    similar_indices = similarities.argsort()[::-1][1:top_k+1]
    
    # Build results
    results = []
    print(f"\n TOP {top_k} RECOMMENDATIONS:")
    print("-"*80)
    
    for rank, idx in enumerate(similar_indices, 1):
        paper = df.iloc[idx]
        sim_score = similarities[idx]
        
        results.append({
            'rank': rank,
            'id': paper['id'],
            'title': paper['title'],
            'year': paper['year'],
            'categories': paper['categories'],
            'similarity': sim_score,
            'abstract': paper['abstract']
        })
        
        print(f"\n{rank}. [Similarity: {sim_score:.4f}]")
        print(f"   {paper['title']}")
        print(f"   Year: {paper['year']} | Category: {paper['categories']}")
        print(f"   Abstract: {paper['abstract'][:150]}...")
    
    print("\n" + "="*80)
    
    return results




# Fetch 3000 papers from multiple CS areas

print("\nFETCHING FULL DATASET")

df = fetch_arxiv_papers_api(
    query='cat:cs.LG', #'OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL',
    max_results=3000
)

# Just deep learning papers
# df = fetch_arxiv_papers_api(query='all:deep learning', max_results=2000)


# Create figures folder if it doesn't exist
import os
os.makedirs('figures', exist_ok=True)


# Generate the plot
plot_papers_over_time(df)

# Generate the plot
plot_abstract_lengths(df)


# ########
# Test tf-idf
tfidf_matrix, vectorizer = create_tfidf_vectorizer(df)
explore_tfidf_matrix(tfidf_matrix, vectorizer, df, paper_idx=0)
# Test it on the first paper
results = recommend_papers(seed_idx=0, tfidf_matrix=tfidf_matrix, df=df, top_k=10)



# # Generate the plot
# plot_papers_over_time(df)



# # Or NLP papers
# df = fetch_arxiv_papers_api(query='cat:cs.CL', max_results=2000)

# # Or recent papers (last 2 years)
# df = fetch_arxiv_papers_api(query='cat:cs.AI AND submittedDate:[2023 TO 2024]', max_results=2000)

# Isave for faster lookup, not for now
# print("\nSaving to CSV...")
# df.to_csv('arxiv_sample_3k.csv', index=False)
# print(" Saved to 'arxiv_sample_3k.csv'")

# Also save as JSON (more flexible)
# df.to_json('arxiv_sample_3k.json', orient='records', lines=True)
# print("Saved to 'arxiv_sample_3k.json'")