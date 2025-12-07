
import urllib.request
import xml.etree.ElementTree as ET
import time
import pandas as pd
import matplotlib.pyplot as plt

def fetch_arxiv_papers(query='cat:cs.LG', max_results=3000):
    """
    Fetch papers from arXiv API.
    
    Parameters:
    - query: arXiv query string (e.g., 'cat:cs.AI', 'cat:cs.LG OR cat:cs.CV')
    - max_results: Number of papers to fetch
    
    Returns:
    - DataFrame with columns: id, title, abstract, authors, categories, year
    """
    base_url = 'http://export.arxiv.org/api/query?'
    papers = []
    start = 0
    per_request = 100
    
    print(f"Fetching {max_results} papers from arXiv...")
    print(f"Query: {query}")
    print("-" * 50)
    
    while len(papers) < max_results:
        query_url = f"{base_url}search_query={query}&start={start}&max_results={per_request}&sortBy=submittedDate&sortOrder=descending"
        
        try:
            print(f"Requesting papers {start} to {start + per_request}...", end=' ')
            
            with urllib.request.urlopen(query_url) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            
            if not entries:
                print("No more results!")
                break
            
            print(f"Got {len(entries)} papers")
            
            for entry in entries:
                try:
                    paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                    title = ' '.join(entry.find('{http://www.w3.org/2005/Atom}title').text.split())
                    abstract = ' '.join(entry.find('{http://www.w3.org/2005/Atom}summary').text.split())
                    
                    author_elements = entry.findall('{http://www.w3.org/2005/Atom}author')
                    authors = ', '.join([a.find('{http://www.w3.org/2005/Atom}name').text for a in author_elements])
                    
                    published = entry.find('{http://www.w3.org/2005/Atom}published').text
                    year = published[:4]
                    
                    primary_cat = entry.find('{http://arxiv.org/schemas/atom}primary_category')
                    categories = primary_cat.attrib['term'] if primary_cat is not None else 'unknown'
                    
                    papers.append({
                        'id': paper_id,
                        'title': title,
                        'abstract': abstract,
                        'authors': authors,
                        'categories': categories,
                        'year': year
                    })
                except Exception as e:
                    print(f"\nWarning: Couldn't parse a paper: {e}")
                    continue
            
            start += per_request
            
            if len(papers) < max_results:
                print("  Waiting 3 seconds (API rate limit)...")
                time.sleep(3)
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
    
    print("-" * 50)
    print(f"✓ Fetched {len(papers)} papers")
    
    df = pd.DataFrame(papers)
    
    # Clean data
    original_len = len(df)
    df = df.dropna(subset=['abstract'])
    df = df[df['abstract'].str.len() > 100]
    df = df.drop_duplicates(subset=['id'])
    df = df.reset_index(drop=True)
    
    if original_len - len(df) > 0:
        print(f"Removed {original_len - len(df)} low-quality papers")
    print(f"Final dataset: {len(df)} papers\n")
    
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

