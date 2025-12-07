
from recommender import PaperRecommender
from arxiv_utils import fetch_arxiv_papers
import pandas as pd


def main():
    print("Paper Recommender - Programmatic Example")
    print("=" * 60)
    
    # Option 1: Load from CSV if available
    # df = pd.read_csv('papers.csv')
    
    # Option 2: Fetch from arXiv
    print("\n1. Fetching papers...")
    df = fetch_arxiv_papers(query='cat:cs.LG', max_results=1000)
    df.to_csv('papers.csv', index=False)
    
    # Create recommender
    print("\n2. Creating recommender...")
    rec = PaperRecommender(df)
    
    # Build both methods
    print("\n3. Building TF-IDF...")
    rec.build_tfidf()
    
    print("\n4. Building SciBERT...")
    rec.build_scibert(batch_size=32)
    
    # Test all three methods
    query = "deep learning for computer vision"
    print("Qeury:", query)
    print("\n" + "=" * 60)
    print("5. Testing all three recommenders")
    print("=" * 60)
    
    print("\n--- TF-IDF ---")
    tfidf_results = rec.recommend_tfidf(query=query, top_k=5)
    
    print("\n--- SciBERT ---")
    scibert_results = rec.recommend_scibert(query=query, top_k=5)
    
    print("\n--- Hybrid ---")
    hybrid_results = rec.recommend_hybrid(query=query, top_k=5)
    
    # Give feedback
    print("\n6. Testing user feedback...")
    rec.give_feedback(hybrid_results[0]['idx'], 'like')
    rec.give_feedback(hybrid_results[2]['idx'], 'dislike')
    
    # Search again with feedback
    print("\n7. Searching with feedback applied...")
    new_results = rec.recommend_scibert(query="neural networks", top_k=5, apply_feedback=True)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("For interactive use, run: python cli.py")
    print("=" * 60)


if __name__ == "__main__":
    main()