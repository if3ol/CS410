
from recommender import PaperRecommender
from arxiv_utils import fetch_arxiv_papers
import pandas as pd
import os
import sys


def display_menu():
    print("\n" + "="*60)
    print("CS410 Paper Recommender Menu")
    print("="*60)
    print("1. Search papers (choose recommender)")
    print("2. Give feedback on a paper")
    print("3. View user profile")
    print("4. Reset user profile")
    print("5. Exit")
    print("="*60)


def choose_recommender():
    """Let user choose which recommender to use."""
    print("\nChoose recommender:")
    print("1. TF-IDF")
    print("2. SciBERT")
    print("3. Hybrid")
    
    while True:
        choice = input("\nYour choice (1-3): ").strip()
        if choice == '1':
            return 'tfidf'
        elif choice == '2':
            return 'scibert'
        elif choice == '3':
            return 'hybrid'
        else:
            print("Invalid choice. Enter 1, 2, or 3.")


def search_papers(rec):
    """Search for papers."""
    # Choose recommender
    method = choose_recommender()
    
    # Get query
    query = input("\nEnter your search query: ").strip()
    if not query:
        print("Empty query. Returning to menu.")
        return
    
    # Get number of results
    while True:
        try:
            top_k = input("How many results? (default: 10): ").strip()
            top_k = int(top_k) if top_k else 10
            if top_k < 1:
                print("Must be at least 1.")
                continue
            break
        except ValueError:
            print("Invalid number.")
    
    # Search
    print(f"\nSearching with {method.upper()}...")
    
    if method == 'tfidf':
        results = rec.recommend_tfidf(query=query, top_k=top_k)
    elif method == 'scibert':
        results = rec.recommend_scibert(query=query, top_k=top_k)
    else:  # hybrid
        results = rec.recommend_hybrid(query=query, top_k=top_k)
    
    return results


def give_feedback(rec, last_results=None):
    """Give feedback on a paper."""
    if last_results is None:
        print("\nNo recent search results. Search first, then give feedback.")
        return
    
    print("\nRecent results:")
    for r in last_results[:10]:  # Show max 10
        print(f"{r['rank']}. {r['title'][:80]}...")
    
    while True:
        rank_input = input("\nEnter rank number to give feedback (or 'done'): ").strip()
        
        if rank_input.lower() == 'done':
            break
        
        try:
            rank = int(rank_input)
            if rank < 1 or rank > len(last_results):
                print(f"Invalid rank. Must be 1-{len(last_results)}")
                continue
            
            feedback = input("Feedback (like/dislike): ").strip().lower()
            if feedback not in ['like', 'dislike', 'l', 'd']:
                print("Invalid feedback. Use 'like' or 'dislike'")
                continue
            
            paper_idx = last_results[rank-1]['idx']
            rec.give_feedback(paper_idx, feedback)
            print(f"✓ Feedback recorded for: {last_results[rank-1]['title'][:60]}...")
            
        except ValueError:
            print("Invalid input. Enter a number or 'done'")


def view_profile(rec):
    print("\n" + "="*60)
    print("User Profile")
    print("="*60)
    
    liked = rec.user_prefs['liked']
    disliked = rec.user_prefs['disliked']
    
    print(f"\nLiked papers: {len(liked)}")
    if liked:
        for i, paper_id in enumerate(liked[:10], 1):
            paper = rec.df[rec.df['id'] == paper_id]
            if not paper.empty:
                print(f"  {i}. {paper.iloc[0]['title']}")
        if len(liked) > 10:
            print(f"  ... and {len(liked) - 10} more")
    
    print(f"\nDisliked papers: {len(disliked)}")
    if disliked:
        for i, paper_id in enumerate(disliked[:10], 1):
            paper = rec.df[rec.df['id'] == paper_id]
            if not paper.empty:
                print(f"  {i}. {paper.iloc[0]['title']}")
        if len(disliked) > 10:
            print(f"  ... and {len(disliked) - 10} more")
    
    print("="*60)


def reset_profile(rec):
    """Reset user profile."""
    confirm = input("\nAre you sure you want to reset your profile? (yes/no): ").strip().lower()
    if confirm in ['yes', 'y']:
        rec.user_prefs = {'liked': [], 'disliked': []}
        rec._save_user_prefs()
        print("User profile reset")
    else:
        print("Reset cancelled")


def main():
    print("\n" + "="*60)
    print("CS410 Academic Paper Recommender System")
    print("="*60)
    
    # Load or fetch data
    if os.path.exists('papers.csv'):
        print("\nFound existing papers.csv")
        choice = input("Use existing data? (yes/no): ").strip().lower()
        if choice in ['yes', 'y']:
            print("Loading papers from papers.csv...")
            df = pd.read_csv('papers.csv')
            print(f"✓ Loaded {len(df)} papers")
        else:
            print("\nFetching 1000 papers from arXiv (cs.LG)...")
            df = fetch_arxiv_papers(query='cat:cs.LG', max_results=1000)
            df.to_csv('papers.csv', index=False)
            print("✓ Saved to papers.csv")
    else:
        print("\nFetching 1000 papers from arXiv (cs.LG)...")
        df = fetch_arxiv_papers(query='cat:cs.LG', max_results=1000)
        df.to_csv('papers.csv', index=False)
        print("✓ Saved to papers.csv")
    
    # Create recommender
    print("\nInitializing recommender...")
    rec = PaperRecommender(df)
    
    # Build TF-IDF
    print("\nBuilding TF-IDF...")
    rec.build_tfidf()
    
    # Build SciBERT
    print("Building SciBERT...")
    rec.build_scibert()
    
    print("\nReady!")
    
    # Main loop
    last_results = None
    
    while True:
        display_menu()
        choice = input("\nYour choice (1-5): ").strip()
        
        if choice == '1':
            last_results = search_papers(rec)
        elif choice == '2':
            give_feedback(rec, last_results)
        elif choice == '3':
            view_profile(rec)
        elif choice == '4':
            reset_profile(rec)
        elif choice == '5':
            print("\nExiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Enter 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)