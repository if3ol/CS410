
import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel


class PaperRecommender:
    def __init__(self, df):
        """Initialize recommender with paper dataframe."""
        self.df = df.reset_index(drop=True)
        
        # TF-IDF components
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        
        # SciBERT components
        self.scibert_embeddings = None
        self.scibert_model = None
        self.scibert_tokenizer = None
        self.device = None
        
        # User feedback
        self.user_prefs = self._load_user_prefs()
        
        print(f"Initialized recommender with {len(df)} papers")
    
    
    # tfidf
    
    def build_tfidf(self, max_features=5000):
        """Build TF-IDF matrix from paper titles and abstracts."""
        documents = self.df['title'] + ' ' + self.df['abstract']
        
        print(f"Building TF-IDF matrix for {len(documents)} documents...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        print(f"✓ TF-IDF matrix: {self.tfidf_matrix.shape}")
        print(f"  Vocabulary: {len(self.tfidf_vectorizer.get_feature_names_out())} terms")
        print(f"  Sparsity: {(1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) * 100:.1f}%\n")
    
    
    def recommend_tfidf(self, query=None, seed_idx=None, top_k=10):
        """
        Recommend using TF-IDF similarity.
        Either provide query (text) or seed_idx (paper index).
        """
        if self.tfidf_matrix is None:
            raise ValueError("Build TF-IDF first with build_tfidf()")
        
        if query is not None:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[::-1][:top_k]
        elif seed_idx is not None:
            seed_vector = self.tfidf_matrix[seed_idx]
            similarities = cosine_similarity(seed_vector, self.tfidf_matrix).flatten()
            similarities[seed_idx] = -1
            top_indices = similarities.argsort()[::-1][:top_k]
        else:
            raise ValueError("Provide either query or seed_idx")
        
        return self._format_results(top_indices, similarities, 'TF-IDF')
    
    
    # SciBERT
    
    def build_scibert(self, batch_size=32, cache_path='scibert_embeddings.pkl'):
        """Build SciBERT embeddings for all papers."""
        # Try cache first
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}...")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                if list(cache['paper_ids']) == list(self.df['id'].values):
                    self.scibert_embeddings = cache['embeddings']
                    print(f"✓ Loaded {len(self.scibert_embeddings)} embeddings\n")
                    return
        
        # Load model
        print("Loading SciBERT model...")
        self.scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.scibert_model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scibert_model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Generate embeddings
        print(f"Generating embeddings for {len(self.df)} papers...")
        documents = (self.df['title'] + ' ' + self.df['abstract']).tolist()
        
        embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"Batch {batch_num}/{total_batches}...", end=' ')
            
            inputs = self.scibert_tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.scibert_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
            
            print("Done")
        
        self.scibert_embeddings = np.vstack(embeddings)
        print(f"\n✓ Embeddings: {self.scibert_embeddings.shape}")
        
        # Cache
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.scibert_embeddings,
                'paper_ids': self.df['id'].values,
                'timestamp': datetime.now().isoformat()
            }, f)
        print(f"✓ Cached to {cache_path}\n")
    
    
    def recommend_scibert(self, query=None, seed_idx=None, top_k=10, apply_feedback=True):
        """
        Recommend using SciBERT embeddings.
        Either provide query (text) or seed_idx (paper index).
        """
        if self.scibert_embeddings is None:
            raise ValueError("Build SciBERT first with build_scibert()")
        
        if query is not None:
            # Load model if not already loaded
            if self.scibert_tokenizer is None:
                self.scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
                self.scibert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
                self.scibert_model.eval()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.scibert_model.to(self.device)
            
            # Encode query
            inputs = self.scibert_tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.scibert_model(**inputs)
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            similarities = cosine_similarity(query_embedding, self.scibert_embeddings).flatten()
        elif seed_idx is not None:
            seed_embedding = self.scibert_embeddings[seed_idx:seed_idx+1]
            similarities = cosine_similarity(seed_embedding, self.scibert_embeddings).flatten()
            similarities[seed_idx] = -1
        else:
            raise ValueError("Provide either query or seed_idx")
        
        if apply_feedback:
            similarities = self._apply_feedback(similarities)
        
        top_indices = similarities.argsort()[::-1][:top_k]
        return self._format_results(top_indices, similarities, 'SciBERT')
    
    # Hybrid 
    
    def recommend_hybrid(self, query=None, seed_idx=None, top_k=10, 
                        tfidf_weight=0.4, scibert_weight=0.6):
        """Combine TF-IDF and SciBERT scores."""
        if self.tfidf_matrix is None or self.scibert_embeddings is None:
            raise ValueError("Build both TF-IDF and SciBERT first")
        
        # Get TF-IDF scores
        if query is not None:
            tfidf_vec = self.tfidf_vectorizer.transform([query])
            tfidf_sims = cosine_similarity(tfidf_vec, self.tfidf_matrix).flatten()
        else:
            tfidf_vec = self.tfidf_matrix[seed_idx]
            tfidf_sims = cosine_similarity(tfidf_vec, self.tfidf_matrix).flatten()
            tfidf_sims[seed_idx] = -1
        
        # Get SciBERT scores
        if query is not None:
            inputs = self.scibert_tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.scibert_model(**inputs)
                query_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            scibert_sims = cosine_similarity(query_emb, self.scibert_embeddings).flatten()
        else:
            seed_emb = self.scibert_embeddings[seed_idx:seed_idx+1]
            scibert_sims = cosine_similarity(seed_emb, self.scibert_embeddings).flatten()
            scibert_sims[seed_idx] = -1
        
        # Combine
        combined = tfidf_weight * tfidf_sims + scibert_weight * scibert_sims
        combined = self._apply_feedback(combined)
        
        top_indices = combined.argsort()[::-1][:top_k]
        return self._format_results(top_indices, combined, f'Hybrid ({tfidf_weight:.1f}/{scibert_weight:.1f})')
    
    def give_feedback(self, paper_idx, feedback):
        """Give feedback on a paper: 'like' or 'dislike'."""
        paper_id = self.df.iloc[paper_idx]['id']
        
        if feedback.lower() in ['like', 'yes', 'y']:
            if paper_id not in self.user_prefs['liked']:
                self.user_prefs['liked'].append(paper_id)
        elif feedback.lower() in ['dislike', 'no', 'n']:
            if paper_id not in self.user_prefs['disliked']:
                self.user_prefs['disliked'].append(paper_id)
        
        self._save_user_prefs()
        print(f"✓ Feedback recorded")
    
    
    def _apply_feedback(self, similarities):
        """Apply user preferences to scores."""
        adjusted = similarities.copy()
        
        for paper_id in self.user_prefs['liked']:
            if paper_id in self.df['id'].values:
                idx = self.df[self.df['id'] == paper_id].index[0]
                adjusted[idx] = min(1.0, adjusted[idx] + 0.1)
        
        for paper_id in self.user_prefs['disliked']:
            if paper_id in self.df['id'].values:
                idx = self.df[self.df['id'] == paper_id].index[0]
                adjusted[idx] = max(0.0, adjusted[idx] - 0.2)
        
        return adjusted
    
    
    def _load_user_prefs(self):
        if os.path.exists('user_prefs.json'):
            with open('user_prefs.json', 'r') as f:
                return json.load(f)
        return {'liked': [], 'disliked': []}
    
    
    def _save_user_prefs(self):
        with open('user_prefs.json', 'w') as f:
            json.dump(self.user_prefs, f)
    
    
    def _format_results(self, indices, similarities, method):
        results = []
        
        print(f"\n{'='*80}")
        print(f"TOP {len(indices)} RECOMMENDATIONS ({method})")
        print(f"{'='*80}")
        
        for rank, idx in enumerate(indices, 1):
            paper = self.df.iloc[idx]
            sim = similarities[idx]
            
            results.append({
                'rank': rank,
                'idx': idx,
                'id': paper['id'],
                'title': paper['title'],
                'similarity': float(sim),
                'year': paper['year'],
                'categories': paper['categories']
            })
            
            print(f"\n{rank}. [Score: {sim:.4f}]")
            print(f"   {paper['title']}")
            print(f"   {paper['year']} | {paper['authors']}")
        
        print(f"\n{'='*80}\n")
        return results