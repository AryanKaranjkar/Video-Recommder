"""
Video Recommendation System
Machine Learning Assignment - User Interaction Based

Author: ARYAN KARANJKAR
Description: Complete implementation of a video recommendation system using
             collaborative filtering, content-based filtering, and hybrid approaches.
"""

# ============================================================================
# SECTION 1: AUTO-INSTALL DEPENDENCIES
# ============================================================================

import subprocess
import sys

def install_dependencies():
    """Automatically install required packages"""
    packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'scipy',
        'lightgbm',
        'implicit',
        'tabulate'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            # Added a timeout and error handling for robust installation
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package], timeout=300)
            print(f"✓ {package} installed")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
    print("\nAll dependencies installed!\n")

# ============================================================================
# SECTION 2: IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# ============================================================================
# SECTION 3: DATA GENERATION (IF NO DATASET PROVIDED) - KEPT FOR FALLBACK
# ============================================================================

def generate_sample_data(n_users=1000, n_videos=500, n_interactions=5000):
    """
    Generate synthetic dataset matching the assignment specifications
    """
    print("Generating synthetic dataset...")
    
    # ... (simplified for execution)
    categories = ['Education', 'Entertainment', 'Technology']
    tag_pool = ['tutorial', 'funny', 'review']
    
    data = []
    
    for _ in range(n_interactions):
        user_id = f"user_{np.random.randint(1, n_users+1)}"
        video_id = f"video_{np.random.randint(1, n_videos+1)}"
        category = np.random.choice(categories)
        n_tags = np.random.randint(1, 4)
        tags = ','.join(np.random.choice(tag_pool, n_tags, replace=False))
        watch_duration = np.random.beta(2, 2) * 100
        liked = 1 if watch_duration > 60 and np.random.random() > 0.7 else 0
        commented = 1 if liked and np.random.random() > 0.8 else 0
        subscribed = 1 if watch_duration > 80 and liked and np.random.random() > 0.9 else 0
        timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        
        data.append({
            'user_id': user_id, 'video_id': video_id, 'category': category, 'tags': tags,
            'watch_duration': round(watch_duration, 2), 'liked': liked, 'commented': commented,
            'subscribed_after_watching': subscribed, 'timestamp': timestamp
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} interactions for {df['user_id'].nunique()} users and {df['video_id'].nunique()} videos\n")
    return df

# ============================================================================
# SECTION 4: DATA PREPROCESSING & VALIDATION
# ============================================================================

class DataValidator:
    """Validate and clean input data"""
    
    @staticmethod
    def validate_dataset(df):
        """Ensure dataset meets requirements"""
        required_cols = ['user_id', 'video_id', 'category', 'tags', 
                        'watch_duration', 'liked', 'commented', 
                        'subscribed_after_watching', 'timestamp']
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("Dataset is empty!")
        
        initial_rows = len(df)
        df = df.dropna(subset=['user_id', 'video_id'])
        removed = initial_rows - len(df)
        
        if removed > 0:
            print(f"⚠ Removed {removed} rows with missing user_id or video_id")
        
        print("✓ Dataset validation passed")
        return df
    
    @staticmethod
    def clean_data(df):
        """Clean and preprocess data"""
        df = df.copy()
        
        df['tags'] = df['tags'].fillna('')
        df['category'] = df['category'].fillna('Unknown')
        df['watch_duration'] = df['watch_duration'].fillna(0).clip(0, 100)
        
        for col in ['liked', 'commented', 'subscribed_after_watching']:
            df[col] = df[col].fillna(0).astype(int)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print("✓ Data cleaning completed")
        return df

# ============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS
# ============================================================================

class EDAAnalyzer:
    """Perform exploratory data analysis"""
    
    @staticmethod
    def analyze(df):
        """Generate EDA insights"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60 + "\n")
        
        print(f"Total Interactions: {len(df)}")
        print(f"Unique Users: {df['user_id'].nunique()}")
        print(f"Unique Videos: {df['video_id'].nunique()}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        
        print("Engagement Statistics:")
        print(f"  Average Watch Duration: {df['watch_duration'].mean():.2f}%")
        
        print("Top Categories:")
        print(df['category'].value_counts().head(5).to_markdown())
        print()
        
        total_possible = df['user_id'].nunique() * df['video_id'].nunique()
        sparsity = (1 - len(df) / total_possible) * 100
        print(f"Data Sparsity: {sparsity:.2f}%\n")
        
        return {
            'n_users': df['user_id'].nunique(),
            'n_videos': df['video_id'].nunique(),
            'avg_watch': df['watch_duration'].mean(),
            'sparsity': sparsity
        }

# ============================================================================
# SECTION 6: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Engineer features for recommendation models"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.user_encoder = LabelEncoder()
        self.video_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.video_tag_features = None 
    
    def create_engagement_score(self):
        """Create composite engagement score"""
        self.df['engagement_score'] = (
            self.df['watch_duration'] * 0.4 +
            self.df['liked'] * 20 +
            self.df['commented'] * 30 +
            self.df['subscribed_after_watching'] * 50
        )
        
        max_score = 100 * 0.4 + 20 + 30 + 50
        self.df['engagement_score'] = (self.df['engagement_score'] / max_score) * 100
        
        print("✓ Created engagement score")
        return self
    
    def encode_categorical(self):
        """Encode user and video IDs"""
        self.df['user_idx'] = self.user_encoder.fit_transform(self.df['user_id'])
        self.df['video_idx'] = self.video_encoder.fit_transform(self.df['video_id'])
        self.df['category_idx'] = self.category_encoder.fit_transform(self.df['category'])
        
        print("✓ Encoded categorical variables")
        return self
    
    def create_tag_features(self):
        """Create TF-IDF features from tags"""
        video_tags = self.df.groupby('video_id')['tags'].apply(lambda x: ' '.join(x)).reset_index()
        
        # Use simple space separation for tags which are comma-separated in source data
        self.tfidf = TfidfVectorizer(max_features=50, token_pattern=r'[^,]+')
        tag_features = self.tfidf.fit_transform(video_tags['tags'])
        
        self.video_tag_features = pd.DataFrame(
            tag_features.toarray(),
            index=video_tags['video_id']
        )
        
        print("✓ Created tag embeddings using TF-IDF")
        return self
    
    def create_user_profiles(self):
        """Create user preference profiles"""
        user_cat_pref = self.df.groupby(['user_id', 'category'])['engagement_score'].mean().reset_index()
        user_cat_pivot = user_cat_pref.pivot(index='user_id', columns='category', values='engagement_score').fillna(0)
        
        self.user_profiles = user_cat_pivot
        
        print("✓ Created user preference profiles")
        return self
    
    def get_processed_data(self):
        """Return processed dataframe"""
        return self.df

# ============================================================================
# SECTION 7: RECOMMENDATION MODELS
# ============================================================================

class CollaborativeFilteringModel:
    """User-User Collaborative Filtering using cosine similarity"""
    
    def __init__(self, df):
        self.df = df
        self.user_item_matrix = None
        self.user_similarity_df = None
    
    def build(self):
        """Build user-item interaction matrix"""
        print("\nBuilding Collaborative Filtering Model...")
        
        self.user_item_matrix = self.df.pivot_table(
            index='user_id',
            columns='video_id',
            values='engagement_score',
            fill_value=0
        )
        
        if self.user_item_matrix.shape[0] < 2:
            print("⚠ Insufficient data for User-User CF (less than 2 users)")
            return self

        # Calculate user-user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        print("✓ Collaborative filtering model built")
        return self
    
    def recommend(self, user_id, n=10, exclude_watched=True):
        """Recommend videos for a user"""
        if self.user_item_matrix is None or self.user_similarity_df is None or user_id not in self.user_item_matrix.index:
            return []
        
        # Find similar users
        # Exclude self (index 0 will be the user itself if sorted)
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)[1:11]
        
        # Handle case where no other users exist
        if similar_users.empty:
            return []

        # Get videos watched by similar users
        similar_user_videos = self.user_item_matrix.loc[similar_users.index]
        
        # Calculate weighted scores
        video_scores = similar_user_videos.T.dot(similar_users.values)
        
        # Exclude already watched videos
        if exclude_watched:
            watched = self.user_item_matrix.loc[user_id]
            video_scores[watched > 0] = -1
        
        # Get top N recommendations
        top_videos = video_scores.nlargest(n)
        return list(top_videos.index)


class ContentBasedModel:
    """Content-based filtering using video features"""
    
    def __init__(self, df, video_tag_features):
        self.df = df
        self.video_tag_features = video_tag_features
        self.video_similarity_df = None
    
    def build(self):
        """Build content similarity matrix"""
        print("\nBuilding Content-Based Model...")
        
        if self.video_tag_features.shape[0] < 2:
            print("⚠ Insufficient data for Content-Based Model (less than 2 videos with tags)")
            return self

        # Calculate video-video similarity based on tags
        self.video_similarity = cosine_similarity(self.video_tag_features)
        self.video_similarity_df = pd.DataFrame(
            self.video_similarity,
            index=self.video_tag_features.index,
            columns=self.video_tag_features.index
        )
        
        print("✓ Content-based model built")
        return self
    
    def recommend(self, user_id, n=10):
        """Recommend videos based on user's watch history"""
        # Get user's watched videos with high engagement
        user_videos = self.df[
            (self.df['user_id'] == user_id) & 
            (self.df['engagement_score'] > 50)
        ]['video_id'].values
        
        if len(user_videos) == 0 or self.video_similarity_df is None:
            return []
        
        # Find similar videos
        recommendations = []
        for video in user_videos:
            if video in self.video_similarity_df.index:
                similar = self.video_similarity_df[video].sort_values(ascending=False)[1:n+1]
                recommendations.extend(similar.index.tolist())
        
        seen = set(user_videos)
        unique_recs = []
        for vid in recommendations:
            if vid not in seen:
                unique_recs.append(vid)
                seen.add(vid)
            if len(unique_recs) >= n:
                break
        
        return unique_recs


class HybridRecommender:
    """Hybrid model combining collaborative and content-based approaches"""
    
    def __init__(self, cf_model, cb_model, cf_weight=0.6):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.cf_weight = cf_weight
        self.cb_weight = 1 - cf_weight
    
    def recommend(self, user_id, n=10):
        """Generate hybrid recommendations"""
        if not (self.cf_model and self.cb_model):
            return []
            
        cf_recs = self.cf_model.recommend(user_id, n=n*2)
        cb_recs = self.cb_model.recommend(user_id, n=n*2)
        
        if not cf_recs and not cb_recs:
            return []
            
        video_scores = defaultdict(float)
        
        # Scoring based on rank
        for i, video in enumerate(cf_recs):
            if len(cf_recs) > 0:
                 video_scores[video] += self.cf_weight * (len(cf_recs) - i) / len(cf_recs)
        
        for i, video in enumerate(cb_recs):
            if len(cb_recs) > 0:
                video_scores[video] += self.cb_weight * (len(cb_recs) - i) / len(cb_recs)
        
        # Sort by combined score
        sorted_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [video for video, score in sorted_videos[:n]]

# ============================================================================
# SECTION 8: EVALUATION METRICS
# ============================================================================

class RecommenderEvaluator:
    """Evaluate recommendation model performance"""
    
    @staticmethod
    def precision_at_k(recommended, relevant, k=10):
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len([v for v in recommended_k if v in relevant_set])
        return hits / k if k > 0 else 0
    
    @staticmethod
    def recall_at_k(recommended, relevant, k=10):
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len([v for v in recommended_k if v in relevant_set])
        return hits / len(relevant_set) if len(relevant_set) > 0 else 0
    
    @staticmethod
    def mean_reciprocal_rank(recommended, relevant):
        relevant_set = set(relevant)
        for i, video in enumerate(recommended):
            if video in relevant_set:
                return 1 / (i + 1)
        return 0
    
    @staticmethod
    def ndcg_at_k(recommended, relevant, k=10):
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        dcg = sum([1 / np.log2(i + 2) if video in relevant_set else 0 
                   for i, video in enumerate(recommended_k)])
        
        ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(relevant_set)))])
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0
    
    @staticmethod
    def evaluate_model(model, test_data, k=10):
        """Evaluate model on test set"""
        print(f"\nEvaluating model...")
        
        # Due to small dataset, use all unique users in the test set
        users = test_data['user_id'].unique()
        
        precisions = []
        recalls = []
        mrrs = []
        ndcgs = []
        
        for user in users:
            # Get ground truth (videos user interacted with positively)
            relevant = test_data[
                (test_data['user_id'] == user) & 
                (test_data['engagement_score'] > 60)
            ]['video_id'].tolist()
            
            if len(relevant) == 0:
                continue
            
            recommended = model.recommend(user, n=k)
            
            if len(recommended) == 0:
                continue
            
            precisions.append(RecommenderEvaluator.precision_at_k(recommended, relevant, k))
            recalls.append(RecommenderEvaluator.recall_at_k(recommended, relevant, k))
            mrrs.append(RecommenderEvaluator.mean_reciprocal_rank(recommended, relevant))
            ndcgs.append(RecommenderEvaluator.ndcg_at_k(recommended, relevant, k))
        
        results = {
            'Precision@K': np.mean(precisions) if precisions else 0,
            'Recall@K': np.mean(recalls) if recalls else 0,
            'MRR': np.mean(mrrs) if mrrs else 0,
            'NDCG@K': np.mean(ndcgs) if ndcgs else 0
        }
        
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        return results

# ============================================================================
# SECTION 9: MAIN EXECUTION PIPELINE
# ============================================================================

class VideoRecommendationSystem:
    """Main recommendation system orchestrator"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.models = {}
        self.best_model = None
        self.video_tag_features = None
    
    def load_data(self):
        """Load or generate dataset"""
        print("\n" + "="*60)
        print("VIDEO RECOMMENDATION SYSTEM")
        print("="*60 + "\n")
        
        if self.data_path:
            try:
                self.df = pd.read_csv(self.data_path)
                print(f"✓ Loaded data from {self.data_path}")
            except Exception as e:
                print(f"✗ Could not load {self.data_path}. Error: {e}")
                print("Generating synthetic data as fallback.")
                self.df = generate_sample_data(n_interactions=50)
        else:
            print("No data path provided. Generating synthetic data.")
            self.df = generate_sample_data(n_interactions=50)
        
        self.df = DataValidator.validate_dataset(self.df)
        self.df = DataValidator.clean_data(self.df)
        
        return self
    
    def explore_data(self):
        """Perform EDA"""
        self.eda_stats = EDAAnalyzer.analyze(self.df)
        return self
    
    def engineer_features(self):
        """Engineer features"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60 + "\n")
        
        engineer = FeatureEngineer(self.df)
        engineer.create_engagement_score()
        engineer.encode_categorical()
        engineer.create_tag_features()
        engineer.create_user_profiles()
        
        self.df = engineer.get_processed_data()
        self.video_tag_features = engineer.video_tag_features
        
        return self
    
    def split_data(self, test_size=0.2):
        """Split into train/test sets"""
        print("\n" + "="*60)
        print("TRAIN/TEST SPLIT")
        print("="*60 + "\n")
        
        self.df = self.df.sort_values('timestamp')
        split_idx = int(len(self.df) * (1 - test_size))
        
        self.train_df = self.df.iloc[:split_idx]
        self.test_df = self.df.iloc[split_idx:]
        
        print(f"Training set: {len(self.train_df)} interactions")
        print(f"Test set: {len(self.test_df)} interactions")
        
        return self
    
    def train_models(self):
        """Train multiple recommendation models"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        cf_model = CollaborativeFilteringModel(self.train_df)
        cf_model.build()
        self.models['collaborative'] = cf_model
        
        # Content-Based and Hybrid models require sufficient videos with features
        if self.video_tag_features is not None and self.video_tag_features.shape[0] >= 2:
            cb_model = ContentBasedModel(self.train_df, self.video_tag_features)
            cb_model.build()
            self.models['content_based'] = cb_model
        
            hybrid_model = HybridRecommender(cf_model, cb_model)
            self.models['hybrid'] = hybrid_model
        else:
            print("Skipping Content-Based and Hybrid models: Insufficient videos/tag features.")
        
        print("\n✓ All relevant models attempted to train")
        
        return self
    
    def evaluate_models(self):
        """Evaluate all models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results = {}
        max_ndcg = -1
        best_name = None

        for name, model in self.models.items():
            print(f"\n--- {name.upper()} MODEL ---")
            results[name] = RecommenderEvaluator.evaluate_model(model, self.test_df, k=5) # Reduced K for small test set
            
            if results[name]['NDCG@K'] > max_ndcg:
                max_ndcg = results[name]['NDCG@K']
                best_name = name
        
        self.best_model = self.models.get(best_name)
        
        if self.best_model:
            print(f"\n✓ Best performing model: {best_name.upper()}")
        else:
            print("\n✗ Could not select a best model (No models trained/evaluated).")
            
        return results
    
    def generate_recommendations(self, user_id, n=5):
        """Generate recommendations for a user"""
        if self.best_model is None or self.df is None:
            raise ValueError("Models not trained or data not loaded!")
        
        recommendations = self.best_model.recommend(user_id, n=n)
        
        video_map = self.df[['video_id', 'category', 'tags']].drop_duplicates(subset=['video_id']).set_index('video_id')
        
        rec_details = []
        for video_id in recommendations:
            if video_id in video_map.index:
                video_info = video_map.loc[video_id]
                rec_details.append({
                    'video_id': video_id,
                    'category': video_info['category'],
                    'tags': video_info['tags']
                })
        
        return rec_details
    
    def run_pipeline(self):
        """Execute full pipeline"""
        self.load_data()
        self.explore_data()
        self.engineer_features()
        self.split_data()
        self.train_models()
        results = self.evaluate_models()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        return self

# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def demonstrate_system(system):
    """Demonstrate the recommendation system"""
    
    print("\n" + "="*60)
    print("EXAMPLE RECOMMENDATIONS")
    print("="*60 + "\n")
    
    if system.train_df is None or system.test_df is None:
        print("Cannot demonstrate: Data split failed.")
        return

    # Get a list of users present in both train and test (to ensure they have a history)
    train_users = set(system.train_df['user_id'].unique())
    test_users = set(system.test_df['user_id'].unique())
    sample_users = sorted(list(train_users.intersection(test_users)))

    if not sample_users:
        # Fallback to any user in the dataset if the split created no overlapping users
        sample_users = system.df['user_id'].unique()[:3]
    else:
        sample_users = sample_users[:3] # Take top 3 overlapping users
    
    if not system.best_model:
        print("Cannot demonstrate: Best model not trained/selected.")
        return

    best_model_name = system.best_model.__class__.__name__
    
    for user in sample_users:
        print(f"\n📺 Recommendations for {user} (using {best_model_name}):")
        print("-" * 60)
        
        try:
            recommendations = system.generate_recommendations(user, n=5)
            
            if not recommendations:
                print("No recommendations found for this user.")
                continue

            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. Video ID: {rec['video_id']}")
                print(f"   Category: {rec['category']}")
                print(f"   Tags: {rec['tags'][:70]}...")
                print()
        except Exception as e:
            print(f"Error generating recommendations for {user}: {e}")


def display_recommendations_output(user_id, recommendations):
    """
    Formats and displays the list of recommendation dictionaries in a clear, tabular format.
    
    Args:
        user_id (str): The ID of the user.
        recommendations (list): A list of dictionaries, where each dict has 
                                'video_id', 'category', and 'tags'.
    """
    print(f"\n📺 Top Recommendations for User: {user_id}")
    print("--------------------------------------------------")
    
    if not recommendations:
        print("No recommendations found.")
        return

    # Convert the list of dicts to a DataFrame for easy formatting
    df = pd.DataFrame(recommendations)
    
    # Add Rank column as the first column
    df.insert(0, 'Rank', range(1, 1 + len(df)))
    
    # Clean up column names for display
    df.columns = ['Rank', 'Video ID', 'Category', 'Tags']
    
    # Print the DataFrame using to_string() for clean console output
    print(df.to_string(index=False))

# ============================================================================
# MAIN EXECUTION (Updated to use the user's file)
# ============================================================================

file_name = "MLAssignmen-VideoRecommendation-Dataset.csv"
# Instantiate the system with the uploaded file path
rec_system = VideoRecommendationSystem(data_path=file_name) 
# Run the complete pipeline
rec_system.run_pipeline() 

# Demonstrate with sample users
target_user = str(input("\nPlease enter your user id:\n")) #example: U001
new_recs = rec_system.generate_recommendations(target_user, n=5)

display_recommendations_output(target_user, new_recs)