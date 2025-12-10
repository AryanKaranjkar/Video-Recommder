# Video Recommendation System - ML Assignment

## 📋 Project Overview

This project implements a machine learning-based video recommendation system that suggests videos to users based on their historical engagement patterns including views, likes, comments, watch duration, and video metadata.

## 🎯 Key Features

- **Multiple ML Approaches**: Collaborative Filtering, Content-Based Filtering, and Hybrid Model
- **Comprehensive Feature Engineering**: Engagement scoring, tag embeddings (TF-IDF), user preference profiling
- **Robust Evaluation**: Precision@K, Recall@K, MRR, and NDCG@K metrics
- **Auto-dependency Installation**: Automatic package installation on first run
- **Synthetic Data Generation**: Built-in dataset generator for testing
- **Clean Code Architecture**: Modular, well-documented, reusable components

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Execution

1. **Clone or download the project files**

2. **Install dependencies automatically** (Option 1):
   ```bash
   python video_recommendation_system.py
   ```
   The script will auto-install all required packages on first run.

3. **Manual installation** (Option 2):
   ```bash
   pip install -r requirements.txt
   python video_recommendation_system.py
   ```

### Using Your Own Dataset

```python
from video_recommendation_system import VideoRecommendationSystem

# Initialize with your CSV file
system = VideoRecommendationSystem(data_path='your_dataset.csv')

# Run the complete pipeline
system.run_pipeline()

# Generate recommendations
recommendations = system.generate_recommendations('user_123', n=10)
print(recommendations)
```

## 📊 Dataset Format

Your CSV file should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Unique user identifier |
| video_id | string | Unique video identifier |
| category | string | Video category |
| tags | string | Comma-separated tags |
| watch_duration | float | Percentage watched (0-100) |
| liked | int | 1 if liked, 0 otherwise |
| commented | int | 1 if commented, 0 otherwise |
| subscribed_after_watching | int | 1 if subscribed, 0 otherwise |
| timestamp | datetime | Interaction timestamp |

## 🏗️ Project Structure

```
├── video_recommendation_system.py  # Main implementation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── report.md                       # Detailed project report
```

## 🧠 Model Approaches

### 1. Collaborative Filtering
- Uses user-user similarity based on interaction patterns
- Recommends videos liked by similar users
- Effective for discovering popular content

### 2. Content-Based Filtering
- Analyzes video features (tags, categories)
- Recommends similar videos to user's watch history
- Good for personalized, relevant suggestions

### 3. Hybrid Model (Recommended)
- Combines both approaches with weighted scores
- Default weights: 60% collaborative, 40% content-based
- Best overall performance in testing

## 📈 Evaluation Metrics

The system evaluates models using:

- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant videos in top-K
- **MRR (Mean Reciprocal Rank)**: Position of first relevant recommendation
- **NDCG@K**: Normalized discounted cumulative gain

## 🎓 Key Implementation Highlights

1. **Engagement Score Formula**:
   ```
   score = watch_duration × 0.4 + liked × 20 + commented × 30 + subscribed × 50
   ```

2. **TF-IDF Tag Embeddings**: Converts video tags into numerical features

3. **Temporal Train/Test Split**: Uses time-based split for realistic evaluation

4. **Error Handling**: Comprehensive validation and error management

## 💡 Usage Examples

```python
# Initialize and train
system = VideoRecommendationSystem()
system.run_pipeline()

# Get recommendations for a user
recs = system.generate_recommendations('user_42', n=10)

# Access individual models
cf_recommendations = system.models['collaborative'].recommend('user_42', n=10)
cb_recommendations = system.models['content_based'].recommend('user_42', n=10)

# View evaluation results
print(system.eda_stats)
```

## 🔧 Customization

### Adjust Hybrid Model Weights
```python
from video_recommendation_system import HybridRecommender

hybrid = HybridRecommender(
    cf_model=system.models['collaborative'],
    cb_model=system.models['content_based'],
    cf_weight=0.7  # 70% collaborative, 30% content-based
)
```

### Change Evaluation Parameters
```python
results = RecommenderEvaluator.evaluate_model(
    model=system.best_model,
    test_data=system.test_df,
    k=20  # Evaluate top-20 recommendations
)
```

## 📦 Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- lightgbm >= 3.3.0
- implicit >= 0.7.0

## 🐛 Troubleshooting

**Issue**: Module not found errors
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Dataset validation fails
- **Solution**: Ensure your CSV has all required columns and no empty critical values

**Issue**: Out of memory errors
- **Solution**: Reduce dataset size or adjust evaluation sample size in code

## 📝 Notes

- Random seed is set to 42 for reproducibility
- Synthetic data generation creates realistic interaction patterns
- System automatically handles missing values and data cleaning
- Temporal split ensures no data leakage in evaluation

## 👤 Author

ML Assignment Solution - Video Recommendation System

## 📄 License

This project is created for educational purposes as part of a Machine Learning assignment.

## 🙏 Acknowledgments

- Assignment specification provided by the course instructor
- Implemented following ML best practices and clean code principles