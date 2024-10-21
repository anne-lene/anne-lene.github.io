---
layout: post
title: "Building a Movie Recommendation System with Python"
date: 2024-10-21
categories: [Data Science, Python]
tags: [Recommendation Systems, Python, Machine Learning]
---


# Building a Movie Recommendation System with Python: Key Concepts, Alternative Approaches, and Lessons Learned

Recommendation systems are a critical part of today’s digital platforms, helping users discover content, products, or media they may enjoy. In this post, I’ll take you through the process of building a **Movie Recommendation System** using Python. We'll explore concepts like **TF-IDF Vectorization** and **cosine similarity**, discuss alternative approaches to recommendation systems, and share insights on important development practices like **logging**, **testing**, and **code style (PEP8)**.

### Project Overview

This Movie Recommendation System suggests movies similar to a given title based on features such as genres, keywords, and cast. Using **TF-IDF Vectorizer**, we convert movie features into numerical representations and apply **cosine similarity** to identify movies that are close in "feature space." This project provides a basic yet effective system for content-based filtering, especially useful for small datasets.

### Key Concepts

#### 1. **Types of Recommendation Systems**
There are various types of recommendation systems. Here’s a brief overview of the most commonly used ones:

- **Content-Based Filtering**: Recommends items similar to those a user has interacted with. This approach relies solely on item features. Our project is an example of content-based filtering.
  
- **Collaborative Filtering**: Recommends items based on the behavior of similar users. Two main approaches are:
  - **User-based**: Recommends items based on the preferences of users with similar tastes.
  - **Item-based**: Recommends items that are similar to those a user has liked.

- **Hybrid Models**: Combines content-based and collaborative filtering to take advantage of both methods. For instance, Netflix uses a hybrid recommendation system to offer highly personalized suggestions based on both content features and user preferences.

#### When to Use Which Method?

- **Content-Based Filtering**: Ideal when the dataset is small or when you have detailed metadata about items but limited user interaction data.
- **Collaborative Filtering**: Suitable for large datasets with substantial user interaction data, such as ratings or purchase history. Works well for established platforms with many users.
- **Hybrid Models**: Best for complex recommendation systems where both item attributes and user behavior are key.

### 2. **TF-IDF Vectorization Explained**

**TF-IDF (Term Frequency - Inverse Document Frequency)** is a method used to represent text data in numerical form by measuring how important a word is in a document relative to a corpus. In this project, we use TF-IDF to process features like genres and keywords.

- **Term Frequency (TF)**: Measures how frequently a word occurs in a document.
- **Inverse Document Frequency (IDF)**: Downweighs commonly used words across many documents, assigning greater weight to rare but informative words.

Here's a visualization of how TF-IDF converts textual movie genres into vectors:

```
Movie 1: Action | Sci-Fi | Adventure
Movie 2: Drama | Adventure | Sci-Fi

TF-IDF matrix:
Action      Adventure   Drama   Sci-Fi
Movie 1    0.58        0.41     0       0.58
Movie 2    0           0.58     0.58    0.58
```

In this matrix, the similarity between movies can be calculated using **cosine similarity**.

### 3. **Cosine Similarity Explained**

Cosine similarity measures how similar two vectors are by calculating the cosine of the angle between them. The value ranges from -1 to 1, where 1 means the vectors are identical, and 0 means they are orthogonal (i.e., no similarity).

Here’s a graphical representation of cosine similarity between two movies:

```
Movie A -> Vector A (0.58, 0.41, 0.58)
Movie B -> Vector B (0.58, 0.58, 0)

Cosine Similarity = (A · B) / (||A|| * ||B||)
= (0.58*0.58 + 0.41*0.58) / (||A|| * ||B||)
≈ 0.82
```

### Project Structure

The project is structured as follows:

```
movie-recommendation/
├── main.py                # Core logic for loading movies, recommendations
├── tests/                 # Unit and integration tests
├── .env                   # Environment variables
├── pyproject.toml         # Poetry configuration
└── README.md              # Project documentation
```

### Code Implementation

The key components of the system include **loading data**, **vectorizing features**, and **calculating similarities**. Below are key parts of the code:

#### Movie Class
```python
class Movie:
    def __init__(self, title, genres, keywords, companies, popularity, release_date, runtime, cast, vote_count, vote_average):
        self.title = title
        self.genres = genres
        self.keywords = keywords
        self.companies = companies
        self.popularity = popularity
        self.release_date = pd.to_datetime(release_date)
        self.runtime = runtime
        self.cast = cast
        self.vote_count = vote_count
        self.vote_average = vote_average
```

#### TF-IDF and Cosine Similarity
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def feature_engineering(self):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'), stop_words='english', token_pattern=None)
    genres_matrix = vectorizer.fit_transform([movie.genres for movie in our.movies])
    combined_features = np.hstack([genres_matrix.toarray()])
    
def calculate_similarity(self, target_movie_title):
    similarity_scores = cosine_similarity([self.feature_matrix[target_idx]], self.feature_matrix)[0]
    return [self.movies[i].title for i in similarity_scores.argsort()[::-1][1:6]]
```

### Things I Have Learned

#### 1. **Logging**
Logging is an essential practice in software development, especially when handling errors or monitoring the system's behavior in production. In this project, logging was used to track file access and to help identify issues during data loading or recommendation calculations.

**How to Set Up Logging:**

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Movie recommendation system started.")
```

Logging helps maintain system transparency, making it easier to debug and maintain code.

#### 2. **Testing**
Writing unit tests is a critical step to ensure code reliability. This project uses **pytest** for testing individual components (e.g., data loading and similarity calculations). By writing tests, we can guarantee that changes in code won’t break the system unexpectedly.

**How to Set Up Tests:**

- Install `pytest` with `poetry add --dev pytest`.
- Write test cases in the `tests/` folder.
- Run tests using `poetry run pytest`.

#### 3. **PEP8 Code Style**
Adhering to **PEP8** guidelines ensures that your code is clean, readable, and maintainable. Some important practices include consistent indentation, meaningful variable names, and avoiding overly complex functions.

**How to Check PEP8 Compliance:**

You can use `flake8` to check for PEP8 compliance:
```bash
poetry add --dev flake8
poetry run flake8 main.py
```

### Future Improvements

- **Advanced Recommendation Algorithms**: Implement collaborative filtering or hybrid methods for more accurate recommendations.
- **Scalability**: Use approximate nearest neighbor search algorithms like **FAISS** for larger datasets.
- **Personalization**: Introduce user-based recommendations by incorporating user preferences, ratings, and behavior.
- **Web Interface**: Create a web interface to allow users to interact with the recommendation system in real-time.

### Conclusion

In this project, we built a basic **Movie Recommendation System** using content-based filtering, TF-IDF vectorization, and cosine similarity. While the system works well for smaller datasets, there are many areas for improvement, especially when scaling up or integrating more advanced techniques.

This project also provided valuable lessons in logging, testing, and adhering to PEP8 standards, which are essential for developing robust and maintainable code.

