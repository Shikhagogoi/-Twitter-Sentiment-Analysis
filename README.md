# -Twitter-Sentiment-Analysis

This project explores sentiment analysis on real-world Twitter data to classify tweets as positive, negative, or neutral. Leveraging Natural Language Processing (NLP) techniques and machine learning models, the project aims to understand public opinion by analyzing tweet content.

1. Key Features
- Data preprocessing: cleaning, tokenization, stopword removal, stemming
- Text vectorization using TF-IDF
- Sentiment classification using Logistic Regression
- Model evaluation with accuracy, precision, recall, and F1-score
- Visualization of sentiment distribution using Matplotlib and Seaborn

2. Dataset
- The dataset used, includes pre-labeled 1.6 million tweets for training and testing the sentiment classifier. Tweets are collected from public Twitter data sources https://www.kaggle.com/datasets/kazanova/sentiment140.
https://github.com/Shikhagogoi/-Twitter-Sentiment-Analysis/blob/807cf0d2976cf1be1654e36420d4758111f2d5f7/Screenshot%20(14).png

3. Technologies Used
- Python, R
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib & Seaborn

 4. Results
- The model achieved decent performance in classifying sentiments, demonstrating how machine learning can be effectively applied to social media analytics.
  https://github.com/Shikhagogoi/-Twitter-Sentiment-Analysis/blob/a591702ab239c4fa81b6f0503487e9780c91da38/Screenshot%20(12).png
The Classification Report provides a detailed evaluation of the model’s performance on binary sentiment classification (0 = Negative, 1 = Positive):

Precision:
- Class 0 (Negative): 79%
- Class 1 (Positive): 77%
Precision indicates how many of the predicted positive/negative tweets were actually correct.

Recall:
- Class 0: 76%
- Class 1: 80%
Recall measures how well the model identifies all actual positive/negative tweets.

F1-Score:
- A harmonic mean of precision and recall. The model performs fairly balanced for both classes (~0.77–0.78), indicating no severe bias toward either.

Accuracy:
- Overall accuracy is 78%, meaning 78% of total predictions are correct.

Confusion Matrix:
- Confusion matrix indicates a good balance in classification, with more than 75% correctness in both classes.

https://github.com/Shikhagogoi/-Twitter-Sentiment-Analysis/blob/807cf0d2976cf1be1654e36420d4758111f2d5f7/Screenshot%20(13).png
The Receiver Operating Characteristic (ROC) curve demonstrates the model's ability to distinguish between positive and negative sentiments.
AUC Score: 0.86
A higher AUC reflects strong performance with good sensitivity and specificity.
