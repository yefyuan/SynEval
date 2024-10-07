import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import Counter
import re

def read_reviews(filename):
    with open(filename, 'r') as file:
        reviews = file.readlines()
    return [review.strip() for review in reviews]

def sentiment_analysis(reviews):
    sentiments = [TextBlob(review).sentiment.polarity for review in reviews]
    categorized_sentiments = ['positive' if score > 0 else 'negative' if score < 0 else 'neutral' for score in sentiments]
    sentiment_counts = Counter(categorized_sentiments)
    total_reviews = len(reviews)
    most_common_sentiment, most_common_count = sentiment_counts.most_common(1)[0]
    percentage = (most_common_count / total_reviews) * 100
    return most_common_sentiment, percentage, sentiment_counts

def keyword_analysis(reviews):
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for review in reviews for word in word_tokenize(review) if word.isalpha() and word.lower() not in stop_words]
    frequency = Counter(words)
    top_keywords = frequency.most_common(3)

    sentiment_words = []
    for review in reviews:
        tokens = word_tokenize(review)
        tagged = nltk.pos_tag(tokens)
        for word, tag in tagged:
            if tag.startswith('JJ'):  # adjective
                sentiment_words.append(word.lower())

    sentiment_word_frequency = Counter(sentiment_words)
    top_sentiment_words = sentiment_word_frequency.most_common(3)
    return top_keywords, top_sentiment_words

def average_review_length(reviews):
    lengths = [len(review.split()) for review in reviews]
    average_length = sum(lengths) / len(lengths)
    return average_length

def main():
    filename = 'Data/reviews_example.txt'
    reviews = read_reviews(filename)

    # Sentiment Analysis
    most_common_sentiment, percentage, sentiment_counts = sentiment_analysis(reviews)
    print(f"Most common sentiment: {most_common_sentiment} ({percentage:.2f}%)")
    print("Sentiment distribution:", sentiment_counts)

    # Keyword Analysis
    print("Top 3 Keywords and their counts:")
    top_keywords, top_sentiment_words = keyword_analysis(reviews)
    for keyword, count in top_keywords:
        print(f"{keyword}: {count} appearances")
    
    print("Top 3 Sentiment-related Words and their counts:")
    for word, count in top_sentiment_words:
        print(f"{word}: {count} appearances")

    # Average Review Length
    avg_length = average_review_length(reviews)
    print(f"Average review length: {avg_length} words")

if __name__ == "__main__":
    main()

