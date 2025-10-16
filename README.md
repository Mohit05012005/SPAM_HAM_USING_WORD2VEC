📧 Spam vs Ham Classification using Word2Vec & Random Forest (Google Colab)

This project demonstrates how to classify text messages as Spam or Ham (Non-Spam) using Word2Vec embeddings and a Random Forest Classifier in Google Colab.
It showcases how Natural Language Processing (NLP) and Machine Learning techniques can automatically filter out spam messages based on text patterns and word meanings.

🧠 Project Overview

The main goal of this project is to build a machine learning model that detects spam messages.
Instead of using traditional Bag-of-Words or TF-IDF features, this project uses Word2Vec embeddings to capture semantic relationships between words — improving the model’s understanding of text.

🧩 Workflow

Environment Setup (in Google Colab)

Install required libraries

Import modules for NLP and ML

Mount Google Drive (if dataset stored there)

Load Dataset

Typically uses the SMS Spam Collection Dataset

Columns:

label → “spam” or “ham”

message → actual text message

Data Preprocessing

Convert text to lowercase

Remove punctuation, special characters, and numbers

Remove stopwords

Tokenize messages using nltk.word_tokenize()

Word2Vec Embedding

Train Word2Vec model using gensim.models.Word2Vec

Convert each message to a vector by averaging all word vectors in that message

Model Training

Use RandomForestClassifier from sklearn.ensemble

Train on the feature vectors generated from Word2Vec

Use train_test_split() for 80-20 data division

Model Evaluation

Evaluate using:

Accuracy

Precision

Recall

F1-Score

Visualize performance with a Confusion Matrix

Prediction

Test model with new text messages (e.g., “You’ve won a prize!” → Spam)

🧰 Technologies & Libraries
Category	Tools Used
Platform	Google Colab
Language	Python
NLP	NLTK, Gensim
ML Algorithm	RandomForestClassifier (Scikit-learn)
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
📊 Example Output
Message	Prediction
“Congratulations! You’ve won a free ticket!”	Spam
“Hey, are you coming to class today?”	Ham
🚀 How to Run in Google Colab

Upload your notebook file:
SPAM_HAM_USING_WORD2VEC.ipynb

Upload or mount the dataset (e.g., spam.csv)

Install dependencies (if not pre-installed):

!pip install pandas numpy nltk gensim scikit-learn matplotlib seaborn


Run all cells in order:

Text preprocessing

Word2Vec training

Random Forest training

Evaluation and predictions

🧪 Results Summary

Model: Random Forest

Embedding: Word2Vec (trained on SMS corpus)

Performance: High accuracy and balanced precision/recall

Word2Vec captured contextual relationships better than simple word counts.

💡 Future Improvements

Use pre-trained embeddings (Google News Word2Vec, GloVe)

Tune Random Forest hyperparameters for optimal results

Compare with deep learning models (LSTM, BERT)

Build a Streamlit app for real-time spam detection

👨‍💻 Author

Mohit Bohra
📧 [bohramohit93199@gmail.com
]
🌐 GitHub Profile
📧 [https://github.com/Mohit05012005/
]
🌐 GitHub Profile
