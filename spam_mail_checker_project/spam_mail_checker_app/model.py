import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('spam.csv')
df["spam"] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

pipe.fit(X_train, y_train)

def predict_spam(input_text):
    # Predict using the trained pipeline
    prediction = pipe.predict([input_text])
    
    # Convert prediction to human-readable format
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    
    return result

