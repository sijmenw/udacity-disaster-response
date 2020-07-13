import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
import nltk.tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

import joblib


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}.db')
    df = pd.read_sql_table("disasterset", con=engine)
    X = list(df['message'])
    Y = df.iloc[:, 4:]
    col_names = Y.columns
    Y = np.array(Y)

    # necessary?
    # because of NaN values in Y, remove indices from 26207
    X = X[:26207]
    Y = Y[:26207]

    # use subset for speed
    X = X[:1500]
    Y = Y[:1500]

    return X, Y, col_names


lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")
stop_words += [',', '.', ':', '?', '!']


def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(x.lower()) for x in tokens if x not in stop_words]
    return tokens


class LengthTransformer(BaseEstimator, TransformerMixin):
    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X):
        pd.Series(X).apply(lambda x: 1 if len(x) > 100 else 0)


def build_model():
    pipeline = Pipeline([
        ('feat', FeatureUnion([
            ('pipe', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('length', LengthTransformer())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        "clf__estimator__n_estimators": [20, 50]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    for c in range(Y_test.shape[1]):
        print(f"Label: {category_names[c]}\n", classification_report(y_pred[:, c], Y_test[:, c]))


def save_model(model, model_filepath):
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
