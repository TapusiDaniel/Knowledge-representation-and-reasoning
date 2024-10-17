import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import argparse

def load_data(data_file, names_file):
    # Read the names file to get feature names
    with open(names_file, 'r') as f:
        features = [line.split(':')[0].strip() for line in f if ':' in line]
    
    # The class (target) variable is not included in the .names file
    # It's the last column in the data file, so we'll add it manually
    columns = features + ['class']
    
    # Read the data file
    df = pd.read_csv(data_file, names=columns, index_col=False)
    return df

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
    
    def calculate_class_priors(self, y):
        """
        Calculate the prior probabilities of each class.
        """
        ## TODO: Implement this method
        unique_classes = np.unique(y) # computes a list of unique class labels
        total_number = len(y) # computes the total number of samples

        for cls in unique_classes:
            counter = 0
            for sample in y: # counts the number of samples in each class
                if sample == cls:
                    counter += 1
            self.class_priors[cls] = counter / total_number # calculates the prior probability of each class 
            #using the formula: P(C = c) = # samples in class c / # total nuber of samples
    
    def calculate_feature_probs(self, X, y):
        """
        Calculate the conditional probabilities of feature values given a class.
        """
        ## TODO: Implement this method
        unique_classes = np.unique(y)
        features = X.columns
        alpha = 1 # Laplace smoothing

        for cls in unique_classes:
            self.feature_probs[cls] = {}
            class_samples = X[y == cls] # all samples that are in class c
            class_count = len(class_samples) # total number of values of feature Xi in class c

            for feature in features:
                self.feature_probs[cls][feature] = {}
                feature_values = X[feature].unique() # all unique values

                for value in feature_values:
                    counter = (class_samples[feature] == value).sum() # number of apparitions of feature xi in samples of class c
                    smoothed_count = counter + alpha # apply laplace smoothing
                    smoothed_total = class_count + len(feature_values) * alpha
                    self.feature_probs[cls][feature][value] = smoothed_count / smoothed_total # calculates the conditional probability of using the formula:
                    #P(Xi = xi | C = c) = # apparitions of feature xi in samples of class c + alpha / 
                    # # total number of values of feature Xi in class c + # num unique values of feature X * alpha

    
    def fit(self, X, y):
        """
        Train the Naive Bayes model.
        """
        ## TODO: Implement this method
        self.calculate_class_priors(y)
        self.calculate_feature_probs(X, y)
        
    def predict(self, X):
        """
        Make predictions for the given samples.
        """
        ## TODO: Implement this method
        predictions = []
        classes = list(self.class_priors.keys()) # create a list of all the classes
        
        for _, sample in X.iterrows():
            class_scores = {}
            for cls in classes:
                score = np.log(self.class_priors[cls]) # log of the prior probability of a class
                
                for feature, value in sample.items():
                    score += np.log(self.feature_probs[cls][feature][value]) # adding to the total score the logarithm of the conditional probabilities for each feature value
                
                class_scores[cls] = score
            
            predicted_class = max(class_scores, key=class_scores.get) # choosing the class with the highest score for this sample
            predictions.append(predicted_class)
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier for UCI C4.5 format data')
    parser.add_argument('--names-file', required=True, help='Path to the .names file containing column information')
    parser.add_argument('--data-file', required=True, help='Path to the .data file containing the dataset')
    args = parser.parse_args()

    df = load_data(args.data_file, args.names_file)
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)
    class_order = ['not_recom', 'recommend', 'very_recom', 'priority', 'spec_prior']

    print("\nGeneral Accuracy:")
    print(f"\nGeneral Accuracy: {round(100 * (y_test == y_pred).mean(), 2)}%")

    cr = classification_report(y_test, y_pred, labels=class_order, zero_division=1)
    print("\nClassification Report:")
    print(cr)

    cm = confusion_matrix(y_test, y_pred, labels=class_order)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()