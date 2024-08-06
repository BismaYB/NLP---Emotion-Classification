# NLP Emotion-Classification

## This project aims to develop machine learning models to classify emotions in text samples using.

## Key Components

## 1. Loading and Preprocessing
   
Loading the Dataset:

The dataset is loaded from a CSV file into a pandas DataFrame.

Preprocessing Steps:

Removing Punctuation and Numbers: Unnecessary characters such as punctuation and numbers are removed to reduce noise in the data.
Converting to Lowercase: All text is converted to lowercase to ensure uniformity and consistency.

Tokenization: The text is split into individual words (tokens), which allows for detailed analysis.

Removing Stopwords: Common words that do not add much meaning, such as "and", "the", and "is", are removed. This step focuses on the words that carry significant meaning.

Impact: These preprocessing techniques help clean the text data, making it more suitable for machine learning models. By focusing on meaningful words and reducing noise, we improve the quality of the data, which enhances model performance.

## 2. Feature Extraction
   
TF-IDF Vectorization:

The TF-IDF (Term Frequency-Inverse Document Frequency) method is used to transform the cleaned text into numerical features.
Term Frequency (TF): Measures how frequently a word appears in a document.
Inverse Document Frequency (IDF): Measures how important a word is by considering its occurrence across all documents.
TF-IDF Score: The product of TF and IDF, indicating the importance of a word in a document relative to the entire dataset.
Impact: TF-IDF converts text data into numerical features by considering the importance of words, making it suitable for machine learning algorithms to process.

## 3. Model Development

Models Used:

Naive Bayes: A simple and efficient probabilistic classifier that assumes independence between features. It is particularly effective for text data and works well with high-dimensional data.

Support Vector Machine (SVM): A powerful classifier that finds the optimal hyperplane to separate different classes. It is effective in high-dimensional spaces and provides strong generalization capabilities.

Impact: Both models are trained on the extracted features to classify emotions in text samples.

## 4. Model Comparison

Evaluation Metrics:

Accuracy: Measures the proportion of correct predictions.

F1-Score: Considers both precision and recall to provide a balanced evaluation of the model's performance.

Impact: Evaluating the models using these metrics helps determine their effectiveness in classifying emotions. The model with higher accuracy and F1-score is considered better suited for the task.

## Conclusion

By following these steps, we successfully preprocess the text data, extract meaningful features, develop and evaluate machine learning models for emotion classification. This project demonstrates the effectiveness of preprocessing techniques and model selection in achieving accurate and reliable emotion classification from text samples.

### Chosen Models and Their Suitability for Emotion Classification

#### Naive Bayes

Model Description: Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming independence between features.

Text Data: Works well with text data where the assumption of feature independence is often reasonable.

Speed and Efficiency: Fast to train and predict, making it suitable for large datasets.

Performance: Often performs well with sparse data, which is common in text classification tasks like emotion classification.

#### Support Vector Machine (SVM)

Model Description: SVM is a powerful classifier that finds the hyperplane that best separates data into classes.

High Dimensional Spaces: Effective in high-dimensional spaces and with sparse data, typical of text features.

Margin Maximization: Focuses on maximizing the margin between classes, which can lead to better generalization on unseen data.

Versatility: Can handle non-linear classification through the use of kernel functions, making it adaptable to complex relationships in the data.

Naive Bayes is suitable for emotion classification due to its simplicity, efficiency, and good performance with text data.

SVM is suitable because of its effectiveness in high-dimensional and sparse datasets, and its strong generalization capabilities.

                   Model  Accuracy  F1-Score
            Naive Bayes  0.915825  0.915876
 Support Vector Machine  0.943603  0.943602

In this model SVM have high accuracy  with 0.943603  and f1-score  with 0.943602


