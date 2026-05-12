# Sentiment-Analysis

### Objective
Develop machine learning models to classify emotions in text samples.


### Dataset
The dataset used for this assignment is `nlp_dataset.csv`, containing text comments and their corresponding emotions. It was loaded from a Google Drive link.

### Key Steps Performed

1.  **Loading and Preprocessing**
    *   Loaded the `nlp_dataset.csv` into a Pandas DataFrame.
    *   Downloaded NLTK resources ('punkt' and 'stopwords').
    *   Performed text preprocessing, including tokenization, converting text to lowercase, removing non-alphabetic tokens, and filtering English stopwords. This created a new `cleaned_comment` column.

2.  **Feature Extraction**
    *   Used `TfidfVectorizer` from `sklearn.feature_extraction.text` to convert the `cleaned_comment` text into a numerical feature matrix (`X`). The resulting matrix has a shape of (5937, 8815), representing 5937 comments and 8815 unique terms (features).

3.  **Model Development and Training**
    *   Split the dataset into training and testing sets (80% train, 20% test) using `train_test_split`.
    *   **Multinomial Naive Bayes Model:** An instance of `MultinomialNB` was trained on the `X_train` and `y_train` data.
    *   **Support Vector Machine (SVM) Model:** An instance of `SVC` with a linear kernel was trained on the `X_train` and `y_train` data.

4.  **Model Evaluation**
    *   Both models were evaluated on the `X_test` data using `accuracy_score`, `precision_score`, `recall_score`, and `f1_score`.
    *   **Naive Bayes Model Performance:**
        *   Accuracy: 0.9116
        *   Precision: 0.9134
        *   Recall: 0.9116
        *   F1-Score: 0.9115
    *   **Support Vector Machine (SVM) Model Performance:**
        *   Accuracy: 0.9470
        *   Precision: 0.9477
        *   Recall: 0.9470
        *   F1-Score: 0.9469

5.  **Confusion Matrix Visualization**
    *   Calculated and visualized the confusion matrix for the SVM model using `seaborn.heatmap`.

### Key Findings & Insights

*   **Preprocessing Impact:** Tokenization and stopword removal effectively reduced noise and dimensionality, helping the models focus on more meaningful terms.
*   **Model Performance:** Both models performed reasonably well, with the SVM model showing slightly superior performance across accuracy, precision, recall, and F1-score compared to the Naive Bayes model for this specific dataset.
*   **Confusion Matrix Analysis (SVM):** The confusion matrix for the SVM model indicated strong performance, with high true positive counts along the diagonal and relatively low misclassifications. For example:
    *   375 instances of the first emotion were correctly predicted.
    *   380 instances of the second emotion were correctly predicted.
    *   370 instances of the third emotion were correctly predicted.
    *   The low off-diagonal values suggest good discriminative power between the emotion classes.

### Next Steps

*   **Hyperparameter Tuning:** Further optimize both Naive Bayes and SVM models through hyperparameter tuning (e.g., using GridSearchCV or RandomizedSearchCV).
*   **Advanced Feature Engineering:** Explore n-grams, word embeddings (Word2Vec, GloVe), or contextual embeddings (BERT, ELMo) for richer text representation.
*   **Other Models:** Experiment with more advanced machine learning models (e.g., Logistic Regression, Random Forest, Gradient Boosting) or deep learning architectures (e.g., LSTMs, Transformers) to potentially achieve higher accuracy and robustness.
