# News Classification Using Machine Learning

## Aim

To develop a machine learning model to classify news articles based on their textual content using Natural Language Processing and machine learning techniques.

## Dataset Description

The dataset contains news articles with the following columns:

- **title** – Title of the news article
- **text** – Full text/content of the article
- **subject** – Subject or topic of the article
- **date** – Publication date
- **label** – Target class of the news article

The `title` and `text` columns are combined and used as input features, while the `label` column is used as the target variable.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression

## Methodology

1. Load the news dataset.
2. Check and handle missing values.
3. Combine the title and text columns.
4. Split the dataset into training and testing data.
5. Convert the text into numerical features using TF-IDF.
6. Train a Logistic Regression classification model.
7. Predict the labels for the test data.
8. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.

## Model Evaluation Results

The Logistic Regression model achieved approximately **97.17% accuracy** on the test dataset.

The model also achieved good precision, recall, and F1-score, indicating that it can effectively classify the news articles.

## Conclusion

The experiment successfully demonstrated the use of Natural Language Processing and machine learning for news classification. TF-IDF was used to convert text into numerical features, and Logistic Regression was used for classification. The high accuracy shows that the model performs effectively on the given dataset.

## Files

- `news_classification.py` – Python implementation of the machine learning model.
- `news_dataset.csv` – Dataset used for training and testing the model.
