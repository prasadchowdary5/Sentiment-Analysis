# Sentiment Analysis Model

## Project Overview
This project aims to build a sentiment analysis model that can classify movie reviews as either positive or negative. The model is trained on a dataset containing thousands of movie reviews and uses machine learning techniques to make predictions based on the text of the reviews.

## Dataset
The dataset used in this project consists of movie reviews labeled as either positive or negative. The dataset is divided into two categories:
- **Positive Reviews**: These are reviews that express a favorable opinion about a movie.
- **Negative Reviews**: These are reviews that express an unfavorable opinion about a movie.

The data was loaded and combined into a single DataFrame, which was then used for model training and evaluation.

## Model Building
The model was built using the following steps:
1. **Data Loading**: Reviews were loaded from the local file system and combined into a single DataFrame.
2. **Data Splitting**: The dataset was split into training and testing sets using an 80/20 split.
3. **Feature Extraction**: The text data was converted into numerical features using TF-IDF Vectorization.
4. **Model Training**: A Logistic Regression model was trained on the vectorized text data.
5. **Model Evaluation**: The model was evaluated on the test set, and its accuracy and classification report were generated.

### Libraries Used
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning tasks including model building, feature extraction, and evaluation.
- **Joblib**: For saving and loading the trained model and vectorizer.
- **Streamlit**: For deploying the sentiment analysis model as a web application.

## Usage
To use the sentiment analysis model, follow these steps:

1. **Clone the Repository**: Clone the project repository to your local machine.

    ```bash
    git clone https://github.com/your-repository-url/sentiment-analysis.git
    ```

2. **Install Dependencies**: Install the required Python packages using pip.

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**: Start the Streamlit application to analyze movie reviews in real-time.

    ```bash
    streamlit run main.py
    ```

4. **Analyze a Review**: Enter a movie review in the input box on the Streamlit interface and click the "Analyze" button. The model will predict whether the review is positive or negative and display the result.

## Model Evaluation
The Logistic Regression model was evaluated on a test set, and the following metrics were obtained:
- **Accuracy**: [Insert accuracy score]
- **Classification Report**: The detailed classification report provides insights into the model's precision, recall, and F1-score for both positive and negative classes.

## Future Work
- **Hyperparameter Tuning**: Explore techniques like GridSearchCV or RandomizedSearchCV to optimize the model's hyperparameters.
- **K-Fold Cross-Validation**: Implement cross-validation to assess the model's performance more robustly.
- **Deployment**: Explore deploying the model using cloud platforms like AWS, Azure, or Google Cloud.

## Conclusion
This project successfully demonstrates the use of machine learning techniques for sentiment analysis on movie reviews. The deployed model provides real-time sentiment predictions and can be further refined for better accuracy and performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
