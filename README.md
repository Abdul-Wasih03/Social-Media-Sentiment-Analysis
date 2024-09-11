# <p align="center"> Social Media Sentiment Analysis </p> 
## Project Overview:
This project involves analyzing social media sentiment from a dataset containing comments from various platforms (Twitter, Instagram, Facebook). The goal is to understand public sentiment towards specific topics, products, or events using natural language processing (NLP) techniques.


## Requirements:
<li>Python 3.x
<li>pandas
<li>numpy
<li>matplotlib
<li>seaborn
<li>spacy


## Dataset:
The dataset includes 732 entries with the following columns:

Text: The content of the social media comment.<br>
Sentiment: The sentiment label associated with the comment.<br>
Timestamp: The date and time of the comment.<br>
User: The user who posted the comment.<br>
Platform: The social media platform (Twitter, Instagram, Facebook).<br>
Hashtags: The hashtags used in the comment.<br>
Retweets: Number of retweets (for Twitter).<br>
Likes: Number of likes.<br>
Country: The country of the user.<br>
Year, Month, Day, Hour: Date and time details.<br>

## Steps:
#### 1. Read the Dataset
To get started, read the dataset into a pandas DataFrame. Ensure that you handle any potential issues with the data, such as inconsistent column names or extra spaces in categorical values.

#### 2. Data Cleaning
Removed extra spaces and cleaned sentiment labels to standardize them into three categories: Positive, Negative, and Neutral.

#### 3. Text Preprocessing
Applied text preprocessing using spaCy:<br>
Converted text to lowercase.<br>
Removed URLs, special characters, and numbers.<br>
Tokenized text and removed stopwords.<br>
Lemmatized words to reduce them to their base form.<br>

#### 4.Sentiment Mapping
Mapped various sentiment labels to three main categories for simplified analysis.

#### 5. Data Visualization
Sentiment Distribution Across Platforms: Visualized how sentiments are distributed across different social media platforms.<br>
Sentiment Trends Over Time: Analyzed sentiment trends over time (by year and month).<br>
Sentiment Analysis by Country: Examined sentiment distribution across different countries.<br>4

# Program:

```python
# Import necessary libraries
import pandas as pd
# Load the dataset
# Replace 'your_dataset.csv' with the actual file name or path of your dataset
df = pd.read_csv('sentimentdataset.csv')
```
```python
# Display the first few rows of the dataset to understand its structure
print("Dataset Preview:")
display(df.head())
```
![1](https://github.com/user-attachments/assets/efabf105-7e06-4bc3-925e-1c3a4a1c3666)

```python
# Get basic information about the dataset (column names, data types, missing values)
print("\nDataset Information:")
df.info()
```
![2](https://github.com/user-attachments/assets/105707ba-8443-4814-9b75-a084d19511df)
```python
# Check for missing values in each column
print("\nMissing Values in Dataset:")
print(df.isnull().sum())
```
![3](https://github.com/user-attachments/assets/c7349610-24b7-4149-8669-5c4daaaee066)

```python
# Get a summary of numerical columns
print("\nStatistical Summary of Numerical Columns:")
display(df.describe())
```
![4](https://github.com/user-attachments/assets/940ed593-7b7f-4a63-87d6-636309e0b3a1)


```python
# Drop the unnecessary columns
df_cleaned = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# Remove leading/trailing spaces from 'Platform' column
df_cleaned['Platform'] = df_cleaned['Platform'].str.strip()

# Verify the changes
print("Cleaned Unique Platforms in the Dataset:")
print(df_cleaned['Platform'].unique())

# Display the first few rows of the cleaned dataset
display(df_cleaned.head())
```
![5](https://github.com/user-attachments/assets/7ceb4155-63ca-4918-a002-23bce33ba048)

```python
import spacy

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text using spaCy
def preprocess_text_spacy(text):
    # Process text with spaCy
    doc = nlp(text.lower())
    # Remove stopwords, punctuation, and lemmatize tokens
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(words)

# Apply the preprocessing to the 'Text' column
df_cleaned['Cleaned_Text'] = df_cleaned['Text'].apply(preprocess_text_spacy)

# Display the cleaned text data
print("Sample of Cleaned Text Data (spaCy):")
display(df_cleaned[['Text', 'Cleaned_Text']].head())
```
![6](https://github.com/user-attachments/assets/6bca9260-d6fe-4ffd-856d-3668906a8f02)

```python
# Plot the distribution of the new mapped sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='Mapped_Sentiment', data=df_cleaned, palette='coolwarm')
plt.title('Distribution of Mapped Sentiments')
plt.xlabel('Mapped Sentiment')
plt.ylabel('Count')
plt.show()

```
![7](https://github.com/user-attachments/assets/235435c7-354a-438c-93dc-7bfc170eedfe)

```python
# Plot sentiment distribution across platforms
plt.figure(figsize=(10, 6))
sns.countplot(x='Platform', hue='Mapped_Sentiment', data=df_cleaned, palette='coolwarm')
plt.title('Sentiment Distribution Across Platforms')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()

```
![8](https://github.com/user-attachments/assets/be8cd232-d0fa-449d-bbba-fe01d2f2d61b)

```python
# Sentiment trend over time (by Year and Month)
df_cleaned['Date'] = pd.to_datetime(df_cleaned[['Year', 'Month', 'Day']])

# Group by Date and Sentiment
sentiment_trends = df_cleaned.groupby(['Date', 'Mapped_Sentiment']).size().unstack(fill_value=0)

# Plot the trend
plt.figure(figsize=(12, 6))
sentiment_trends.plot(kind='line', figsize=(12, 6), marker='o')
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Comments')
plt.legend(title='Sentiment')
plt.show()

```
![9](https://github.com/user-attachments/assets/d72dc647-4129-4d55-b4eb-aca9eb61bd21)

```python
# Plot sentiment distribution across countries
plt.figure(figsize=(12, 8))
sns.countplot(x='Country', hue='Mapped_Sentiment', data=df_cleaned, palette='coolwarm', order=df_cleaned['Country'].value_counts().index)
plt.title('Sentiment Distribution Across Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.show()

```
![10](https://github.com/user-attachments/assets/51b0a9d5-49cf-43d4-ba60-682766d6528e)

## Applications
1. Public Sentiment Analysis:<br>
      Analyze social media comments to gauge public sentiment towards specific topics, products, or events.

2. Trend Analysis:<br>
      Track and visualize sentiment trends over time to identify shifts in public opinion.

3. Platform Comparison:<br>
      Compare sentiment distribution across different social media platforms (Twitter, Instagram, Facebook) to understand platform-specific reactions.

4. Geographical Insights:<br>
      Examine sentiment variations by country to uncover regional differences in public sentiment.

5. Hashtag Impact:<br>
      Investigate how specific hashtags correlate with sentiment to assess the impact of social media campaigns or trends.


## Result:
Thus, The Social Media Sentiment Analysis successfully provided insights into public sentiment trends, platform comparisons, geographical variations, and hashtag impacts.


## For a detailed analysis and visualizations, please refer to the attached Jupyter Notebook file.
