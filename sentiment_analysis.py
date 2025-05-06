import csv
import pandas as pd
import dotenv
from data_cleansing import data_cleansing
import nltk

# Download resource NLTK di sini saja
nltk.download('stopwords')
nltk.download('punkt')

#import ENV
config = dotenv.dotenv_values(".env")

dataset_columns = ['full_text', 'created_at', 'username', 'location', 'quote_count', 'reply_count', 'retweet_count', 
                    'favorite_count', 'mentions', 'mention_count', 'source', 'target']


data_file = pd.read_csv(config["FILE_PATH"], encoding=config["DATASET_ENCODING"], names=dataset_columns)

# Ambil kolom full_text
texts = data_file['full_text'].astype(str)
print("texts")
print(data_cleansing(texts))
# Terapkan ke semua data
data_file['clean_text'] = texts.apply(data_cleansing)
print("data_file['clean_text']")
print(data_file['clean_text'])

# Lihat hasil
print(data_file[['full_text', 'clean_text']].head())
data_file[['full_text', 'clean_text']].to_csv(
    "cleaned_output.csv",
    index=False,
    quoting=csv.QUOTE_ALL
)