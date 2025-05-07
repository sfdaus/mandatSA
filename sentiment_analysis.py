import csv
import pandas as pd
import dotenv
from tqdm.auto import tqdm
from data_cleansing import data_cleansing
import nltk
from nltk.data import find
from sentiment_labeling import label_sentiment

tqdm.pandas()

# Download resource NLTK
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

#import ENV
config = dotenv.dotenv_values(".env")

data_file = pd.read_csv(config["FILE_PATH"])

# Ambil kolom full_text
texts = data_file['full_text'].astype(str)

# Terapkan ke semua data
print("#### Data Cleaning ####")
data_file['clean_text'] = texts.progress_apply(data_cleansing)

# Labeling otomatis dengan BERT
print("#### Sentiment Labeling BERT ####")
data_file['sentiment'] = label_sentiment(data_file['clean_text'])

# Lihat hasil
data_file.to_csv(
    "cleaned_output_with_labeling.csv",
    index=False,
    quoting=csv.QUOTE_ALL
)
