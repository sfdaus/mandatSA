import csv
import pandas as pd
import dotenv
from data_cleansing import data_cleansing
import nltk
from nltk.data import find

# Download resource NLTK di sini saja
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
data_file['clean_text'] = texts.apply(data_cleansing)

# Lihat hasil
data_file.to_csv(
    "cleaned_output.csv",
    index=False,
    quoting=csv.QUOTE_ALL
)
