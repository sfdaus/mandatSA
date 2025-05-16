import pandas as pd
import dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split 

config = dotenv.dotenv_values(".env")

df = pd.read_csv(config["FILE_LABELED"])
df = df.dropna(subset=['clean_text', 'label'])

# 2. Vectorization: ubah teks menjadi angka menggunakan TF–IDF
vectorizer = TfidfVectorizer(
    max_features=5000,  
    ngram_range=(1,2),   
    min_df=1             
)
x = vectorizer.fit_transform(df['clean_text'])
print("Fitur akhir:", x.shape[1])
y = df['label']  

X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,      
    stratify=y,         
    random_state=42
)

model = LogisticRegression(
    solver='liblinear', C=1.0, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
