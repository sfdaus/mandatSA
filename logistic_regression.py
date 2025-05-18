from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,cohen_kappa_score
from sklearn.model_selection import train_test_split

config = dotenv.dotenv_values(".env")

df = pd.read_csv(config["FILE_LABELED"])
df_2 = pd.read_csv(config["FILE_LABELED_BERT"])
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
    solver='liblinear', C=1, random_state=42, class_weight='balanced',penalty='l2'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))

df_kappa = df_2[['label','label_manual']].dropna()
df_kappa['label'] = df_kappa['label'].astype(str)
df_kappa['label_manual'] = df_kappa['label_manual'].astype(str)

kappa = cohen_kappa_score(df_kappa['label'], df_kappa['label_manual'])
print(f"Cohen's κ = {kappa:.3f}")

# Compute confusion matrix
labels = model.classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Plot confusion matrix heatmap
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Jumlah prediksi', rotation=-90, va="bottom")

# Set tick labels
ax.set(
    xticks=np.arange(len(labels)), 
    yticks=np.arange(len(labels)),
    xticklabels=labels, 
    yticklabels=labels,
    xlabel='Prediksi', 
    ylabel='Label Sebenarnya',
    title='Confusion Matrix Heatmap'
)

# Rotate the tick labels and set alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate cells
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="black" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()