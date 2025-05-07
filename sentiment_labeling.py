from transformers import pipeline
import tqdm

# Load pipeline sentiment analysis IndoBERT (pakai pre-trained w11wo)
classifier = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")

def label_sentiment(texts):
    sentiments = []
    for text in tqdm.tqdm(texts):
        try:
            result = classifier(text[:512])[0]
            sentiments.append(result['label'])
        except Exception:
            sentiments.append("unknown")
    return sentiments