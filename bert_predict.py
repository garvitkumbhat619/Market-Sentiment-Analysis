import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./models/bert_sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


def predict(tweet):

    inputs = tokenizer(
        tweet,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[0][prediction].item()

    return prediction, confidence


tweet = input("Enter tweet: ")

prediction, confidence = predict(tweet)

print("\nPrediction:", prediction)
print("Confidence:", round(confidence, 4))

if prediction == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")