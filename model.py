from transformers import pipeline

# Downloads the model
# The file has weights determined through training
# Outputs two numbers, 1 for how positive the other for how negative, higher becomes the label
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze(text):
    #Runs text through the AI Model, and classifier()[0] returns a list,
    #grabs the first item, a dictionary containing the label and score
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    return label, score


# runs only if you execute model.py directly (__name__ is set to __main__ if you run pytho3 model.py)
if __name__ == "__main__":
    test_sentences = [
        "I love this, it's amazing!",
        "This is the worst thing ever.",
        "It was okay, nothing special."
    ]
    for sentence in test_sentences:
        label, score = analyze(sentence)
        print(f"{sentence}")
        print(f"  → {label} ({score:.2%} confidence)\n")