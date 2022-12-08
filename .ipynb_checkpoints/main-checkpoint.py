from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import MySentimentModel
import pandas as pd

sia = SentimentIntensityAnalyzer()
myModel = MySentimentModel.MySentimentModel()

def robertaModel(sentence):
    model = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    encoded_text = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


def get_scores(content):
    myModelScores = myModel.sentimentAnalysis(content)[0]
    sia_scores = sia.polarity_scores(content)

    return pd.Series({
        'content': content,
        'my model': myModelScores,
        'nltk': sia_scores['compound'],
    })


def main():
    pd.set_option("display.max_colwidth", 400)
    df = pd.DataFrame({'content': [
        "I love love love love this kitten",
        "I hate hate hate hate this keyboard",
        "I'm not sure how I feel about toast",
        "Did you see the world cup game yesterday?",
        "The package was delivered late and the contents were broken",
        "Trashy television shows are some of my favorites",
        "I'm seeing a Kubrick film tomorrow, I hear not so great things about it.",
        "I find chirping birds irritating, but I know I'm not the only one",
    ]})
    scores = df.content.apply(get_scores)
    scores.style.background_gradient(cmap='RdYlGn', axis=None, low=0.4, high=0.4)
    print(scores)


main()










