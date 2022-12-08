from nltk.sentiment import SentimentIntensityAnalyzer
import MySentimentModel
import pandas as pd

sia = SentimentIntensityAnalyzer()
myModel = MySentimentModel.MySentimentModel()


def get_scores(content):
    myModelScores = myModel.sentimentAnalysis(content)[0]
    sia_scores = sia.polarity_scores(content)

    return pd.Series({
        'content': content,
        'nltk': sia_scores['compound'],
        'my model': myModelScores,
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
        "I do not dislike cabin cruisers",
        "Disliking people is not really my thing.",
        "I'd really truly love going out in this weather!",
    ]})
    scores = df.content.apply(get_scores)
    scores = scores.style.background_gradient(cmap='RdYlGn', axis=None, low=0.4, high=0.4)
    myModel.compareWithNLTK()

main()










