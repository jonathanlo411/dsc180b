from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient import discovery
from time import sleep

class ModelCollection:
    
    
    def __init__(self, gcp_api_key, rate_limit_timeout = 1):
        self.perspective_client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=gcp_api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False
        )
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.rate_limit_timeout = rate_limit_timeout
      
    
    def queryPerspective(self, text: str):
        """ Sends a request to Perspective API for classification.
        """
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = self.perspective_client.comments().analyze(body=analyze_request).execute()
        perspective_result = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
        return round(perspective_result,3)
    
    
    def queryTextBlobPolairty(self, text: str):
        """ Returns the polarity of the text using TextBlob.
        """
        return TextBlob(text).sentiment.polarity

    
    def queryTextBlobObjectivity(self, text: str):
        """ Returns the objectivity of the text using TextBlob.
        """
        return TextBlob(text).sentiment.subjectivity
    
    
    def queryVaderSentiment(self, text: str):
        """ Returns the sentiment score given by VaderSentiment.
        """
        res = self.vader_analyzer.polarity_scores(text)
        return res['compound']
    
    
    def queryAllModelsSingle(self, text: str):
        """ Querys all of the models.
        """
        result = {
            'perspectiveScore': self.queryPerspective(text),
            'tbPolairty': self.queryTextBlobPolairty(text),
            'tbObjectivity': self.queryTextBlobObjectivity(text),
            'vsScore': self.queryVaderSentiment(text)
        }
        return result
    
    
    def queryAllModelsBulk(self, text_items: list):
        """ Querys all of the models for each item in the sentence.
        """
        results = []
        for sentence in text_items:
            sleep(self.rate_limit_timeout)
            results.append(self.queryAllModelsSingle(sentence))
        return results
    