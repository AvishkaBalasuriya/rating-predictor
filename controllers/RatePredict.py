from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import pandas as pd
import swifter
import pickle
import spacy
import nltk
import re

class RatePredict:
    def __init__(self, model_path:str):
        self._model = None
        self._model_path = model_path
        self._stemmer = SnowballStemmer("english")
        self._stop_words = set(stopwords.words('english'))
        self._stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
        self._re_stop_words = re.compile(r"\b(" + "|".join(self._stop_words) + ")\\W", re.I)
        
        nltk.download('stopwords')
        
    def _create_df(self, texts:list[dict[str:str]]) -> pd.DataFrame:
        return pd.DataFrame(texts)
            
    def _apply_preprocessing(self, input_df:pd.DataFrame) -> pd.DataFrame:
        input_df['Cleaned'] = input_df['text'].str.lower()
        input_df['Cleaned'] = input_df['Cleaned'].swifter.apply(self._cleanHtml)
        input_df['Cleaned'] = input_df['Cleaned'].swifter.apply(self._cleanPunc)
        input_df['Cleaned'] = input_df['Cleaned'].swifter.apply(self._keepAlpha)
        input_df['Cleaned'] = input_df['Cleaned'].swifter.apply(self._removeStopWords)
        input_df['Cleaned'] = input_df['Cleaned'].swifter.apply(self._stemming)
        
        return input_df
    
    def _load_model(self) -> object:
        with open(self._model_path, 'rb') as file:
            self._model = pickle.load(file)

    def _stemming(self, sentence:str) -> str:
        stemSentence = ""
        for word in sentence.split():
            stem = self._stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    def _removeStopWords(self, sentence:str) -> str:
        return self._re_stop_words.sub(" ", sentence)

    def _cleanHtml(self, sentence:str) -> str:
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(sentence))
        return cleantext

    def _cleanPunc(self, sentence:str) -> str:
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n"," ")
        return cleaned

    def _keepAlpha(self, sentence:str) -> str:
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent
    
    def _format_predictions(self, input_texts:list[dict[str:str]], predictions:list[int]):
        for index in range(len(predictions)):
            input_texts[index]['output'] = str(predictions[index])
        
        return input_texts
    
    def predict(self, input_texts:list[dict[str:str]]) -> list[int]:
        self._load_model()
        df = self._create_df(input_texts)
        df = self._apply_preprocessing(df)
        predictions = self._model.predict(df['Cleaned'].to_list())
        return self._format_predictions(input_texts, predictions)