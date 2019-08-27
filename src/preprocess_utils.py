import re
import nltk

class PreProcessUtils:
    """
    The PreProcessUtils Class is used for common pre-processing methods and
    contains default or specified regular expressions (regex).

    :param re_urls: regex for URLs
    :param re_mentions: regex for mentions in Twitter
    :param re_html_entities: regex for HTML entities
    :param re_whitespace: regex for whitespace
    :param re_punctuation: regex for punctuation
    :param re_numbers: regex for numbers
    """

    _re_urls = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    _re_mentions = '@[\w\-]+'
    _re_whitespace = '\s+'
    _re_html_entities = '&[^\s]*;'
    _re_punctuation = '[^\w\s]\_'
    _re_numbers = '\d'


    def __init__(self, re_urls=None, re_mentions=None, re_html_entities=None,
                 re_whitespace=None, re_punctuation=None, re_numbers=None):
        # ToDo: include condition to check if a string is a regular expressions
        if re_urls is not None: _re_urls = re_urls
        if re_mentions is not None: _re_mentions = re_mentions
        if re_whitespace is not None: _re_whitespace = re_whitespace
        if re_html_entities is not None: _re_html_entities = re_html_entities
        if re_punctuation is not None:  _re_punctuation = re_punctuation
        if re_numbers is not None: _re_numbers = re_numbers
        if re_numbers is not None: _re_numbers = re_numbers


    def remove_noise(self, text, mentions=False, urls=False, html_entities=False, replacement=''):
        """
        Remove noise from the input text based on specified parameters:

        :param text: Input text to process
        :type text: string
        :param mentions: Remove or replace Twitter mentions
        :type mentions: boolean
        :param boolean urls: Remove or replace URLs
        :type urls: boolean
        :param html_entities: Remove or replace HTML entities
        :type html_entities: boolean
        :param replacement: Value to use for replacement
        :type replacement: string
        :return: A processed text with noise removed
        """
        processed_text = text
        return processed_text


    def normalise(self, text, lowercase=True, punctuation=True, numbers=False, whitespace=True,
                  replacement='', stopwords=True, other_stopwords=list()):
        """
        Normalise the input text based on specified parameters:

        :param text: Input text to process
        :type text: string
        :param lowercase: Convert all characters to lowercase
        :type lowercase: boolean
        :param punctuation: Remove or replace punctuation
        :type punctuation: boolean
        :param numbers: Remove or replace numbers
        :type numbers: boolean
        :param whitespace: Remove or replace whitespace
        :type whitespace: boolean
        :param replacement: Value to use for replacement
        :type replacement: string
        :param stopwords: Remove English stop words using nltk corpus
        :type stopwords: boolean
        :param other_stopwords: Additional stop words to remove
        :type other_stopwords: array
        :return: Normalised text
        """
        normalised_text = text
        return normalised_text


    @staticmethod
    def tokenize(text):
        """Split text into smaller pieces, or tokens"""
        tokenized_text = text
        return tokenized_text


    @staticmethod
    def _to_lowercase(text):
        """Convert all characters in text to lowercase"""
        lowercase_text = text
        return lowercase_text


    @staticmethod
    def _remove_stopwords(text, other_stopwords):
        """Remove stop words from text"""
        no_stopwords_text = text
        return no_stopwords_text


    @staticmethod
    def _stem_words(text):
        """Stem words in text"""
        stemmed_text = text
        return stemmed_text
