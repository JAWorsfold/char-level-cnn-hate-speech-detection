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
    _re_punctuation = '[^\w\s]|_'
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


    def remove_noise(self, text, mentions=False, urls=True, html_entities=True, replacement=''):
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
        :return: Pre-processed text with noise removed
        """
        pp_text = text
        if mentions:      pp_text = re.sub(self._re_mentions, replacement, pp_text)
        if urls:          pp_text = re.sub(self._re_urls, replacement, pp_text)
        if html_entities: pp_text = re.sub(self._re_html_entities, replacement, pp_text)
        return pp_text


    def normalise(self, text, lowercase=True, punctuation=True, numbers=False, whitespace=True,
                  replacement='', stopwords=True, other_stopwords=list(), stem_words=False):
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
        :param stem_words: Stem words each word in the text
        :type stem_words: boolean
        :return: Normalised text
        """
        nl_text = text
        if lowercase:   nl_text = self._to_lowercase(nl_text)
        if punctuation: nl_text = re.sub(self._re_punctuation, replacement, nl_text)
        if numbers:     nl_text = re.sub(self._re_numbers, replacement, nl_text)
        if stopwords:   nl_text = self._remove_stopwords(nl_text, other_stopwords)
        if stem_words:  nl_text = self._stem_words(nl_text)
        ws = self._re_whitespace
        if whitespace:  nl_text = re.sub(ws, replacement, nl_text)
        # remove any leading or trailing whitespace
        lead_trail_ws = f"^{ws}|{ws}$"
        nl_text = re.sub(lead_trail_ws, '', nl_text)
        return nl_text


    @staticmethod
    def tokenize(text):
        """Split text into smaller pieces, or tokens"""
        tokenized_text = text
        return tokenized_text


    @staticmethod
    def _to_lowercase(text):
        """Convert all characters in text to lowercase"""
        return text.lower()


    @staticmethod
    def _remove_stopwords(text, add_words):
        """Remove stop words from text"""
        # ToDo: allow user to pass a different corpus of stop words as an argument
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.extend(add_words)
        stopped_text = ' '.join([w for w in text.split(' ') if w not in stop_words])
        return stopped_text


    @staticmethod
    def _stem_words(text):
        """Stem words in text"""
        # ToDo: allow user to pass a different stemmer as an argument
        stemmer = nltk.PorterStemmer()
        stemmed_text = ' '.join([stemmer.stem(w) for w in text.split()])
        return stemmed_text
