"""

"""

import

class PreprocessUtils:

    # default regular expressions used for text transformations
    _re_urls = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    _re_mentions = '@[\w\-]+'
    _re_whitespace = '\s+'
    _re_html_entities = '&[^\s]*;'
    _re_punctuation = '[^\w\s]\_'
    _re_numbers = '\d'


    def __init__(self, re_urls=None, re_mentions=None, re_html_entities=None,
                 re_whitespace=None, re_punctuation=None, re_numbers=None):
        """
        Initialise PreprocessUtils Class with default or specified regular expressions (regex).
        :param re_urls: regex for URLs
        :param re_mentions: regex for mentions in Twitter
        :param re_html_entities: regex for HTML entities
        :param re_whitespace: regex for whitespace
        :param re_punctuation: regex for punctuation
        :param re_numbers: regex for numbers
        """
        if re_urls is not None: _re_urls = re_urls
        if re_mentions is not None: _re_mentions = re_mentions
        if re_whitespace is not None: _re_whitespace = re_whitespace
        if re_html_entities is not None: _re_html_entities = re_html_entities
        if re_punctuation is not None:  _re_punctuation = re_punctuation
        if re_numbers is not None: _re_numbers = re_numbers
        if re_numbers is not None: _re_numbers = re_numbers


    def remove_noise(self, mentions=False, urls=False, html_entities=False):
        """"""
        pass


    def normalise(self):
        """"""
        pass


    def tokenize(self, ):
        """"""
        pass


    def _to_lowercase(self):
        """"""
        pass


    def _replace_chars(self, punctuation=False, numbers=False, whitespace=False ):
        """"""
        pass


    def _remove_stopwords(self):
        """"""
        pass


    def _stem_words(self, asList=?):
        """"""
        pass
