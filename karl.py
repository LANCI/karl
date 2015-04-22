# *.* coding: utf-8 *.*
# Karl, by Louis Chartrand 2015

# Requires: numpy, scipy, nltk

import numpy as np
import collections as coll
import scipy.sparse as sp
from itertools import chain
from pattern import vector

import re
re.UNICODE = True

# All methods are batch methods

nonLettre = re.compile(u"[0-9:;,.’()[\]*&?%$#@!~|\\\/=+_¬}{¬¤¢£°±\n\r‘’“”«—·–»…¡¿̀`~^><'\"\xa0]+")
espaces = re.compile(u"[\s'-]+")
def charfilter(text, re_pattern = nonLettre, spaces = espaces):
    """Filter that removes punctuation and numbers, and replaces space-like"""
    """characters with spaces."""

    if type(text) not in [str, unicode]: return text

    r = re_pattern.sub(u" ", text)
    r = r.lower()
    return spaces.sub(u" ", r)

# Segmenters
##############
#
# They split roughly unstructured text into slightly more manageable chunks

# Preset choice values:
SMParagraph = 1
SMSentence = 2
SMConcordance = 3
SMWordWindow = 4

class Segmenter:
    """Segmenters split text into segments."""
    method = 0
    # Can be:
    #   1 - Paragraph
    #   2 - Sentence
    #   3 - Concordance (pre-treatment or post-treatment)
    #   4 - Word windows (pre-treatment or post-treatment)

    priority = 0
    #   0 - Segmentation ought to be done before all treatment
    #   1 - Segmentation ought to be done before word split
    #   2 - Segmentation ought to be done post word split

    wordsep = espaces
    charfilter_func = None

    def __init__(self, word_separator = espaces, charfilter_function = charfilter):
        self.wordsep = word_separator
        self.charfilter_func = charfilter_function


class ParagraphSegmenter(Segmenter):
    """
    Splits text into paragraph. The optional separator argument is a regular expression – by default, 2 or more linefeeds.
    """

    sep = "(\n{2,}|\r{2,}|(\n\r){2,})"

    def __init__(self, separator = "(\n|\r|\n\r){2,}",
                    word_separator = espaces,
                    charfilter_function = charfilter):
        self.method = SMParagraph
        self.priority = 0
        self.sep = separator

        Segmenter.__init__(self, word_separator, charfilter_function)

    def parse(self, text):
        r = re.split(self.sep, text)
        r = map(self.charfilter_func, r)
        return [ self.wordsep.split(i) for i in r]

class SentenceSegmenter(Segmenter):
    """
    Splits text into paragraph. The optional separator argument is a regular expression – by default, 2 or more linefeeds.
    """
    badendings = re.compile("(?<=[.?!])(?=\S)")

    def __init__(self):
        self.method = SMSentence
        self.priority = 0

        from nltk.tokenize.punkt import PunktSentenceTokenizer
        self.tokenizer = PunktSentenceTokenizer()

    def parse(self, text):
        t = self.badendings.sub(" ",text)
        return self.tokenizer.tokenize(t)

class ConcordanceSegmenter(Segmenter):
    """
    Makes a segmentation from a word-based concordance.
    """

    def __init__(self, word, nleft = 50, nright = 50):
        self.method = SMConcordance
        self.priority = 2

        self.word = word
        self.nleft = nleft
        self.nright = nright

    def parse(self, text):
        t = self.charfilter_func(text)
        wl = self.wordsep.split(t)
        posls = np.arange(len(wl))[wl == word]

        return [ wl[max(i-nleft):i+nright] for i in posls ]

class WordWindowSegmenter(Segmenter):

    def __init__(self):
        self.method = SMWordWindow
        self.priority = 2

    def parse(self, text):
        t = self.charfilter_func(text)
        wl = self.wordsep.split(t)
        return [ wl[x:x+window] for x in xrange(len(wl), step = window) ]

class Stemmer:
    """A Porter stemmer, exploits function from Pattern (Clips)"""
    lang = 'fr'

    def __init__(self, lang = 'fr'):
        self.lang = lang
        vector.language = lang

    def build(self, words):
        """
        Builds an index. Words must be properly sorted so that their index
        reflects the one that they'll have after digitization.
        """

        s = map(vector.stem, words)
        self.map = np.array(s)

    def parse(self, wordlist):
        return self.map[wordlist]

class Lemmatizer(Stemmer):
    """Lemmatizer, exploits function from Pattern (Clips), based on Lefff."""
    def __init__(self, lang = 'fr'):
        self.lang = lang
        vector.language = lang
        vector.stem.func_defaults = ("lemma",)

class TextParser:
    """
    Brings together all that is necessary to go from unstructured text to matrix
    object.
    """

    segmentation_method = None
    word_window = 100

    charfilter_method = None

    stoplist = []

    lower_freq_bound = 0.0
    upper_freq_bound = 1.0


    def __init__(self,
            charfilter_function = charfilter,   # Filters characters we do not need, like some punctuation.
            segmentation_method = None,         # Objects which inherit Segmenter. Splits text into workable chunks.
            wordfilter = None,           # Word in → word out. Perfect for a stemmer.

            stoplist = [],                      # Stoplist

            lower_freq_bound = 0.0,             # These parameters are for restraining number of unifs.
            upper_freq_bound = 1.0
            ):

        if segmentation_method == None:
            self.segmentation_method = ParagraphSegmenter()
        else:
            self.segmentation_method = segmentation_method

        self.wordfilter = wordfilter

        self.stoplist = stoplist
        self.lower_freq_bound = lower_freq_bound
        self.upper_freq_bound = upper_freq_bound
        self.charfilter_function = charfilter_function

    def parse(self, text):
        '''Guesses best parsing function based on data type'''
        if isinstance(text, str) or isinstance(text, unicode):
            return self.parse_unstructured_text(text)
        elif isinstance(text, coll.Iterable):
            return self.parse_segmented_text(text)

    def parse_unstructured_text(self, text):
        '''Unstructured, unsegmented text comes in. Segmented, digitized,'''
        '''matricized text comes out.'''
        if self.segmentation_method.priority == 0:
            segs = np.array(self.segmentation_method.parse(text))
            segs = np.array(map(self.charfilter_function,segs))
        elif self.segmentation_method.priority == 1:
            segs = np.array(self.segmentation_method.parse(self.charfilter_function(text)))

        return self.parse_segmented_text(segs)

    def parse_segmented_text(self, txtiter):
        """
        Parses text that has already been segmented into a matrix.
        """
        mat = Matrix()
        segs = txtiter

        # Build unif & domif lists
        unifs = np.unique(chain(*segs))

        # Remove the small things
        unifs = [ i for i in unifs if len(i) > 1 or (len(i) == 1 and i.isalpha()) ]

        # Save unifs and domifs
        mat.unifs = np.array(unifs)
        mat.domifs = np.arange(len(segs))

        #Apply word filter (e.g. stemmer/lemmatizer), digitizes

        if self.wordfilter != None:
            self.wordfilter.build(unifs)

        print len(self.wordfilter.map)

        segments = []
        for seg in segs:
            s = filter(unifs.__contains__, seg)
            s = map(unifs.index, s)
            if self.wordfilter != None:
                s = self.wordfilter.parse(s)
            segments.append(s)

        segs = segments

        # if self.wordfilter_method == None:
        #     segs = np.array([ map(unifs.index, i) for i in segs ])
        # else:
        #     self.wordfilter_method.build(mat.unifs)
        #     segs = np.array([ self.wordfilter_method.parse(map(unifs.index, i)) for i in segs ])

        # Apply boundaries to cut out to frequent or two infrequent
        if self.lower_freq_bound != 0.0 or self.upper_freq_bound != 1.0:
            c = coll.Counter(chain(*segs))
            freqmap = np.arange()

        print unifs[:30]
        print segs[0][:30]

        # Build data list for matrixes
        col = coll.deque()
        row = coll.deque()
        data = coll.deque()

        segnum = 0
        for seg in segs:
            c = coll.Counter(seg)
            row.extend([segnum] * len(seg))
            col.extend(c.keys())
            data.extend(c.values())

        # Save Matrix
        mat.segments = segs
        mat.coo_matrix = sp.coo_matrix((data, (row, col)), shape = (len(row), len(mat.unifs)))
        mat.csr_matrix = mat.coo_matrix.tocsr()

        return mat

class Matrix:
    """
    Object based on scipy's sparse matrix, holds word-space model data.
    """
    csr_matrix = None
    segments = np.array([[]])
    unifs = np.array([])
    domifs = np.array([])

    def __init__(self, csr_matrix = None, segments = None, unifs = None, domifs = None):
        if csr_matrix != None:
            self.csr_matrix = csr_matrix

        if segments != None:
            self.segments = np.array(segments)
        if unifs != None:
            self.unifs = np.array(unifs)
        if domifs  != None:
            self.domifs = np.array(domifs)

    def __repr__(self):
        return self.csr_matrix.__repr__()

    def _str2colindex(s):
        if type(s) not in [str, unicode]:
            return s

        return [ val(i) if i.isdigit() else self.unifs.index(i) for i in espaces.split(s) ]

    def _str2rowindex(s):
        if type(s) not in [str, unicode]:
            return s

        return [ val(i) if i.isdigit() else self.domifs.index(i) for i in espaces.split(s) ]

    def __getitem__(self, index):
        return Matrix(None, self.csr_matrix[index], segments[index], self.unifs, domifs[index])

    def __setitem__(self, index, value):
        self.csr_matrix[index] = value

    def __delitem__(self, index):
        del self.csr_matrix[index]
        del self.domifs[index]

    def get_column(self, index):
        i = self._str2colindex(index)
        return Matrix(
            csr_matrix = self.csr_matrix.transpose()[i],
            segments = self.segments,
            unifs = self.unifs[i],
            domifs = self.domifs
            )

    def set_column(self, index, value):
        i = self._str2colindex(index)
        t = self.csr_matrix.transpose()
        t[i] = value
        self.csr_matrix = t

    def del_column(self, index):
        i = self._str2colindex(index)
        t = self.csr_matrix.transpose()
        del t[i]
        self.csr_matrix = t

        del self.unifs[i]
