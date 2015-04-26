# *.* coding: utf-8 *.*
# Karl, by Louis Chartrand 2015

# Requires: numpy, scipy, pattern

import numpy as np
import collections as coll
import scipy.sparse as sp
from itertools import chain, ifilter, ifilterfalse
from scipy.stats import norm

from pattern import vector

import re
re.UNICODE = True

# All methods are batch methods

nonLettre = re.compile(u"[0-9:;,.’()[\]*&?%$#@!~|\\\/=+_¬}{¬¤¢£°±\n\r‘’“”«—·–»…¡¿̀`~^><'\"\xa0]+")
espaces = re.compile(u"[\s'-]+")
def charfilter(text, re_pattern = nonLettre, spaces = espaces):
    """
    Filter that removes punctuation and numbers, and replaces space-like
    characters with spaces.
    """

    if type(text) not in [str, unicode]: return text

    r = re_pattern.sub(u" ", text)
    r = r.lower()
    return spaces.sub(u" ", r)

# Segmenters
##############
#
# They split roughly unstructured text into slightly more manageable chunks

# Preset choice values:

class Segmenter:
    """
    Segmenters split text into segments.
    """
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
    Splits text into paragraph. The optional separator argument is a regular
    expression – by default, 2 or more linefeeds.

    Parameters
    ----------

    separator: string, optional, default: "\\n{2,}|"
        Regex string used to spot and segment paragraphs.

    word_separator: sre.SRE_Pattern, optional, default: karl.espaces
        Pattern used to spot spaces and separated words.

    charfilter_function: function, optional, default: karl.charfilter
        Filters unneeded characters, e.g. punctuation.

    segmentre: sre.SRE_Pattern, optional, default: None
        Only keep segments that matches this pattern.
    """

    sep = "\n{2,}"

    def __init__(self, separator = "\n{2,}",
                    word_separator = espaces,
                    charfilter_function = charfilter,
                    segmentre = None):
        self.priority = 0
        self.sep = separator
        self.segmentre = segmentre

        Segmenter.__init__(self, word_separator, charfilter_function)

    def parse(self, text):

        # Standardize returns
        t = text.replace("\n\r", "\n")
        t = t.replace("\r", "\n")

        # Split
        r = re.split(self.sep, text)

        # Apply character filter
        r = map(self.charfilter_func, r)

        # Apply segment filter
        if self.segmentre != None:
            r = filter(self.segmentre.match, r)

        return [ self.wordsep.split(i) for i in r]

class SentenceSegmenter(Segmenter):
    """
    Splits text into paragraph. The optional separator argument is a regular
    expression – by default, 2 or more linefeeds.

    Parameters
    ----------

    word_separator: sre.SRE_Pattern, optional, default: karl.espaces
        Pattern used to spot spaces and separated words.

    charfilter_function: function, optional, default: karl.charfilter
        Filters unneeded characters, e.g. punctuation.

    segmentre: sre.SRE_Pattern, optional, default: None
        Only keep segments that matches this pattern.
    """
    badendings = re.compile("(?<=[.?!])(?=\S)")

    def __init__(self,
                word_separator = espaces,
                charfilter_function = charfilter,
                segmentre = None):
        self.priority = 0
        self.segmentre = segmentre

        Segmenter.__init__(self, word_separator, charfilter_function)

        from nltk.tokenize.punkt import PunktSentenceTokenizer
        self.tokenizer = PunktSentenceTokenizer()

    def parse(self, text):
        t = self.badendings.sub(" ",text)
        r = self.tokenizer.tokenize(t)

        # Apply character filter
        r = map(self.charfilter_func, r)

        # Apply segment filter
        if self.segmentre != one:
            r = filter(self.segmentre.match, r)

        return r

class ConcordanceSegmenter(Segmenter):
    """
    Makes a segmentation from a word-based concordance.

    Parameters
    ----------

    word: string, mandatory
        Word to be used for concordance.

    nleft: int, optional, default: 50
    nright: int, optional, default: 50
        How many words to retrieve on the right and on the left.

    word_separator: sre.SRE_Pattern, optional, default: karl.espaces
        Pattern used to spot spaces and separated words.

    charfilter_function: function, optional, default: karl.charfilter
        Filters unneeded characters, e.g. punctuation.
    """

    def __init__(self,
                    word,
                    nleft = 50,
                    nright = 50,
                    word_separator = espaces,
                    charfilter_function = charfilter):
        self.priority = 2

        self.word = word
        self.nleft = nleft
        self.nright = nright

        Segmenter.__init__(self, word_separator, charfilter_function)

    def parse(self, text):
        t = self.charfilter_func(text)
        wl = np.array(self.wordsep.split(t))
        posls = np.arange(len(wl))[wl == self.word]

        return [ wl[max(i-self.nleft, 0):i+self.nright] for i in posls ]

class WordWindowSegmenter(Segmenter):
    """
    Segments a text based on word windows.

    Parameters
    ----------

    window: int, optional, default: 100
        Number of words in the word window to retrieve.

    word_separator: sre.SRE_Pattern, optional, default: karl.espaces
        Pattern used to spot spaces and separated words.

    charfilter_function: function, optional, default: karl.charfilter
        Filters unneeded characters, e.g. punctuation.

    segmentre: sre.SRE_Pattern, optional, default: None
        Only keep segments which contains one re more word that matches this
        pattern.
    """

    def __init__(self,
                window = 100,
                word_separator = espaces,
                charfilter_function = charfilter,
                segmentre = None):
        self.priority = 2

        Segmenter.__init__(self, word_separator, charfilter_function)

        self.window = window
        self.segmentre = segmentre

    def parse(self, text):
        t = self.charfilter_func(text)
        wl = self.wordsep.split(t)

        # Make word windows
        r = [ wl[x:x+self.window] for x in xrange(0, len(wl), self.window) ]

        # Apply segment filter
        if self.segmentre != None:
            r = [ i for i in r if any(map(self.segmentre.match,i)) ]

        return r

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
        s = map(s.index, s)
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

    Parameters
    ----------

    segmentation_method: karl.Segmenter, optional, default: None
        Object which splits text into workable chunks.

    wordfilter: karl.Stemmer, optional, default: None
        Associate a word out to a word in. Used for stemming, lemmatization,
        etc.

    stoplist: list, optional, default: []
        List of stopwords, to be removed from text to be analysed.

    lower_freq_bound: float, optional, default: 0.0
    upper_freq_bound: float, optional, default: 1.0
        When the vocabulary is too large, can be used to set arbitrary bounds,
        corresponding to the proportion of segments in which they appear.
    """

    segmentation_method = None
    word_window = 100

    charfilter_method = None

    stoplist = []

    lower_freq_bound = 0.0
    upper_freq_bound = 1.0


    def __init__(self,
            segmentation_method = None,
            wordfilter = None,

            stoplist = [],

            lower_freq_bound = 0.0,
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

    def parse(self, text):
        '''Guesses best parsing function based on data type'''
        if isinstance(text, str) or isinstance(text, unicode):
            return self.parse_unstructured_text(text)
        elif isinstance(text, coll.Iterable):
            return self.parse_segmented_text(text)

    def parse_unstructured_text(self, text):
        '''
        Unstructured, unsegmented text comes in. Segmented, digitized,
        matricized text comes out.
        '''

        segs = np.array(self.segmentation_method.parse(text))

        return self.parse_segmented_text(segs)

    def parse_segmented_text(self, txtiter):
        """
        Parses text that has already been segmented into a matrix.
        """
        mat = Matrix()
        segs = txtiter

        # Build unif & domif lists
        unifs = set(chain(*segs))

        # Remove the small things
        unifs = [ i for i in unifs if len(i) > 1 or (len(i) == 1 and i.isalpha()) ]

        # Remove stopwords
        unifs = list(ifilterfalse(self.stoplist.__contains__, unifs))

        # Save unifs and domifs
        mat.unifs = np.array(unifs)
        mat.domifs = np.arange(len(segs))

        #Apply word filter (e.g. stemmer/lemmatizer), digitizes

        if self.wordfilter != None:
            self.wordfilter.build(unifs)

        segments = []
        for seg in segs:
            # Filter words -- they ought to be contained in predetermined list
            s = filter(unifs.__contains__, seg)

            # Digitize
            s = map(unifs.index, s)

            # Applies stemming/lemmarization
            if self.wordfilter != None:
                s = self.wordfilter.parse(s)

            segments.append(s)

        segs = segments

        # Apply boundaries to cut out to frequent or two infrequent
        if self.lower_freq_bound != 0.0 or self.upper_freq_bound != 1.0:
            c = coll.Counter(chain(*segs))
            freqmap = np.arange()

        # Build data list for matrixes
        col = coll.deque()
        row = coll.deque()
        data = coll.deque()

        segnum = 0
        for seg in segs:
            c = coll.Counter(seg)
            row.extend([segnum] * len(c))
            col.extend(c.keys())
            data.extend(c.values())

            segnum += 1

        # Save Matrix
        mat.segments = np.array(segs)
        mat.coo_matrix = sp.coo_matrix((data, (row, col)), shape = (len(segs), len(mat.unifs)))
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

    def _str2colindex(self,s):
        if type(s) not in [str, unicode]:
            return s

        return [ val(i) if i.isdigit() else self.unifs.index(i) for i in espaces.split(s) ]

    def _str2rowindex(self,s):
        if type(s) not in [str, unicode]:
            return s

        return [ val(i) if i.isdigit() else self.domifs.index(i) for i in espaces.split(s) ]

    def __getitem__(self, index):
        return Matrix(
                    csr_matrix = self.csr_matrix[index],
                    segments = self.segments[index],
                    unifs = self.unifs,
                    domifs = self.domifs[index]
                    )

    def __setitem__(self, index, value):
        self.csr_matrix[index] = value

    def __delitem__(self, index):
        invindex = np.ones(len(index), dtype = bool)
        self.csr_matrix = self.csr_matrix[index]
        self.domifs = self.domifs[index]

    def get_column(self, index):
        i = self._str2colindex(index)
        return Matrix(
            csr_matrix = self.csr_matrix[:,i],
            segments = self.segments,
            unifs = self.unifs[i],
            domifs = self.domifs
            )

    def set_column(self, index, value):
        i = self._str2colindex(index)
        self.csr_matrix[:,i] = t

    def del_column(self, index):
        i = self._str2colindex(index)

        invindex = np.ones(self.csr_matrix.shape[1], dtype=bool)
        invindex[i] = False
        self.csr_matrix = self.csr_matrix[:,invindex]
        self.unifs = self.unifs[invindex]

    def purge_unused_unifs(self):
        idx = self.csr_matrix.sum(0) == 0
        self.del_column(idx)

#
# Association measures
#

def _get_abcd(matrix, index, fterm = False):
    """
    Internal usage. Gets parameters for association measures.
    """

    m = len(matrix.segments)
    invindex = np.ones(m, dtype = bool)
    invindex[index] = False

    # Get the sets
    incluster = np.zeros(matrix.csr_matrix.shape[0], dtype=int)
    incluster[index] = 1
    outofcluster = np.zeros(matrix.csr_matrix.shape[0], dtype=int)
    outofcluster[invindex] = 1

    # Get the matrixes
    matrice = matrix.csr_matrix.todense().clip(0,1)
    invmatrice = abs(matrice-1)

    # Filter out words absent from index
    notempty = np.squeeze(np.asarray(matrice[index].sum(0) != 0))
    matrice = matrice.T[notempty].T
    invmatrice = invmatrice.T[notempty].T
    unifs = matrix.unifs[notempty]

    a = np.squeeze(np.asarray((incluster * matrice).sum(0)))
    b = np.squeeze(np.asarray((incluster * invmatrice).sum(0)))
    c = np.squeeze(np.asarray((outofcluster * matrice).sum(0)))
    d = np.squeeze(np.asarray((outofcluster * invmatrice).sum(0)))

    if fterm: # For chi2P
        return a,b,c,d, unifs, np.squeeze(np.asarray(matrice.sum(0)))
    else:
        return a,b,c,d, unifs

def chi2(matrix, index):
    """
    Calculates χ² association between a segment extension and each words it
    contains.

    Parameters
    ----------

    matrix: karl.Matrix
        The matrix in which the association is to be calculated

    index: list of booleans or list of integers
        Indexes segments which form the extension with which the association is
        to be measured.
    """

    n11, n10, n01, n00, unifs = _get_abcd(matrix, index)

    N = n11 + n10 + n01 + n00
    r= N * ( n11 * n00 - n10 * n01)**2 / np.array( (n11+n00) * (n11+n10) * (n00+n01) * (n00+n10), dtype=float )

    return sorted(zip(r, unifs), reverse=True)

def chi2P(matrix, index):
    """
    Calculates χ²P association between a segment extension and each words it
    contains.

    Parameters
    ----------

    matrix: karl.Matrix
        The matrix in which the association is to be calculated

    index: list of booleans or list of integers
        Indexes segments which form the extension with which the association is
        to be measured.

    From Ogura, Amano & Kondo (2010), "Distinctive characteristics of a metric
    using deviations from Poisson for feature selection"
    """

    a,b,c,d, unifs, fterm = _get_abcd(matrix, index, fterm = True)

    nC = a+b
    nnC = c+d
    lambdai = fterm/matrice.sum()

    ae = nC * (1 - np.exp(-lambdai))
    be = nC * np.exp(-lambdai)
    ce = nnC * (1 - np.exp(-lambdai))
    de = nnC * np.exp(-lambdai)

    chi2P = ((a-ae)**2 / ae) + ((b-be)**2 / be) + ((c-ce)**2 / ce) + ((d-de)**2 / de)
    return sorted(zip(chi2P, unifs), reverse=True)

def gini(matrix, index):
    """
    Calculates Gini association between a segment extension and each words it
    contains.

    Parameters
    ----------

    matrix: karl.Matrix
        The matrix in which the association is to be calculated

    index: list of booleans or list of integers
        Indexes segments which form the extension with which the association is
        to be measured.

    From Ogura, Amano & Kondo (2010), "Distinctive characteristics of a metric
    using deviations from Poisson for feature selection"
    """

    a,b,c,d, unifs = _get_abcd(matrix, index)

    gini = ( 1 / ((a+c)**2) ) * ( ((a**2 / (a+b))**2) + ((c**2 / (c+d))**2) )
    return sorted(zip(gini, unifs), reverse=True)

def information_gain(matrix, index):
    """
    Calculates information gain between a segment extension and each words it
    contains.

    Parameters
    ----------

    matrix: karl.Matrix
        The matrix in which the association is to be calculated

    index: list of booleans or list of integers
        Indexes segments which form the extension with which the association is
        to be measured.

    From Ogura, Amano & Kondo (2010), "Distinctive characteristics of a metric
    using deviations from Poisson for feature selection"
    """

    a,b,c,d, unifs = _get_abcd(matrix, index)

    N = a+b+c+d

    IG = (a/N) * np.log( (a/N) / ( ((a+c)/N) * ((a+b)/N) ) ) + \
         (b/N) * np.log( (b/N) / ( ((b+d)/N) * ((a+b)/N) ) ) + \
         (c/N) * np.log( (c/N) / ( ((a+c)/N) * ((c+d)/N) ) ) + \
         (d/N) * np.log( (d/N) / ( ((b+d)/N) * ((c+d)/N) ) )

    return sorted(zip(IG, unifs), reverse=True)

def binormal_separation(matrix, index):
    """
    Calculates the binormal separation between a segment extension and each words
    it contains.

    Parameters
    ----------

    matrix: karl.Matrix
        The matrix in which the association is to be calculated

    index: list of booleans or list of integers
        Indexes segments which form the extension with which the association is
        to be measured.

    """

    a,b,c,d, unifs = _get_abcd(matrix, index)

    tpr = a / np.array(a + c, dtype=float)
    fpr = b / np.array(b + d, dtype=float)

    BNS = abs(norm.ppf(tpr)-norm.ppf(fpr))

    return sorted(zip(BNS, unifs), reverse=True)

def _get_pairwise_params(matrix, index):
    """
    Internal usage. Gets parameters for association measures that work with
    pairwise comparison (usually for associating two words).
    """

    # Index is set; we need an index set for the "outgroup"
    m = len(matrix.segments)
    invindex = np.ones(m, dtype = bool)
    invindex[index] = False

    # Get the sets
    incluster = np.zeros(matrix.csr_matrix.shape[0], dtype=int)
    incluster[index] = 1
    outofcluster = np.zeros(matrix.csr_matrix.shape[0], dtype=int)
    outofcluster[invindex] = 1

    # Get the matrixes
    matrice = matrix.csr_matrix.todense().clip(0,1)
    invmatrice = abs(matrice-1)

    # Filter out words absent from index
    notempty = np.squeeze(np.asarray(matrice[index].sum(0) != 0))
    matrice = matrice.T[notempty].T
    invmatrice = invmatrice.T[notempty].T
    unifs = matrix.unifs[notempty]

    # Calculate parameters
    N = matrice.sum()
    fXC = np.squeeze(np.asarray((incluster * matrice).sum(0)))
    fX = np.squeeze(np.asarray(matrice.sum(0)))
    fC = len(index) - invindex.sum()

    return fX, fC, fXC, N, unifs

def tScore(matrix, index):
    """
    Calculates the t-score between a segment extension and each words it
    contains.

    Parameters
    ----------

    matrix: karl.Matrix
        The matrix in which the association is to be calculated

    index: list of booleans or list of integers
        Indexes segments which form the extension with which the association is
        to be measured.

    """

    fX, fC, fXC, N, unifs = _get_pairwise_params(matrix, index)

    tScore = ( fXC - ( fX * fC / float(N) ) ) / np.sqrt(fXC)

    return sorted(zip(tScore, unifs), reverse=True)

def mutual_information(matrix, index):
    """
    Calculates the mutual information between a segment extension and each words
    it contains.

    Parameters
    ----------

    matrix: karl.Matrix
        The matrix in which the association is to be calculated

    index: list of booleans or list of integers
        Indexes segments which form the extension with which the association is
        to be measured.

    """

    fX, fC, fXC, N, unifs = _get_pairwise_params(matrix, index)

    MI = np.log( fXC * N / (fX * fC))

    return sorted(zip(MI, matrix.unifs), reverse=True)
