## Index
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

## What is vectorization? (Word Embeddings or Word vectorization)
In text Analysis the **raw data**, **a sequence of symbols** cannot be fed directly to the ML algorithms themselves _as most of them expect **numerical feature vectors with a fixed size** rather than the raw text documents with variable length_.

**Vectorization** is jargon for a classic approach of _converting input data from its raw format (i.e. text ) into vectors of real numbers_ which is the format that ML models support. 

In Machine Learning, vectorization is a step in _feature extraction_. The idea is to get some distinct features out of the text for the model to train on, **_by converting text to numerical vectors_**.

## Vectorization techniques
### 1. Bag of Words
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
**BoW** models are concerned with _whether a known word occurs in a document and how many times it occurs_ -- _not the order in which it appears_, _nor its context_.
**BoW** is often implemented as a [Python dictionary Datastructure](https://docs.python.org/3/tutorial/datastructures.html#dictionaries). 

Each **key** in the _dictionary_ is _set to a word_, and each **value** is set to _the number of times the word appears_.

BoW plays an important role in natural language processing, information retrieval from documents and document classification.

It involves three operations:

#### _Tokenization_
First, the input text is tokenized. A sentence is represented as a list of its constituent words, and it’s done for all the input sentences.

#### _Vocabulary creation_
Of all the obtained tokenized words, only unique words are selected to create the vocabulary and then sorted by alphabetical order.

#### _Vector creation_
Finally, a sparse matrix is created for the input, out of the frequency of vocabulary words. In this sparse matrix, each row is a sentence vector whose length (the columns of the matrix) is equal to the size of the vocabulary.

## Example #1 
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
*  Python code for 'N-Gram', 'Unigram' and 'Bigram'

### _Data_
```Python
sents = ['This document is the first document.',
    'coronavirus is a highly infectious disease',
   'coronavirus affects older people the most', 
   'older people are at high risk due to this disease']
```
### _The Library_
#### CountVectorizer [_link_](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* Convert a collection of text documents to a matrix of token counts.

```Python
from sklearn.feature_extraction.text import CountVectorizer
```
### *Unigram*
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* By default it is Unigram

```Python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sents) 
vectorizer.get_feature_names_out()
```
```
array(['affects', 'are', 'at', 'coronavirus', 'disease', 'document',
       'due', 'first', 'high', 'highly', 'infectious', 'is', 'most',
       'older', 'people', 'risk', 'the', 'this', 'to'], dtype=object)
```
```Python
print(X.toarray())
```
```
[[0 0 0 0 0 2 0 1 0 0 0 1 0 0 0 0 1 1 0]
 [0 0 0 1 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0]
 [1 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0]
 [0 1 1 0 1 0 1 0 1 0 0 0 0 1 1 1 0 1 1]]
 ```
### *N-Gram,Unigram, Bigram*
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
#### ***`ngram_range : tuple (min_n, max_n), default=(1, 1)`***
* The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such
    such that min_n <= n <= max_n will be used. 
    
* For example an ***`ngram_range=(1, 1)`*** means only unigrams, ***`ngram_range=(1, 2)`*** means unigrams and bigrams and ***`ngram_range=(2, 2)`*** means only bigrams. Only applies if ***`analyzer`*** is not callable.

#### ***`analyzer : {'word', 'char', 'char_wb'} or callable, default='word'`***
* Whether the feature should be made of word `n-gram` or character n-grams. Option 'char_wb' creates character n-grams only from text inside word boundaries; n-grams at the edges of words are padded with space.
* If a `callable` is passed it is used to extract the sequence of features out of the raw, unprocessed input.

```Python
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 2))
X2 = vectorizer2.fit_transform(sents)
vectorizer2.get_feature_names_out()
```
```
array(['affects', 'affects older', 'are', 'are at', 'at', 'at high',
       'coronavirus', 'coronavirus affects', 'coronavirus is', 'disease',
       'document', 'document is', 'due', 'due to', 'first',
       'first document', 'high', 'high risk', 'highly',
       'highly infectious', 'infectious', 'infectious disease', 'is',
       'is highly', 'is the', 'most', 'older', 'older people', 'people',
       'people are', 'people the', 'risk', 'risk due', 'the', 'the first',
       'the most', 'this', 'this disease', 'this document', 'to',
       'to this'], dtype=object)
```
```Python
print(X2.toarray())
```
```
[[0 0 0 0 0 0 0 0 0 0 2 1 0 0 1 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0
  1 0 1 0 0]
 [0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0]
 [1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 1 0 1
  0 0 0 0 0]
 [0 0 1 1 1 1 0 0 0 1 0 0 1 1 0 0 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 1 1 0 0 0
  1 1 0 1 1]]
```
```Python
sorted(vectorizer2.vocabulary_.keys())
```
```
['affects', 'affects older', 'are', 'are at', 'at', 'at high', 'coronavirus', 'coronavirus affects', 'coronavirus is',
'disease', 'document', 'document is', 'due', 'due to', 'first', 'first document', 'high', 'high risk', 'highly', 
'highly infectious', 'infectious', 'infectious disease', 'is', 'is highly', 'is the', 'most', 'older', 'older people', 
'people', 'people are', 'people the', 'risk', 'risk due', 'the', 'the first', 'the most', 'this', 'this disease',
'this document', 'to', 'to this']
```

## References
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* [Neptune.ai](https://neptune.ai/blog/vectorization-techniques-in-nlp-guide) 
* [SkLearn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
