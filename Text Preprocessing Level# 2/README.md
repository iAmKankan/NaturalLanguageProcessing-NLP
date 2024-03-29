## Index
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
### [What is vectorization](#what-is-vectorization-word-embeddings-or-word-vectorization)
### [_Vectorization techniques_](#vectorization-techniques)
1. [Bag of Words](#1-bag-of-words)
    * [Example #1 N-Gram,Unigram, Bigram](#example-1)
    * [Example #2 Python code](https://nbviewer.org/github/iAmKankan/NaturalLanguageProcessing-NLP/blob/master/Text%20Preprocessing%20Level%23%202/Bag_of_Words.ipynb)
    
2. [TF-IDF](#2-tf-idf-term-frequency-inverse-document-frequency)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/151629694-24298066-ce77-4eed-a32c-a7e8114aa18e.png" width=70% />
  <br><ins><i><b>Various Vectorization techniques </i></b></ins>
</p>

## What is vectorization? (Word Embeddings or Word vectorization)
In text Analysis the **raw data**, **a sequence of symbols** cannot be fed directly to the ML algorithms themselves _as most of them expect **numerical feature vectors with a fixed size** rather than the raw text documents with variable length_.

**Vectorization** is jargon for a classic approach of _converting input data from its raw format (i.e. text ) into vectors of real numbers_ which is the format that ML models support. 

In Machine Learning, vectorization is a step in _feature extraction_. The idea is to get some distinct features out of the text for the model to train on, **_by converting text to numerical vectors_**.

## Vectorization techniques
### 1. Bag of Words
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
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

### Example #1 
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
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
### Example #2
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

### [Python code](https://nbviewer.org/github/iAmKankan/NaturalLanguageProcessing-NLP/blob/master/Text%20Preprocessing%20Level%23%202/Bag_of_Words.ipynb)
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)

## 2. TF-IDF (term-frequency inverse document-frequency)
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
While working with frequency of words in a text we always find some words are very frequent but carry little to no significance in text processing. In the other hand words with lesser frequncy might have more meaning to the text. 

**TF-IDF** (Term Frequency - Inverse Document Frequency) is a handy algorithm that uses the _frequency of words_ **to determine how relevant those words are to a given document**.

<img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{tf-idf=&space;tf(t,d)&space;\times&space;idf(t)}&space;}" title="{\color{Purple}\mathbf{tf-idf= tf(t,d) \times idf(t)} }" />

### Definition
* The **tf–idf** is the product of two statistics, **term frequency** and **inverse document frequency**. There are various ways for determining the exact values of both statistics.
* A formula that aims to define the importance of a keyword or phrase within a document or a web page.

### Term frequency
Term frequency, $\large{\color{Purple}\textrm{tf(t,d)} }$ , is the [relative frequency](https://github.com/iAmKankan/Statistics/tree/main/frequency-distribution#relative-frequency-distribution) of term $\large{\color{Purple}\textrm{t} }$ within document $\large{\color{Purple}\textrm{d} }$,

$$\large{\color{Purple} \textrm{tf}(t,d)= \frac{f_{t,d}}{\sum_{t^{\prime} \in d}  f_{t^\prime,d} }}$$

where $\large{\color{Purple}f_{t,d} }$ is the raw count of a term in a document, **i.e.** the number of times that term $\large{\color{Purple}\textrm{t} }$ occurs in document $\large{\color{Purple}\textrm{d} }$. 

**Note** the denominator is simply the total number of terms in document $\large{\color{Purple}\textrm{d} }$ (counting each occurrence of the same term separately). There are various **other ways** to define term frequency.

* the raw count itself: $\large{\color{Purple}\textrm{tf}(t,d) = f_{t,d}}$
* Boolean "frequencies": $\large{\color{Purple}\textrm{tf}(t,d) = 1}$ if $\large{\color{Purple}t}$ occurs in $\large{\color{Purple}d}$ and $\large{\color{Purple}0}$ otherwise;
* logarithmically scaled frequency/log normalization: $\large{\color{Purple}\mathrm{tf}(t,d) = \log (1 + f_{t,d})}$
* augmented frequency, to prevent a bias towards longer documents, e.g. raw frequency divided by the raw frequency of the most frequently occurring term in the document: $\large{\color{Purple}{\displaystyle \mathrm {tf} (t,d)=0.5+0.5\cdot {\frac {f_{t,d}}{\max \\{ f_{t',d}:t'\in d \\} }}}}$

### Inverse document frequency
The inverse document frequency is a measure of how much information the word provides, i.e., if it is common or rare across all documents. It is the logarithmically scaled inverse fraction of the documents that contain the word (obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient):

$$\large{\color{Purple}\mathrm{idf}(t, D) =  \log \frac{N}{| \\{ d \in D: t \in d \\} |}}$$

with
* $\large{\color{Purple}{\displaystyle N}:}$ total number of documents in the corpus $\large{\color{Purple}{\displaystyle N={|D|}}}$
* $\large{\color{Purple}{\displaystyle |\{d\in D:t\in d\}|}}$ : number of documents where the term $\large{\color{Purple}{\displaystyle t}}$ appears (i.e., $\large{\color{Purple}{\displaystyle \mathrm {tf} (t,d)\neq 0} }$ ). If the term is not in the corpus, this will lead to a division-by-zero. It is therefore common to adjust the denominator to $\large{\color{Purple}{\displaystyle 1+|\{d\in D:t\in d\}|}}$.


### Terminologies:
**Term Frequency:** In document $\large{\color{Purple}\textrm{d} }$ , the frequency represents the number of instances of a given word $\large{\color{Purple}\textrm{t} }$. Therefore, we can see that it becomes more relevant when a word appears in the text, which is rational. Since the ordering of terms is not significant, we can use a vector to describe the text in the bag of term models. For each specific term in the paper, there is an entry with the value being the term frequency.


Using the `TfidfTransformer`’s default settings, `TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)` the term frequency, the number of times a term occurs in a given document, is multiplied with idf component, which is computed as

<img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{idf(t)=\log&space;\frac{1&plus;n}{1&plus;df(t)}&plus;1}&space;}" title="{\color{Purple}\mathbf{idf(t)=\log \frac{1+n}{1+df(t)}+1} }" />

where **_n_** is the total number of documents in the document set and **_df(t)_** is the number of documents in the document set that contain term **_t_**. The resulting **_tf-idf_** vectors are then **normalized** by the _**Euclidean norm**_:

<img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{v_{norm}=\frac{v}{\parallel&space;v\parallel_2}=\frac{v}{\sqrt{v_1^2&plus;v_2^2&plus;...&plus;v_n^2}}}}" title="{\color{Purple}\mathbf{v_{norm}=\frac{v}{\parallel v\parallel_2}=\frac{v}{\sqrt{v_1^2+v_2^2+...+v_n^2}}}}" />

This was originally a term weighting scheme developed for information retrieval (as a ranking function for search engines results) that has also found good use in document classification and clustering.

![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)

The following sections contain further explanations and examples that illustrate how the tf-idfs are computed exactly and how the tf-idfs computed in scikit-learn’s TfidfTransformer and TfidfVectorizer differ slightly from the standard textbook notation that defines the idf as

 <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{idf(t)=\log&space;\frac{n}{1&plus;df(t)}}&space;}" title="{\color{Purple}\mathbf{idf(t)=\log \frac{n}{1+df(t)}} }" />
 


## References
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* [Neptune.ai](https://neptune.ai/blog/vectorization-techniques-in-nlp-guide) 
* [Bag of Words SkLearn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* [TF-IDF sklearn](https://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation)
