## Index
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* [What is vectorization](#what-is-vectorization-word-embeddings-or-word-vectorization)
#### [_Vectorization techniques_](#vectorization-techniques)
1. [Bag of Words](#1-bag-of-words)
    * [Example #1](#example-1)
    * [Example #2 Python code](https://nbviewer.org/github/iAmKankan/NaturalLanguageProcessing-NLP/blob/master/Text%20Preprocessing%20Level%23%202/Bag_of_Words.ipynb)
    
2. [TF-IDF](#2-tf-idf)

<img src="https://latex.codecogs.com/svg.image?\large&space;\begin{align&space;*}\begin{tabular}{|c|c|}\hline&space;{\color{Purple}\textbf{Symbols}}&&space;{\color{Purple}\textbf{Typical&space;Meaning}}\\\hline&space;\hline&space;{\color{Purple}\textit{$\varnothing$&space;}}&space;&space;&&space;\textbf{Empty&space;set}\\&space;&space;\hline&space;&space;{\color{Purple}\textit{A$\setminus$B}}&space;&space;&&space;\textbf{A&space;without&space;B:&space;the&space;set&space;of&space;elements&space;in&space;A&space;but&space;not&space;in&space;B}&space;\\&space;&space;\hline&space;&space;{\color{Purple}\textit{D}}&space;&space;&&space;\textbf{Number&space;of&space;dimensions;&space;indexed&space;by&space;d&space;&space;1,...,D}\\&space;&space;\hline&space;&space;{\color{Purple}\textit{N}}&space;&space;&&space;\textbf{Number&space;of&space;data&space;points;&space;indexed&space;by&space;n&space;&space;1,...,N}\\&space;&space;\hline&space;\hline&space;{\color{Purple}\textit{$I_m$}}&space;&space;&&space;\textbf{Identity&space;matrix&space;of&space;size&space;m&space;&space;m}\\&space;&space;\hline&space;&space;{\color{Purple}\textit{$0_{m,n}$}}&space;&space;&&space;\textbf{Matrix&space;of&space;zeros&space;of&space;size&space;m&space;&space;n}\\&space;&space;\hline&space;&space;{\color{Purple}\textit{$1_{m,n}$}}&space;&space;&&space;\textbf{Matrix&space;of&space;ones&space;of&space;size&space;m&space;&space;n}\\&space;&space;\hline&space;&space;{\color{Purple}\textit{$e_i$}}&space;&space;&&space;\textbf{Standard/canonical&space;vector&space;(where&space;i&space;is&space;the&space;component&space;that&space;is&space;1)}\\&space;&space;\hline&space;&space;&space;\end{align}\end{align}" title="https://latex.codecogs.com/svg.image?\large \begin{align *}\begin{tabular}{|c|c|}\hline {\color{Purple}\textbf{Symbols}}& {\color{Purple}\textbf{Typical Meaning}}\\\hline \hline {\color{Purple}\textit{$\varnothing$ }} & \textbf{Empty set}\\ \hline {\color{Purple}\textit{A$\setminus$B}} & \textbf{A without B: the set of elements in A but not in B} \\ \hline {\color{Purple}\textit{D}} & \textbf{Number of dimensions; indexed by d 1,...,D}\\ \hline {\color{Purple}\textit{N}} & \textbf{Number of data points; indexed by n 1,...,N}\\ \hline \hline {\color{Purple}\textit{$I_m$}} & \textbf{Identity matrix of size m  m}\\ \hline {\color{Purple}\textit{$0_{m,n}$}} & \textbf{Matrix of zeros of size m  n}\\ \hline {\color{Purple}\textit{$1_{m,n}$}} & \textbf{Matrix of ones of size m  n}\\ \hline {\color{Purple}\textit{$e_i$}} & \textbf{Standard/canonical vector (where i is the component that is 1)}\\ \hline \end{align}\end{align}" />

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
[Python code](https://nbviewer.org/github/iAmKankan/NaturalLanguageProcessing-NLP/blob/master/Text%20Preprocessing%20Level%23%202/Bag_of_Words.ipynb)

### 2. TF-IDF
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
#### Tf–idf term weighting( term-frequency inverse document-frequency)
Some words in enery language, carry very little meaningful information about the actual contents of the document (e.g. “**the**”, “**a**”, “**is**” in English) . If we were to feed the direct count data directly to a classifier _those very frequent terms would shadow the frequencies of rarer yet more interesting terms_.

In order to _re-weight_ the count features into **floating point values** suitable for usage by a classifier it is very common to use the **tf–idf** transform.

<img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{tf-idf=&space;tf(t,d)&space;\times&space;idf(t)}&space;}" title="{\color{Purple}\mathbf{tf-idf= tf(t,d) \times idf(t)} }" />

Using the `TfidfTransformer`’s default settings, `TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)` the term frequency, the number of times a term occurs in a given document, is multiplied with idf component, which is computed as

<img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{idf(t)=\log&space;\frac{1&plus;n}{1&plus;df(t)}&plus;1}&space;}" title="{\color{Purple}\mathbf{idf(t)=\log \frac{1+n}{1+df(t)}+1} }" />

where **_n_** is the total number of documents in the document set and **_df(t)_** is the number of documents in the document set that contain term **_t_**. The resulting **_tf-idf_** vectors are then **normalized** by the _**Euclidean norm**_:

<img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{v_{norm}=\frac{v}{\parallel&space;v\parallel_2}=\frac{v}{\sqrt{v_1^2&plus;v_2^2&plus;...&plus;v_n^2}}}}" title="{\color{Purple}\mathbf{v_{norm}=\frac{v}{\parallel v\parallel_2}=\frac{v}{\sqrt{v_1^2+v_2^2+...+v_n^2}}}}" />

This was originally a term weighting scheme developed for information retrieval (as a ranking function for search engines results) that has also found good use in document classification and clustering.

The following sections contain further explanations and examples that illustrate how the tf-idfs are computed exactly and how the tf-idfs computed in scikit-learn’s TfidfTransformer and TfidfVectorizer differ slightly from the standard textbook notation that defines the idf as

 <img src="https://latex.codecogs.com/svg.image?{\color{Purple}\mathbf{idf(t)=\log&space;\frac{n}{1&plus;df(t)}}&space;}" title="{\color{Purple}\mathbf{idf(t)=\log \frac{n}{1+df(t)}} }" />
 


## References
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* [Neptune.ai](https://neptune.ai/blog/vectorization-techniques-in-nlp-guide) 
* [Bag of Words SkLearn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
* [TF-IDF sklearn](https://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation)
