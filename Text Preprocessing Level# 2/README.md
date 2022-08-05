## Index
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

## What is vectorization? (Word Embeddings or Word vectorization)
**Vectorization** is jargon for a classic approach of _converting input data from its raw format (i.e. text ) into vectors of real numbers_ which is the format that ML models support. 

In Machine Learning, vectorization is a step in _feature extraction_. The idea is to get some distinct features out of the text for the model to train on, **_by converting text to numerical vectors_**.

## Vectorization techniques
### 1. Bag of Words
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
**BoW** models are concerned with _whether a known word occurs in a document and how many times it occurs_ -- _not the order in which it appears_, _nor its context_.
**BoW** is often implemented as a [Python dictionary Datastructure](https://docs.python.org/3/tutorial/datastructures.html#dictionaries). 

Each **key** in the _dictionary_ is _set to a word_, and each **value** is set to _the number of times the word appears_.

BoW plays an important role in natural language processing, information retrieval from documents and document classification.

Most simple of all the techniques out there. It involves three operations:

#### _Tokenization_
First, the input text is tokenized. A sentence is represented as a list of its constituent words, and it’s done for all the input sentences.

#### _Vocabulary creation_
Of all the obtained tokenized words, only unique words are selected to create the vocabulary and then sorted by alphabetical order.

#### _Vector creation_
Finally, a sparse matrix is created for the input, out of the frequency of vocabulary words. In this sparse matrix, each row is a sentence vector whose length (the columns of the matrix) is equal to the size of the vocabulary.

Let’s work with an example and see how it looks in practice. We’ll be using the Sklearn library for this exercise.














![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

### Bag-of-words model
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
> **_Bag of Words (BoW)_** is an algorithm that _counts how many times a word appears in a document_. Those word-counts allow us to compare documents and gauge their similarities for applications like **search**, **document classification** and **topic modeling**. 

**BoW** is a also method for preparing text for input in a **deep-learning net**. BoW lists words paired with their word counts per document. In the table where the words and documents that effectively become vectors are stored, **each row is a word**, **each column is a document**, and **each cell is a word count**. Each of the documents in the corpus is represented by columns of equal length. Those are wordcount vectors, an output stripped of context.

**Before they’re fed to the neural network, each vector of _wordcounts is normalized_ such that all elements of the vector add up to one**.Thus, the frequency of each word is effectively converted to represent the probabilities of those words’ occurrence in the document.  Probabilities that surpass certain levels will activate nodes in the network and influence the document’s classification. We need the way to represent text data for the machine learning algorithms(In vector of number), and the bag-of-words model helps us to achieve the task. It is the way of extracting features from the text for the use in machine learning algorithms.

**In this approach, we use the tokenised words for each of observation and find out the frequency of each token.**

#### Example #1
* Let’s consider following three sentences to understand this concept in depth.
```
“It is going to rain today.”
“Today, I am not going outside.”
“I am going to watch the season premiere.”
```
##### Step #1
* We treat each sentence as a separate document and we make a list of all the words from all three documents, excluding the punctuation.
* And we get- 
**[‘It’, ’is’, ’going’, ‘to’, ‘rain’, ‘today’ ‘I’, ‘am’, ‘not’, ‘outside’, ‘watch’, ‘the’, ‘season’, ‘premiere’]**
##### Step #2
* The next step is to create vectors. Vectors convert text into numbers that can be used by the machine learning algorithm.
* We take the first document — **“It is going to rain today”**, and we check the frequency of words from the ten unique words.
```
“It” = 1
“is” = 1
“going” = 1
“to” = 1
“rain” = 1
“today” = 1
“I” = 0
“am” = 0
“not” = 0
“outside” = 0
```
* Rest of the documents will be:
```
“It is going to rain today” = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] 
“Today I am not going outside” = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1] 
“I am going to watch the season premiere” = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
```
* In this approach, each word (a token) is called a **“gram”**.
* Creating the vocabulary of two-word pairs is called a **bigram model**. 
* The process of converting the NLP text into numbers is called **vectorisation** in ML.
* There are different ways to convert text into the vectors :
 - Counting the number of times that each word appears in the document.
 - I am calculating the frequency that each word appears in a document out of all the words in the document.



## References
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* [Neptune.ai](https://neptune.ai/blog/vectorization-techniques-in-nlp-guide) 
