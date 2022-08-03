## Index
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

### BOW, TFIDF, Unigram Bigram

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
* We treat each sentence as a separate document and we make the list of all words containt by those documents excluding the punctuation.
* We get- 

**[‘It’, ’is’, ’going’, ‘to’, ‘rain’, ‘today’ ‘I’, ‘am’, ‘not’, ‘outside’, ‘watch’, ‘the’, ‘season’, ‘premiere.’]**
