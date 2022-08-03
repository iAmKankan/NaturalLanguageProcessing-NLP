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
