## Table of Content
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

* [Natural Language Processing](#natural-language-processing)
* [Word Embeddings or Word vectorization](#word-embeddings)
* [TF-IDF](https://github.com/iAmKankan/NaturalLanguageProcessing-NLP/blob/master/tfidf.md)
* [Main approaches in NLP](#main-approaches-in-nlp)
  * [Rule Based Approach](#rule-based-approach)
  * [Machine Learning or Traditional Approach](#machine-learning-or-traditional-approach)
  * [ Deep Learning Approach](#deep-learning-approach)
 
* [Preprocessing](https://github.com/iAmKankan/NaturalLanguageProcessing-NLP/blob/master/preprocessing.md)
* [NLP Terminologies](#nlp-terminologies)
* [Libraries used for NLP](#libraries-used-for-nlp)
* [NLP Uses](#nlp-uses)

## Natural Language Processing
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* **Natural language processing (NLP)**: It is the branch of artificial intelligence that helps computers understand, interpret and manipulate human language. 
* NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.

### Natural Language Processing and Deep Learning
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* The developments in the field of deep learning that have led to massive increases in performance in NLP.
* Deep learning, which is the study of Neural Networks, has revolutionised our understanding of machine learning in the past decade.
* NLP being a subset of machine learning has also been a beneficiary of this revolution.
* Before deep learning, the main techniques used in NLP were the bag of words model and techniques like TF-IDF, Naive Bays and Support Vector Machine(SVM).
* In fact, this is a quick, robust, simple system in today standerd.


* _**Before Deep Learning**_--

<img src="https://user-images.githubusercontent.com/12748752/134845951-a33b34fc-5e32-45bd-8cc9-cbcd7c474669.png" width=50%/>


* **Nowadays**, in advanced areas of NLP, we use techniques like **Hidden Markov Models** to do things like *speech recognition* and *parts of speech tagging*.

* **Problem with Bag of Words**
    > Consider the phrases - *dog toy* and *toy dog*
    
    >  These are different things, but in a Bag of Words Model ordered does not matter, and so these would be treated the same.
* **Solution**
    > **Neural Network**- Modeling sentences as sequences and as hierarchy(**LSTM**) has led to state of the art improvements over previous go to techniques.
    
    > **Word Embeddings**- These give words a neural representation so that words can be plugged into a Neural Network just like any other feature vector.



 
### Feature Vector
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* A vector is a series of numbers, like a matrix with one column but multiple rows, that can often be represented spatially. 
* A feature is a numerical or symbolic property of an aspect of an object. A feature vector is a vector containing multiple elements about an object.
* Putting feature vectors for objects together can make up a feature space.
* Let's start with a basic machine learning problem.
* **Example 01**: 
    * Suppose I have some books about biology and physics and I would like to be able to separate the biology books and the physics books using only a feature vector that is related somehow to its content.
    *  I chose the words chromosome and gravity.
    * Now I can make a table for each book I can see **how many times that the word chromosome appear**, **how many times does the word gravity appear.**
    * Well quite predictably the biology books are going to have the word chromosome appear a large number of times whereas the physics books are going to have the word gravity appear a large number of times.

<img src="https://user-images.githubusercontent.com/12748752/134860195-90fd0094-db46-4da7-9758-9915d80ff8f6.png" width=50% height=50%>

* **Advantages**
* For this example-
    * And of course you should notice something interesting all the words which are related to each other tend to show up close together and they form clusters.
    * All the words from the biology book show up close together.
    
<img src="https://user-images.githubusercontent.com/12748752/134924166-041de71c-6be5-4eff-b1ac-c763231fd306.png" width=60% >

### Word Embeddings
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* **Word Embeddings or Word vectorization is a methodology in NLP to map words or phrases from vocabulary to a corresponding vector of real numbers which used to find word predictions, word similarities/semantics.**
* The operation (text to number) or vectorization is done either on **"word"** or on "**character**" level.

* The process of converting words into numbers are called **Vectorization**.
* This is one of the most important advances in Deep NLP research.
* Word Emeddings allow you to map words into a vector space.
* Once you can represent something as a vector, you can perform arithmetic on it.
* So this is where the famous **king - man = queen - woman**, **December - Novemeber = July - June** or **France - Paris = England - London** come from.
* There are two most popular algorithms for finding Word Embeddings-
    - **Word2vac**
    - **GloVe**
### What is Word Embedding?
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)

* Word embedding is just a fancy name for a feature vector that represents a word.
* We can take a categorical object a word in this case and then map this object to a list of numbers in other words a vector we say we have embedded this word into a vector space. So that's why we call them word embeddings.

![featurevectors](https://ds055uzetaobb.cloudfront.net/brioche/uploads/JERsKXkW4T-screen-shot-2016-05-05-at-123118-pm.png?width=800)


##  Word Analogy 
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* One of the most popular applications of word embeddings is word analogies.
* This is where the famous **king - man = queen - woman** comes from.
* So,these two vectors are approximately the same.
* It means their magnitude and direction are very close to one another.
* **How do we actually find these analogies?**
*  **king - man + woman = ?** .  Find the fourth word representedby this unknown vector
* what I want to do is find the word in my vocabulary whose word vector is closest to King  - man + woman

```
closest-distance =infinity
best_word = None
test_vector= king - man + woman

for word, vector in vocabulary:
    distance = get_distance (test_vector, vector)
    if distance < closest distance:
        closest_distance=distance
        best_word = word
```
* Note: we can use built-in Numpy-stack functions to vectorize this.
* In Practical scenario we use Euclidean distance, Cosine similarity, Manhattan distance. etc
* Most common use is cosine distance.
* Ultimately you should choose what yields the best results for your application.

* **One remarkable fact about Neural Word Embedding algorithms is that you can't find these analogies at all.**
    * If we have V words, and each is represented by a vector if size D, what do we have - a V x D matrix!
    * What's remarkable is algorithms like word2vec and Glove have no concept of analogies.




![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)













### NLP Uses
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
- Semantic Analysis
- Automatic summarization
- Text classification
- Question Answering
- Some real-life example of NLP is IOS Siri, the Google assistant, Amazon echo.

### Main approaches in NLP
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

* In order to comprehend and interpret human language, NLP adopts 3 essential approaches to execute NLP tasks.
* All the 3 approaches are well-recognized and are widely used across numerous segments. 
* The evolutionary approaches have standalone benefits and continue to aid NLP task to deliver best results.

#### Rule Based Approach
 ![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
   
   - Regular expressions
   - Context-free grammars

A rule-based approach is perfect to acquire a specific language phenomenon: it efficiently decodes the linguistic relationship between words to translate the sentence.

* It is easily achieved through focus on pattern-matching or parsing.
* It can be counted as the ‘fill in the blanks’ method.
* It offers high performance cases when used specifically but fails to impress when generalized.

Hence, it’s vital to make a good choice for query analysis that’s meant to perform.

#### Machine Learning or Traditional Approach
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)

Traditional machine learning approach is described by:

* Training data or annotated corpus – one that has a corpus with mark-up.
* Feature engineering – capitalized, singulars, word type, surrounding words, etc.
* A general ML system of a training set or training a model on defined parameters, followed by fitting on test data.
* Applying model to test data or inference which is characterized by finding most probable words, next word, best category, etc.
* ‘Semantic slot filling’ tag the words or tokens which carry meaning to the sentences or translate utterances to logical form.



* Likelihood maximization
* Linear classifiers




| Rule-based Grammar                         	| Machine Learning algorithm                	|
|:--------------------------------------------	|:-------------------------------------------	|
| **Advantages**                               	| **Advantages**                               	|
| It’s easily adaptable                      	| It’s can scale effortlessly               	|
| Simple to debug                            	| Learnability without clear programming    	|
| Enormous training corpus not needed        	| Quick development if dataset is available 	|
| Comprehends the language                   	| High recall coverage                      	|
| High perfection                            	|                                           	|
|**Disadvantages**                             	| **Disadvantages**                            	|
| Proficient developers & linguists required 	| Training corpus with annotation needed    	|
| Slow parser development                    	| Hard to debug                             	|
| Moderate recall (coverage)                 	| Zero understanding of the language        	|


### Deep Learning Approach
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)

* Recurrent Neural Networks (RNNs)
*  Convolutional Neural Networks (CNNs)

* This is similar to "traditional" machine learning, but with a few differences:

    * feature engineering is generally skipped, as networks will "learn" important features (this is generally one of the claimed big benefits of using neural networks for NLP)
    * instead, streams of raw parameters ("words" -- actually vector representations of words) without engineered features, are fed into neural networks
    * very large training corpus
 

### NLP Terminologies
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

* **Weights and Vectors:**
    - TF-IDF,
    - length(TF-IDF, doc),
    - Word Vectors, 
    - Google Word Vectors

* **Text Structure:**
    - Part-Of-Speech Tagging, 
    - Head of sentence,
    - Named entities

* **Sentiment Analysis:**
    - Sentiment Dictionary,
    - Sentiment Entities,
    - Sentiment Features

* **Text Classification:**
    - Supervised Learning, 
    - Train Set, 
    - Dev(=Validation) Set, 
    - Test Set, 
    - Text Features,
    - LDA.

* **Machine Reading:**
    * Entity Extraction, 
    * Entity Linking,
    * dbpedia, 
    * FRED (lib) / Pikes.



![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)


* For solving NLP problems we can use two types of approaches **1. Word level**, **2. Character level**.

![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)






### Libraries used for NLP
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)

 * We usually use these libraries in NLP, which are:
 
|  NLTK (Natural language Tool kit)| TextBlob| CoreNLP| Polyglot| Gensim| SpaCy| Scikit-learn| Megatron library|
|---|---|---|---|---|---|---|---|

### Tokenization
* Tokenisation is the act of breaking a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements called tokens. 
* Tokens can be individual words, phrases or even whole sentences. 
* In the process of tokenisation, some characters like punctuation marks are discarded.

```
'Natural Language Processing'
['Natural', 'Language', 'Processing']

```


### Stemming
* **Stemming**: It is the process of reducing inflexions in words to their root forms such as mapping a group of words to the same stem even if stem itself is not a valid word in the Language.

|    | Word        | StemWord |
|----|-------------|----------|
| 0  | Connect     | Connect  |
| 1  | Connections | Connect  |
| 2  | Connection  | Connect  |
| 3  | Connects    | Connect  |
| 04 | Connected   | Connect  |

### Lemmatisation
* **Lemmatisation**: It is the process of the group together the different inflected forms of the word so that they can be analysed as a single item. It is quite similar to stemming, but it brings context to the words. So it links words with similar kind meaning to one word.

![Lammi](lammi.png)

## Word Embeddings or Word vectorization


### Bag-of-words model
* We need the way to represent text data for the machine learning algorithms(In vector of number), and the bag-of-words model helps us to achieve the task.
* It is the way of extracting features from the text for the use in machine learning algorithms.
* **In this approach, we use the tokenised words for each of observation and find out the frequency of each token.**

---
* Let’s do an example to understand this concept in depth.
```
“It is going to rain today.”
“Today, I am not going outside.”
“I am going to watch the season premiere.”
```
* We treat each sentence as the separate document and we make the list of all words from all the three documents excluding the punctuation.
* We get- 
```
[‘It’, ’is’, ’going’, ‘to’, ‘rain’, ‘today’ ‘I’, ‘am’, ‘not’, ‘outside’, ‘watch’, ‘the’, ‘season’, ‘premiere.’]
```
* The next step is the create vectors. Vectors convert text that can be used by the machine learning algorithm.
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
* Creating the vocabulary of two-word pairs is called a bigram model. 
* The process of converting the NLP text into numbers is called **vectorisation** in ML.
* There are different ways to convert text into the vectors :
 - Counting the number of times that each word appears in the document.
 - I am calculating the frequency that each word appears in a document out of all the words in the document.





## spaCy:
* This is completely optimized and highly accurate library widely used in deep learning.






## NLP-NLTK
