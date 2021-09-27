## Supervised Machine Learning
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
- In supervised machine learning you have input features X and a set of labels Y.
- The goal is to minimize your error rates or cost as much as possible.
- To do this, run the prediction function which takes in parameters data to map your features to output labels Ŷ.
- The best mapping from features to labels is achieved when the difference between the expected values Y and the predicted values Ŷ hat is minimized.
- The cost function F does this by comparing how closely the output Ŷ is to the label Y.
- Update the parameters and repeat the whole process until your cost is minimized.
![01](https://user-images.githubusercontent.com/12748752/134764768-3c35e880-a503-497d-99c3-bf6162caa0e7.png)


## Sentiment Analysis
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* Inorder to get started building a logistic regression classifier that's capable of predicting sentiments of an arbitrary tweet. 
    * You will first process the raw tweets in your training sets and extract useful features.
      * Tweets with a positive sentiment have a label of one, and the tweets with a negative sentiment have a label of zero.
    * Then you will train your logistic regression classifier while minimizing the cost. 
    * And finally you'll be able to make your predictions.
![02](https://user-images.githubusercontent.com/12748752/134764773-8352729a-6053-45c3-87c0-0ef837718f07.png)

### Feature Extraction
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

### Represent a text as a vector
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* We have to build a vocabulary and that will allow to encode any text or any tweet as an array of numbers.
* The vocabulary *V* would be the list of unique words from your list of tweets.
* To get that list, you'll have to go through all the words from all your tweets and save every new word that appears in your search.
* To extract features using your vocabulary. 
* We have to check if every word from the vocabulary appears in the tweet.
*  If it does like in the case of the word I, you would assign a value of 1 to that feature, like this.
*  If it doesn't appear, you'd assign a value of 0, like that. 
*  This non-zero values is called a sparse representation
![03](https://user-images.githubusercontent.com/12748752/134764775-5a6ab569-b3c0-4a14-8aec-ed2c5c99bb32.png)

*  **Problems with sparse represenation:**
    - A logistic regression model would have to learn N+1 parameters, where N is the size of the vocabulary *V* 
    - Large training time
    - Large prediction time

![04](https://user-images.githubusercontent.com/12748752/134764777-ba2dcbda-3bc3-4f4e-bde1-5c1c9da22d8d.png)

* **Negative and Positive Frequencies** For this particular example of sentiment analysis, you have two classes.
*  One class associated with positive sentiment and the other with negative sentiment. 
*  So taking your corpus, you'd have a set of two tweets that belong to the positive class, and the sets of two tweets that belong to the negative class. 
*  Let's take the sets of positive tweets. Now, take a look at your vocabulary.
*   To get the positive frequency in any word in your vocabulary, you will have to count the times as it appears in the positive tweets.
![05](https://user-images.githubusercontent.com/12748752/134764778-e30215db-26aa-4b8f-8731-cf401c99813f.png)

### Feature Extraction with Frequencies
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
![06](https://user-images.githubusercontent.com/12748752/134764780-4de98363-f86d-4b3a-84cd-dc4e6e3dbfd8.png)

## Preprocessing
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* We will use stemming and stop words to preprocess your texts.
### Stop Words
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* Those are frequently used words, punctuations.
### Stemming
  ![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* Stemming in NLP is simply transforming any word to its base stem.


