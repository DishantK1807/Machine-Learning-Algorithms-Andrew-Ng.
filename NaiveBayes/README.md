#Naive Bayes

- Naive Bayes method is a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features. 
- In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations, famously document classification and spam filtering. 
- They require a small amount of training data to estimate the necessary parameters. 

###Sample Implementation

- The dataset is a preprocessed subset of the Ling-Spam Dataset, provided by Ion Androutsopoulos. 

- It is based on 960 real email messages from a linguistics mailing list.

- The data is pre-processed, only requires implementation of `Naive Bayes`.

- The dataset is split into two subsets: 
  - a 700-email subset for training
  - a 260-email subset for testing. 
  
- Each of the training and testing subsets contain 50% spam messages and 50% nonspam messages.

- Additionally, the emails have been preprocessed in the following ways:
  - **1. Stop word removal:** Certain words like "and," "the," and "of," are very common in all English sentences and are not very meaningful in deciding spam/nonspam status
- **2. Lemmatization:** Words that have the same meaning but different endings have been adjusted so that they all have the same form. For example, "include", "includes," and "included," would all be represented as "include." All words in the email body have also been converted to lower case.
- **3. Removal of non-words:** Numbers and punctuation have both been removed. All white spaces (tabs, newlines, spaces) have all been trimmed to a single space character.

- The lines of `train-features.txt` document have the following form:
```
2 977 2
2 1481 1
2 1549 1
```
  The first number in a line denotes a document number, the second number indicates the ID of a dictionary word, and the third number is the number of occurrences of the word in the document. So in the snippet above, the first line says that Document 2 has two occurrences of word 977. To look up what word 977 is, use the `feature-tokens.txt` file, which lists each word in the dictionary alongside an ID number.

- Classify an email as spam if : `log p(x|y=1)+log p(y=1) > log p(x|y=0)+log p(y=0)`

####Output:

- A sample run gives the classification error as: `fraction_wrong =  0.019231`.


