# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Mark Grech

The aim of this document is to briefly explain the inner workings and the methodology employed behind the development of assessment no. 2.

**Part 1**
The first part of the assignment entails the loading of the GMB data file. Loading of file simply occurs through the parsing of each row.
Each word is lemmatised to ensure the corpus is standardised as much as possible. In addition, words that are deemed
to be of no relevance or put differently, of little predictive power to NE class are excluded. 
The removal of such words resulted in the reduction of observations/rows by 34%. 

**Part 2**
The second part of the assignment involves the implementation of the core functionality as it entails the construction of sentences present within the text file and the identification of words that are deemed to be key Named Entities. Each observation is tagged with a sentence number which, when considered in sequential order of loading, the group of words constitute to a single sentence.
The logic behind the transformation of observations stored in the original dataframe to class instances and back to a dataframe is performed by the gmb_processor class. The class implements the following methods/functions:

**Entity**
Class which stores the properties associated with each word namely tag, POS (part-of-speech), position of word in the entire sentence as well as other helper properties (eg. isFirst, isLast,isNE). 

**gmb_processor**
1. __aggregator_function(). A row iterator function that removes punctuation, constructs each sentence and creates an instance of class Entity.     
2. __aggregate(). Groups observations by Sentence# and applies the aggregator function to the entire dataset.
3. fit(). Converts the aggregated dataset to a list of class instances.
4. get_instances(n,pad,skip_ne). The main function which instigates the transformation process and returns a dataset having Named Entity Classes and the n surrounding words that happen to occur on the left/right hand side of the NE.
                                 n - Determines the number of words that should be considered before and after the NE. If n=5 then the return dataset consists of (x classes/rows, n*2 words/columns )   
                                 pad - Whether padding should be applied in order to retain the position of words in relation to the beginning and end of the sentence. 
                                 skip_ne - Determines whether words that happen to be NE should be removed from the peripheral words.
   

**Part 3**
Responsible for the vectorisation of class instances. This implies the conversion of class and the associated words to vectors through the application of BOW or TFIDF techniques.
The target feature named _class_ represents the NE which is of categorical data type and consists of 8 levels namely   
'art', 'eve', 'geo', 'gpe', 'nat', 'org', 'per', 'tim'. A description of each class follows:
geo = Geographical Entity
org = Organization
per = Person
gpe = Geopolitical Entity
tim = Time indicator
art = Artifact
eve = Event
nat = Natural Phenomenon

The second part partitions the dataset to the typical train/test subsets with an 80:20 ratio using the random holdout method (no bootstapping). No stratification sampling has been applied. 

**Part 4**
The training of the Support Vector based model occurs and the model's performance is evaluated on both the train and test subsets.

**Part 5**
The predictive performance of the SVM data model is evaluated against the accuracy metric. It has been observed that high accuracy scores are yielded on the training dataset (0.9) whilst lower scores (0.5) are achieved on the test dataset. These figures suggest model overfitting whereby the SVM model has learnt well the associations between the words and the respective NE classes. However, the model is unable to generalise well on unseen/test data. As a matter of fact the 50% accuracy score on test data suggests that the model is as good as a random guess.

The normalised confusion matrix portrays low True Positive (TP) rates as well as severe multi-class misclassifications.
By way of example, only 28% and 38% the GPE and NAT classes respectively were properly classified.
The EVE Named Entity achieved the highest misclassification score, most of which (32%) where incorrectly classified as a ORG Named Entity class.

![confusion matrix](https://github.com/P15241328/univ_msc_gotenborg/blob/main/images/confusion%20matrix.png)

Given that the data modelling phases is typically carried out iteratively, the following actions were taken in attempt to improve the accuracy of the linear SVM model.
1. Increase the number of boundary words with n=10, but resulted in a sparser dataset. Part 2.
2. Inclusion of other NE in boundary words during the building of class instances. Part 2.
3. The removal of padding using surrogate words such as S1,S2,S3,S4,S5. Part 2.
4. The inclusion of TFIDF instead of BOW vectorisation method. Part 3.
5. Evaluating the performance of other ML algorithms through the use of LazyClassifer library.

![comparision of different ML classification algorithms](https://github.com/P15241328/univ_msc_gotenborg/blob/main/images/model_comparison_accuracy.jpg)

**Bonus Part A**
Through basic exploratory data analysis it is clear that the Named Entity dataset is highly imbalanced.
The GEO class is represented by 25% of the class instances whilst the NAT class is highly under-represented (only 0.3% of the instances) 
It is to be expected that such a highly imbalanced dataset result in misclassification of the under-represented classes.

NE Class Percentage of class instances 
geo      25.51%
per      23.02%
org      20.87%
tim      15.24%
gpe      13.32%
eve      0.87%
art      0.86%
nat      0.31%

![class disribution](https://github.com/P15241328/univ_msc_gotenborg/blob/main/images/class%20distribution.png)
