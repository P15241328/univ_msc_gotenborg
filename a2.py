import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
import scikitplot as skplt

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.



# Function for Part 1
#1. To preprocess the text (lowercase and lemmatize; punctuation can be preserved as it gets its own rows).
class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def lemmatize(self, text, remove_stop=True):
        if remove_stop and (text in self.stop_words):
            return np.nan
        else:
            return self.lemmatizer.lemmatize(text)


def preprocess(inputfile,remove_stop=False):
    def parse_line(line):
        file_line = (line.split('\t'))
        file_line[len(file_line) - 1] = file_line[len(file_line) - 1].split('\n')[0]
        return file_line[1:]  # ignore line sequence id

    # df = pd.read_csv('GMB_dataset.txt',sep='\t', lineterminator='\r',header=0)

    # read text from file
    lines = inputfile.readlines()  # read all lines
    if (inputfile is not None):
        inputfile.close()

    # Parse file rows
    num_rows = len(lines) - 1
    header = (parse_line(lines[0]))
    parse_lines = [parse_line(line) for line in lines[1:]]

    # Convert to a dataframe
    df = pd.DataFrame(parse_lines, columns=header)
    df['Word'] = df['Word'].apply(lambda x: str(x).lower())

    # Lemmatise and remove stop
    lt = LemmaTokenizer()
    df['Word'] = df['Word'].apply(lambda x: lt.lemmatize(x, remove_stop))

    #split tag column into prefix and entities
    df_tags = df['Tag'].str.split('-', expand=True)
    df.insert(len(df.columns), 'Tag_prefix', df_tags.iloc[:, 0])
    df.insert(len(df.columns), 'Tag_entity', df_tags.iloc[:, 1])

    # Drop rows that have been tagged as stop/NAN
    df.dropna(inplace=True, subset=["Word"])

    df['Word_seq'] = df.groupby('Sentence #').cumcount()+1

    return df

# Code for part 2
class sentence:
    def __init__(self):
        self.id = None
        self.entities = []
        self.sentence_text = ""
        self.sentence_list = []
        self.length = 0

    def create(self, r):
        self.sentence_id = r[0]
        self.sentence_text = r[1][0]
        self.sentence_list = r[1][1]
        self.entities = r[1][2]

        self.length = len(self.sentence_list)  # number of entities/words present in the sentence
        for entity in self.entities:
            entity.set_parent(self)

    def previous_entities(self, position, n):
        if position > self.length:
            ValueError('Invalid parameter specific for position', position)

        return self.entities[:position - 1][-n:]

    def next_entities(self, position, n):
        if position > self.length:
            ValueError('Invalid parameter specific for position', position)

        return self.entities[position:][:n]

    def get_features(self, n=5, pad=True,skip_ne=True):
        VALID_TAGS = ['art', 'eve', 'geo', 'gpe', 'nat', 'org', 'per', 'tim']
        words = []

        if not isinstance(n, int) or n < 0:
            ValueError('Invalid parameter specific for n', n)

        # Generate features
        encode_strings = ["S{}".format(i) for i in range(1, n + 1)]

        sentence_features = []
        for entity in self.entities:

            # check whether tag is valid
            if entity.tag_entity in VALID_TAGS:
                # ---previous---
                prev_features = [ent.word for ent in self.previous_entities(entity.position, n) if not (ent.isne and skip_ne)]

                # paddings
                paddings = encode_strings[len(prev_features):]
                if pad and len(paddings) > 0:
                    prev_features = paddings + prev_features

                # ---next---
                next_features = [ent.word for ent in self.next_entities(entity.position, n) if not (ent.isne and skip_ne)]

                # paddings
                paddings = encode_strings[len(next_features):][::-1]
                if pad and len(paddings) > 0:
                    next_features = next_features + paddings

                #concat both prev and next
                sentence_features.append(
                    pd.DataFrame.from_dict({entity.tag_entity: prev_features + next_features}, orient='index'))

        # concat all dataframes
        if sentence_features:
            return pd.concat(sentence_features)
        else:
            return pd.DataFrame()


# entity
class entity:
    def __init__(self, word, pos, tag_prefix, tag_entity, position):
        self.word = word
        self.pos = pos  # part of speech
        self.tag_prefix = tag_prefix
        self.tag_entity = tag_entity
        self.position = position  # position of word in text
        self.isfirst = (position == 1)
        self.islast = None
        self.isne   = not (tag_entity is None)
    def __str__(self):
        return self.word

    def set_parent(self, parent):
        self.parent = parent
        self.islast = (self.position == parent.length)


class gmb_processor:
    def __init__(self, data):
        self.data = data
        self.df_aggregated_sentences = None
        self.sentences = []

    def __aggregator_fuction(self, s):
        sentence_list = []
        sentence_entities = []
        sentence_text = " "

        # iterate
        position = 0
        for (w, p, tp, te) in (zip(s["Word"].values.tolist(),
                                   s["POS"].values.tolist(),
                                   s["Tag_prefix"].values.tolist(),
                                   s["Tag_entity"].values.tolist())):

            if str(w).isalnum():  # ignore punctuation
                sentence_list.append(w)
                sentence_entities.append(entity(w, p, tp, te, position + 1))
                position = position + 1

        sentence_text = " ".join(sentence_list)

        return pd.Series({"sentence_text": sentence_text,
                          "sentence_list": sentence_list,
                          "sentence_entities": sentence_entities})

    def __aggregate(self):  # Agg df to sentences
        ds_result = self.data.groupby("Sentence #").apply(self.__aggregator_fuction)
        ds_result["Sentence #"] = ds_result.index.astype(float).astype(int)

        return ds_result

    def fit(self):
        # Agg df to sentences
        self.df_aggregated_by_sentence_id = self.__aggregate()

        # Convert df to sentence instance
        for row in self.df_aggregated_by_sentence_id.iterrows():
            sent = sentence()
            sent.create(row)
            self.sentences.append(sent)
        return self.sentences

    def get_instances(self,n, pad,skip_ne):
        if (len(self.df_aggregated_by_sentence_id) == 0) or (len(self.sentences) == 0):
            raise Exception("You need to run fit() first")

        instances = []
        # iterate over all sentences
        for sent in self.sentences:
            instances.append(sent.get_features(n,pad,skip_ne))

        return pd.concat(instances)

def create_instances(data,n=5, pad=True,skip_ne=True):
    #processor
    processor = gmb_processor(data)

    #Convert df to sentences
    sentences = processor.fit()

    #return feature instances as a df
    df_instances = processor.get_instances(n=n,pad=pad,skip_ne=skip_ne)
    return df_instances


# Code for part 3
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer

def create_table(instances,method='tfidf'):
    #Convert instances to text documents
    ds_instance_documents = instances.apply(lambda row: " ".join([elem for elem in row]),axis=1)

    if method=='cv':
        #Count vectorizer
        vectorizer = CountVectorizer()
        result = vectorizer.fit_transform(ds_instance_documents.to_list()).toarray()

        df = pd.DataFrame(data=result, columns=vectorizer.get_feature_names())
        df.insert(0, '_class_', ds_instance_documents.index)

    elif method=='tfidf':
        #tfidf
        #Mindf = 1. Include all tokens that appear at least in one document (all tokens)
        #Maxdf = 1.0. Include all tokens that appear in ALL documents (100%) (all tokens)
        tfidf = TfidfVectorizer(ds_instance_documents.to_list())
        result = tfidf.fit_transform(ds_instance_documents.to_list()).toarray()

        df = pd.DataFrame(data=result, columns=tfidf.get_feature_names())
        df.insert(0, '_class_', ds_instance_documents.index)

    else:
        ValueError('Invalid parameter specific for method',method)


    return df

def ttsplit(bigdf,test_size=0.2):
    X = bigdf.drop(columns=['_class_'],axis=1)
    y = bigdf['_class_']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

    return X_train, y_train.to_numpy(), X_test, y_test.to_numpy()


# Code for part 5
def confusion_matrix(truth, predictions,normalize=True):
    skplt.metrics.plot_confusion_matrix(truth, predictions, normalize=normalize)

# Code for bonus part B
def bonusb(filename):
    pass
