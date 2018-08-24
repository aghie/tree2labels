from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Input, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
# Fit LabelEncoder with our list of classes
from sklearn.preprocessing import LabelEncoder

import keras
import numpy as np


"""
Class that implements a simple feed-forward network with one hidden layer that is used
for sequential labeling using a window both of previous and upcoming context.
It is fed with word and postag embeddings
"""
class EmbeddedPerceptronTagger(object):
    
    EMPTY = "-EMPTY-"

    def __init__(self, hidden_neurons, vocab, postags, n_labels, context_len=3):

        self.vocab = vocab.add("-UNKNOWN-")
        self.postags = postags.add("-UNKNOWN-")
        self.iforms = {self.EMPTY:0}
        self.iforms.update({w:i for i,w in enumerate(sorted(vocab),1)})

        self.ipostags = {self.EMPTY:0}
        self.ipostags.update({p:i for i,p in enumerate(sorted(postags),1)})
        

        self.iforms_reverse = {self.iforms[w]:w for w in self.iforms}
        input = Input(shape=(context_len,), dtype='float32')
        input_tags = Input(shape=(context_len,), dtype='float32')

        embedding_layer = Embedding(len(self.iforms),
                                    100,
                                    embeddings_initializer="glorot_uniform",
                                    input_length=context_len,
                                    name = "e_IW",
                                    trainable=True)(input)
                                    
        pos_embedding_layer = Embedding(len(self.ipostags),
                                        20,
                                        embeddings_initializer="glorot_uniform",
                                        input_length=context_len,
                                        name = "e_IP",
                                        trainable=True)(input_tags)

        x = keras.layers.concatenate([embedding_layer, pos_embedding_layer], axis=-1)
        
        dr = 0.5
        for l in range(0, 1):
            x = Dense(hidden_neurons)(x)
            x = Dropout(0.5)(x)  
            x = Flatten()(x)
            x = Activation('relu')(x)
            
        preds = Dense(n_labels, activation='softmax')(x)
        self.model = Model(inputs = [input,input_tags], outputs=[preds])        
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',#keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=2e-6, nesterov=False),#'sgd',
                      metrics=['accuracy'])
      
    
 
    def add_basic_features(self, sent,i , prev_words, next_words):
        

        words = [self.iforms[sent[i][0]] if sent[i][0] in self.iforms else self.iforms["-UNKNOWN-"]]
        postags = [self.ipostags[sent[i][1]] if sent[i][1] in self.ipostags else self.iforms["-UNKNOWN-"]]

        for j in range(1,prev_words+1):
            iaux = i-j
            if i > 0:
                word1 = self.iforms[sent[iaux][0]] if sent[iaux][0] in self.iforms else self.iforms["-UNKNOWN-"]
                postag1 = self.ipostags[sent[iaux][1]] if sent[iaux][1] in self.ipostags else self.ipostags["-UNKNOWN-"]
                words.append(word1)
                postags.append(postag1)
            else:
                words.append(self.iforms["-EMPTY-"])
                postags.append(self.ipostags["-EMPTY-"])
                
                

        for j in range(1,next_words+1): 
            iaux = i+j
            if i < len(sent)-j:
                word1 = self.iforms[sent[iaux][0]] if sent[iaux][0] in self.iforms else self.iforms["-UNKNOWN-"]
                postag1 = self.ipostags[sent[iaux][1]] if sent[iaux][1] in self.ipostags else self.ipostags["-UNKNOWN-"]
                words.append(word1)
                postags.append(postag1)
            else:
                words.append(self.iforms["-EMPTY-"])
                postags.append(self.ipostags["-EMPTY-"])
                
        return words,postags
    
    
    def transform(self,sentences, previous, next):
        """
        Split tagged sentences to X and y datasets and append some basic features.
     
        :param tagged_sentences: a list of POS tagged sentences
        :param tagged_sentences: list of list of tuples (term_i, tag_i)
        :return: 
        """
        X, X_tags, y = [], [], []
        for sentence in sentences:
         #   print sentence
            for index, (word, postag, label) in enumerate(sentence):
                # Add basic NLP features for each token in the snippet
                aux = self.add_basic_features(sentence, index,previous,next)
                X.append(np.array(aux[0]))
                X_tags.append(np.array(aux[1]))
                y.append(label)
        return X, X_tags, y
    
   
    def transform_test(self,sentences, previous, next):
        """
        Split tagged sentences to X and y datasets and append some basic features.
     
        :param tagged_sentences: a list of POS tagged sentences
        :param tagged_sentences: list of list of tuples (term_i, tag_i)
        :return: 
        """
        X, X_tags, y = [], [], []
        for sentence in sentences:
            for index, (word, postag) in enumerate(sentence):
                # Add basic NLP features for each sentence term
                aux = self.add_basic_features(sentence, index,previous,next)
                X.append(np.array(aux[0]))
                X_tags.append(np.array(aux[1]))
            
        return X, X_tags
    
    
    
    def samples(self,x_source, x_tags_source, y_source, size, label_encoder):
        while True:
            for i in range(0, x_source.shape[0], size):
                j = i + size 
                if j > x_source.shape[0]:
                    j = x_source.shape[0]
                yield [x_source[i:j], x_tags_source[i:j]], np_utils.to_categorical(y_source[i:j],   num_classes=len(label_encoder.classes_))


  
    def samples_test(self,x_source, x_tags_source, size):
        while True:
            for i in range(0, x_source.shape[0], size):
                j = i + size
                
                if j > x_source.shape[0]:
                    j = x_source.shape[0]
        
                yield [x_source[i:j], x_tags_source[i:j]]
    




"""
Class that implements a simple feed-forward network with one hidden layer that is used
for sequential labeling using a window both of previous and upcoming context.
It is fed with word and postag embeddings
"""
class PerceptronTagger(object):
    """
    Based on the tutorial https://techblog.cdiscount.com/part-speech-tagging-tutorial-keras-deep-learning-library/
    """
    
    def __init__(self,input_dim, hidden_neurons, output_dim):
        """
        Construct, compile and return a Keras model which will be used to fit/predict
        """
        self.model = Sequential([
            Dense(hidden_neurons, input_dim=input_dim),
            Activation('relu'),
            Dropout(0.5),
            Dense(output_dim, activation='softmax')
        ])
     
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        #return model
        

    @classmethod
    def builder(cls):
        return PerceptronTagger(1,1,1)

   
    def add_basic_features(self, sent, i, prev_words, next_words):
    
        word = sent[i][0]
        postag = sent[i][1]
        features = {
            'is_first': i == 0,
            'is_last': i == len(sent) - 1,
            'word.lower=': word.lower(),
            'word[-3:]=': word[-3:],
            'word[-2:]=': word[-2:],
            'word.isupper=': word.isupper(),
            'word.istitle=': word.istitle(),
            'word.isdigit=': word.isdigit(),
            'postag=': postag,
            'postag[:2]=': postag[:2]
        }
        
        for j in range(1,prev_words+1):
            iaux = i-j
            if i > 0:
                word1 = sent[iaux][0]
                postag1 = sent[iaux][1]
                features.update({
                    '-'+str(j)+':word.lower=': word1.lower(),
                    '-'+str(j)+':word.istitle=':  word1.istitle(),
                    '-'+str(j)+':word.isupper=': word1.isupper(),
                    '-'+str(j)+':word.isdigit=': word1.isdigit(),
                    '-'+str(j)+':postag=': postag1,
                    '-'+str(j)+':postag[:2]=': postag1[:2]
                })


        for j in range(1,next_words+1): 
            iaux = i+j
            if i < len(sent)-j:
                word1 = sent[iaux][0]
                postag1 = sent[iaux][1]
                features.update({
                    '+'+str(j)+':word.lower=': word1.lower(),
                    '+'+str(j)+':word.istitle=': word1.istitle(),
                    '+'+str(j)+':word.isupper=': word1.isupper(),
                    '+'+str(j)+':word.isdigit=': word1.isdigit(),
                    '+'+str(j)+':postag=': postag1,
                    '+'+str(j)+':postag[:2]=': postag1[:2]})

        return features



    def samples(self,x_source,  y_source, size):
        while True:
            for i in range(0, x_source.shape[0], size):
                j = i + size
                
                if j > x_source.shape[0]:
                    j = x_source.shape[0]
 
                yield x_source[i:j].toarray(), y_source[i:j]#.toarray()    


    def samples_test(self,x_source, size, dict_vectorizer):
        while True:
            for i in range(0, len(x_source), size):
            #for i in range(0, x_source.shape[0], size):
                j = i + size
                
                if j > len(x_source):#x_source.shape[0]:
                    j = len(x_source)#x_source.shape[0]
 
                yield dict_vectorizer.transform(x_source[i:j]).toarray()# x_source[i:j].toarray()#.toarray()   

    

    def transform(self,sentences, previous, next):
        """
        Split tagged sentences to X and y datasets and append some basic features.
     
        :param tagged_sentences: a list of POS tagged sentences
        :param tagged_sentences: list of list of tuples (term_i, tag_i)
        :return: 
        """
        X, y = [],[]
     
        for sentence in sentences:
            for index, (word, postag, label) in enumerate(sentence):
                # Add basic NLP features for each sentence term
                aux = self.add_basic_features(sentence, index,previous,next)
                X.append(aux)
                y.append(label)
        return X,  y
    


    def transform_test(self,sentences,previous, next):
        """
        Split tagged sentences to X and y datasets and append some basic features.
     
        :param tagged_sentences: a list of POS tagged sentences
        :param tagged_sentences: list of list of tuples (term_i, tag_i)
        :return: 
        """
        X, y = [],[]
     
        for sentence in sentences:
            for index, (word, postag) in enumerate(sentence):
                # Add basic NLP features for each sentence term
                aux = self.add_basic_features(sentence, index,previous,next)
                X.append(aux)
            
        return X


def word2features(sent, i, prev_words, next_words):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'is_first=%s'% str(i == 0),
        'is_last=%s' % str(i == len(sent) - 1),
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    
    for j in range(1,prev_words+1):
        iaux = i-j
        if i > 0:
            word1 = sent[iaux][0]
            postag1 = sent[iaux][1]
            features.extend([
                '-'+str(j)+':word.lower=' + word1.lower(),
                '-'+str(j)+':word.istitle=%s' % word1.istitle(),
                '-'+str(j)+':word.isupper=%s' % word1.isupper(),
                '-'+str(j)+':word.isdigit=%s' % word1.isdigit(),
                '-'+str(j)+':postag=' + postag1,
                '-'+str(j)+':postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('-'+str(j)+':BOS')
            
    for j in range(1,next_words+1): 
        iaux = i+j
        if i < len(sent)-j:
            word1 = sent[iaux][0]
            postag1 = sent[iaux][1]
            features.extend([
                '+'+str(j)+':word.lower=' + word1.lower(),
                '+'+str(j)+':word.istitle=%s' % word1.istitle(),
                '+'+str(j)+':word.isupper=%s' % word1.isupper(),
                '+'+str(j)+':word.isdigit=%s' % word1.isdigit(),
                '+'+str(j)+':postag=' + postag1,
                '+'+str(j)+':postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('+'+str(j)+':EOS')
    
    return features

def sent2features(sent, ngram_prev, ngram_next):
    return [word2features(sent, i, ngram_prev, ngram_next) for i in range(len(sent))]

def sent2features_test(sent, ngram_prev, ngram_next):
    return [word2features(sent, i, ngram_prev, ngram_next) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]
            
def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
    return backoff


"""
Prepares the sentences, previously processed by an leaf unary chain tagger, for
the sequence labeling parser
@param sentences: A list of list of tuples (word,postag) for each sentence
@param unary_preds: A list of unary predictions
"""
def get_samples_retagged(sentences, unary_preds):
    
    unary_preds_aux = []
    
    ipos = 0
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for word,postag in sentence:
            
            if unary_preds[ipos] == "-EMPTY-" or word in ["-BOS-","-EOS-"]:
          #      if word not in  ["-BOS-","-EOS-"]:
          #          unary_preds_aux.append(postag)
                unary_preds_aux.append(postag)
                new_sentence.append((word,postag))
             #   f_aux.write("\t".join((word,postag,label))+"\n")
            else:
                unary_preds_aux.append(unary_preds[ipos]+"+"+postag)
                new_sentence.append((word,unary_preds[ipos]+"+"+postag))
              #  f_aux.write("\t".join((word,unary_preds[ipos]+"+"+postag,label))+"\n")
            
            ipos+=1
        new_sentences.append(new_sentence)
            
    unary_preds = unary_preds_aux 
    
    return new_sentences, unary_preds


"""
It changes to a predefined valid label missclassifications into the -BOS-, -EOS-
and NONE_X classes that occur in the middle on the sentence (and therefore they are not valid).
This happens marginally, but anyway we need to deal with it.
@returns A list of valid predictions 
"""
def process_labels(sentences, preds, label_encoder, unaries):
        
    if not unaries:
        dummy_eos_labels =  label_encoder.transform(["-EOS-"])
        dummy_bos_labels =  label_encoder.transform(["-BOS-"])
        dummy_none_labels = set(label_encoder.transform([e for e in list(label_encoder.classes_)
                                                        if e.startswith("NONE")]))
        
        #TODO: Workaround. This can be avoided if for sentences of length one we generate NONE and not ROOT,
        #which should make perfectly sense

        root_retagger = label_encoder.transform(["ROOT"]) if "ROOT" in label_encoder.classes_ else -1
        none_label = label_encoder.transform(["NONE"])
        
        try:
            root_label = label_encoder.transform(["ROOT_S"])[0]
        except ValueError:
            root_label = label_encoder.transform(["ROOT_IP"])[0]
            
        valid_none_indexes = set([])
        valid_eos_indexes = set([])
        valid_bos_indexes = set([])
        i = 0
        for s in sentences:
            valid_bos_indexes.add(i)
            valid_eos_indexes.add(i+len(s)-1)
            valid_none_indexes.add(i+len(s)-2)
            i+=len(s)
            
        preds = preds.argmax(axis=-1)
        for j,pred in enumerate(preds):
            if pred in dummy_eos_labels and j not in valid_eos_indexes: 
                preds[j] = root_label
            elif pred in dummy_bos_labels and j not in valid_bos_indexes: 
                preds[j] = root_label
            elif pred in dummy_none_labels and j not in valid_none_indexes:
                preds[j] = root_label
            #TODO: This is currently needed as a workaround for the retagging strategy and sentences of length one
            elif preds[j] == root_retagger:
                preds[j] = none_label 
           
    
    else:
        preds = preds.argmax(axis=-1)
    
    preds = list(label_encoder.inverse_transform(preds))

        
    return preds



def format_output(sentences,preds,unary_preds,retagger):
    if retagger:
        i=0
        j=0
        pred_aux = []
        pred_unary_aux = []
         
        for k,s in enumerate(sentences):

            pred_aux.append( preds[i:i+len(s)] )
            pred_unary_aux.append(unary_preds[j:(j+len(s))])
         
            i+=len(s)
            j+=len(s)
         
        preds = pred_aux
        unary_preds = pred_unary_aux
    else:
        i=0
        pred_aux = []
        for k,s in enumerate(sentences):
            pred_aux.append( preds[i:i+len(s)] )
            i+=len(s)
        preds = pred_aux     
        unary_preds = None
     
    return preds, unary_preds

