'''
It evaluates some traditional baselines using for regular PoS-tagging or chunking

It uses the implementations from the NLTK


TRAINING

PYTHONPATH=. python baselines/baselines.py \
--train /home/david.vilares/Escritorio/Papers/seq2constree/dataset/gold-tags-ptb-train.seqtrees \
--test /home/david.vilares/Escritorio/Papers/seq2constree/dataset/gold-tags-ptb-dev.seqtrees \
--out /home/david.vilares/Escritorio/Papers/seq2constree/baselines/gold-tags-ptb \
--status train

TEST

@author: david.vilares
'''

from argparse import ArgumentParser
from baseline_utils import *
from utils import sequence_to_parenthesis, flat_list, get_enriched_labels_for_retagger
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from keras.models import load_model
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Input, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
# Fit LabelEncoder with our list of classes
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# Convert integers to dummy variables (one hot encoded)

import keras
import codecs
import functools
import os
import nltk
import pickle
import tempfile
import time
import os
import numpy as np
import sys
import tensorflow as tf
import random as rn
import uuid



#Uncomment/Comment these lines to determine when and which GPU(s) to use
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

STATUS_TEST = "test"
STATUS_TRAIN = "train"
SPLIT_SYMBOL = "~"


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    
    
    arg_parser.add_argument("--train", dest="train", help="Path to the training file", default=None)
    arg_parser.add_argument("--test", dest="test", help ="Path to the development/test file", default=None)
   # arg_parser.add_argument("--dir", dest="dir", help="Path to the output directory where to store the models", default=None)
    arg_parser.add_argument("--model", dest="model", help="Path to the model")
   # arg_parser.add_argument("--name", dest="name", help="Path to the name of the file")
    arg_parser.add_argument("--baseline", dest="baseline", help="Path to the baseline directory. Options: [emlp|mlp|crf]", default=None)
    arg_parser.add_argument("--gold", dest="gold", help="Path to the gold file", default=None)
    arg_parser.add_argument("--status", dest="status", help="")
    arg_parser.add_argument("--prev_context",dest="prev_context",type=int, default=1)
    arg_parser.add_argument("--next_context", dest="next_context",type=int,default=1)
    arg_parser.add_argument("--retagger", dest="retagger", default=False, action="store_true")
    arg_parser.add_argument("--unary", dest="unary",default=False, action="store_true")
    arg_parser.add_argument("--output_unary", dest="output_unary", help="Use together with unary to store the output in the desired file")
    arg_parser.add_argument("--output_decode", dest="output_decode", help="Path to store the predicted trees", default="/tmp/trees.txt")
    arg_parser.add_argument("--evalb",dest="evalb",help="Path to the script EVALB")
    arg_parser.add_argument("--gpu",dest="gpu",default="False")
    
    args = arg_parser.parse_args()
    if args.status.lower() == STATUS_TEST:
        
        if args.gpu.lower() == "true":
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        #TODO: Change for a temporaryfile, but getting problems with the Chinese encoding
        tmpfile = codecs.open(args.output_decode,"w")     
         
        with codecs.open(args.test, encoding="utf-8") as f_dev:
            content = f_dev.read()
            gold_samples = [[ tuple(l.split("\t")) for l in sentence.split("\n")] for sentence in content.split("\n\n")
                                if sentence != ""]               
    
        sentences =[[(word,postag) for word, postag, label in sample] for sample in gold_samples] 
        
        #######################################################################
        #               EVALUATING A PERCEPTRON WITH EMBEDDINGS
        #######################################################################
        if args.baseline.lower() == "emlp":
            
            batch= 128          
            new_sentences = sentences
            unary_preds = None
            init_time = None
       
            with codecs.open(args.model+".emlp.labels") as f:
                label_encoder = pickle.load(f)                     
            with codecs.open(args.model+".emlp.features") as f:
                vocab,postags,all_labels, hidden_size, prev_context, next_context = pickle.load(f)        
            emlp_parser = EmbeddedPerceptronTagger(hidden_size, vocab, postags, len(all_labels))
            emlp_parser.model = load_model(args.model+".emlp.hdf5")

            #Loading and running the retagger, if needed
            if args.retagger:
                with codecs.open(args.model+"-unary.emlp.labels") as f:
                    unary_label_encoder = pickle.load(f)         
                with codecs.open(args.model+"-unary.emlp.features") as f:
                    re_vocab,re_postags,re_all_labels, re_hidden_size, re_prev_context, re_next_context = pickle.load(f)   
                    emlp_retagger = EmbeddedPerceptronTagger(re_hidden_size, re_vocab, re_postags, len(re_all_labels))
                    emlp_retagger.model = load_model(args.model+"-unary.emlp.hdf5")

                #The time starts here, applying the retagging, if needed                
                init_time = time.time()
                X_test_unary,X_tags_test_unary = emlp_retagger.transform_test(sentences, re_prev_context, re_next_context)
                X_test_unary = np.array(X_test_unary)
                X_tags_test_unary = np.array(X_tags_test_unary)

                unary_preds =  emlp_retagger.model.predict_generator(emlp_retagger.samples_test(X_test_unary,X_tags_test_unary,batch), 
                                                                     steps= (X_test_unary.shape[0]/batch)+1)     
                unary_preds = list(unary_label_encoder.inverse_transform ( unary_preds.argmax(axis=-1) ))  
                new_sentences, unary_preds = get_samples_retagged(sentences, unary_preds)
            
            #If we are not applying the retagging strategy, we start here to measure the time
            if init_time is None: 
                init_time = time.time() 

            X_test, X_tags_test = emlp_parser.transform_test(new_sentences, prev_context, next_context)
            X_test = np.array(X_test)
            X_tags_test = np.array(X_tags_test)   
            preds = emlp_parser.model.predict_generator(emlp_parser.samples_test(X_test,X_tags_test,batch), 
                                             steps= (X_test.shape[0]/batch)+1)     
            
            preds = process_labels(new_sentences, preds, label_encoder, args.unary)
            preds, unary_preds = format_output(new_sentences, preds, unary_preds, args.retagger)


        #######################################################################
        #              EVALUATING A ONE-HOT VECTOR PERCEPTRON
        #######################################################################
        elif args.baseline.lower() == "mlp":         
 
            new_sentences = sentences
            unary_preds = None
            batch= 128
            init_time = None
            
         #   loading_parsing_time = time.time()      
            with codecs.open(args.model+".mlp.features") as f:
                dict_vectorizer, hidden_size, prev_context, next_context = pickle.load(f)
            with codecs.open(args.model+".mlp.labels") as f:
                label_encoder = pickle.load(f)         
            mlp_parser = PerceptronTagger.builder() 
            mlp_parser.model = load_model(args.model+".mlp.hdf5")
         #   end_loading_parsing_time = time.time() - loading_parsing_time
            
            #Running the retagger, if needed
            if args.retagger:
                with codecs.open(args.model+"-unary.mlp.features") as f:
                    dict_unary_vectorizer,re_hidden_size, re_prev_context, re_next_context = pickle.load(f)    
                with codecs.open(args.model+"-unary.mlp.labels") as f:
                    unary_label_encoder = pickle.load(f)     
                
                mlp_retagger = PerceptronTagger.builder()       
                mlp_retagger.model = load_model(args.model+"-unary.mlp.hdf5")
                init_time = time.time()
                X_test_unary = mlp_retagger.transform_test(sentences, re_prev_context, re_next_context)
                unary_preds =  mlp_retagger.model.predict_generator(mlp_retagger.samples_test(X_test_unary,batch,
                                                                                            dict_unary_vectorizer), 
                                                              steps= (len(X_test_unary)/batch)+1)
                unary_preds = list(unary_label_encoder.inverse_transform ( unary_preds.argmax(axis=-1) ))
                new_sentences, unary_preds = get_samples_retagged(sentences, unary_preds)
                
            #If we are not applying the retagging strategy, we start here to measure the time
            if init_time is None: 
                init_time = time.time() 

            X_test = mlp_parser.transform_test(new_sentences, prev_context,next_context)
            preds = mlp_parser.model.predict_generator(mlp_parser.samples_test(X_test,batch, dict_vectorizer), 
                                             steps= (len(X_test)/batch)+1)   
          
            preds = process_labels(sentences, preds, label_encoder, args.unary)
            preds, unary_preds = format_output(new_sentences, preds, unary_preds, args.retagger)


        #######################################################################
        #             EVALUATING A CONDITIONAL RANDOM FIELDS
        #######################################################################
        elif args.baseline.lower() == "crf":
            
            new_sentences = sentences
            unary_preds = None
            init_time = None
            
            with codecs.open(args.model+".crf.pickle","rb") as f:
                crf_parser, prev_context, next_context = pickle.load(f)
            
            #Running the retagger
            if args.retagger:    
            
                with codecs.open(args.model+"-unary.crf.pickle","rb") as f:
                    crf_retagger, re_prev_context, re_next_context= pickle.load(f)
            
                init_time = time.time()
                X_test = [sent2features_test(s,re_prev_context, re_next_context) for s in new_sentences]
                unary_preds = crf_retagger.predict([x for x in X_test])
      
                unary_preds_aux =[]
                for unary_pred in unary_preds:
                    for element in unary_pred:
                        unary_preds_aux.append(element)
                        
                unary_preds = unary_preds_aux  
                new_sentences, unary_preds = get_samples_retagged(new_sentences, unary_preds)
             
            
            if init_time is None: 
                init_time = time.time() 

            X_test = [sent2features_test(s,prev_context, next_context) for s in new_sentences]
            preds = crf_parser.predict(X_test)
            
            preds_aux =[]
            for pred in preds:
                for element in pred:
                    preds_aux.append(element)
            preds = preds_aux
            
            preds, unary_preds = format_output(new_sentences, preds, unary_preds, args.retagger)
            
            #Postprocessing the labels for the CRF
            for j,pred in enumerate(preds):
                for k,p in enumerate(pred):
                    if (p in ["-EOS-","-BOS-"] or p.startswith("NONE")) and k != 0 and k < len(pred)-1:
                        pred[k] = "ROOT_S"       
                       
        else:
            raise NotImplementedError
        #########################################################################
        #                DECODING AND POSPROCESS
        #########################################################################
        
        if args.unary:
            if not os.path.exists(args.output_unary):
                with codecs.open(args.output_unary,"w") as f:
                    for j,sentence in enumerate(sentences):
                      
                        for (word,postag), retag in zip(sentence,preds[j]):
                            f.write("\t".join([word,postag,retag])+"\n")
                        f.write("\n")
            else:
                raise ValueError("File already exist:", args.output_unary)
            exit()


        parenthesized_trees = sequence_to_parenthesis(new_sentences,preds)
        final_time = time.time()
        tmpfile.write("\n".join(parenthesized_trees)+"\n")
        os.system(" ".join([args.evalb,args.gold, tmpfile.name]))
        gold_labels = [e[2] for e in flat_list(gold_samples)]
 
        if args.retagger:
            enriched_preds = get_enriched_labels_for_retagger(preds, unary_preds)
            flat_preds = flat_list(enriched_preds)
        else:
            flat_preds = flat_list(preds)
            


        print "Accuracy",round(accuracy_score(gold_labels, flat_preds),4) 
        total_time = final_time - init_time
        print "Total time:", round(total_time,4)
        print "Sents/s",round(len(gold_samples) /  (total_time),2)

        
        
    #########################################################
    #
    #                    TRAINING PHASE                     #
    #
    #########################################################    
        
    elif args.status.lower() == STATUS_TRAIN:
        
        # For reproducibility, if wanted
        os.environ['PYTHONHASHSEED'] = '17'
        np.random.seed(17)
        rn.seed(17)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        from keras import backend as K
        tf.set_random_seed(17)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
    
    
        
        ###################################################################
        #            TRAINING AN EMBEDDED PERCEPTRON
        ###################################################################
        if args.baseline.lower() == "emlp":
            hidden_size = 100
            batch = 8
            context_len = 1+args.prev_context+args.next_context
            with codecs.open(args.test, encoding="utf-8") as f_dev:
                dev_samples = [[ tuple(l.split("\t")) for l in sentence.split("\n")] for sentence in f_dev.read().split("\n\n")
                                if sentence != ""]     

            with codecs.open(args.train, encoding="utf-8") as f_train:
                train_samples = [[ tuple(l.split("\t")) for l in sentence.split("\n")] for sentence in f_train.read().split("\n\n")
                                 if sentence != ""]       
                
            vocab = set([])
            postags = set([])
            labels = set([])
            for g in train_samples:
                for word,postag,label in g:
                    vocab.add(word)
                    postags.add(postag)
                    labels.add(label)
                    
            all_labels = labels
            for g in dev_samples:
                for _,_,label in g:
                    all_labels.add(label)        

            emlp_tagger = EmbeddedPerceptronTagger(hidden_size, vocab, postags, len(all_labels), context_len = context_len)
            X_train, X_tags_train, y_train = emlp_tagger.transform(train_samples, args.prev_context, args.next_context)
            X_dev, X_tags_dev, y_dev = emlp_tagger.transform(dev_samples, args.prev_context, args.next_context)  

            label_encoder = LabelEncoder()
            label_encoder.fit(y_train + y_dev)   
            y_train = label_encoder.transform(y_train)
            y_dev = label_encoder.transform(y_dev)            
            X_train = np.array(X_train)
            X_tags_train = np.array(X_tags_train)
            X_dev = np.array(X_dev)
            X_tags_dev = np.array(X_tags_dev)
            
            with codecs.open(args.model+".emlp.features","wb") as f:
                pickle.dump((vocab,postags, all_labels, hidden_size, args.prev_context, args.next_context),f)

            with codecs.open(args.model+".emlp.labels","wb") as f:
                pickle.dump(label_encoder,f)
                        
            checkpoint = keras.callbacks.ModelCheckpoint(args.model+".emlp.hdf5", save_best_only=True)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')

            emlp_tagger.model.fit_generator(emlp_tagger.samples(X_train,X_tags_train,y_train, batch, label_encoder),
                                 validation_data=emlp_tagger.samples(X_dev,X_tags_dev,y_dev,batch, label_encoder),
                                 steps_per_epoch=(X_train.shape[0]/batch)+1, epochs=30,  verbose=1,
                                 validation_steps=(X_dev.shape[0]/batch)+1,
                                 callbacks=[checkpoint,early_stopping])

            print emlp_tagger.model.evaluate_generator(emlp_tagger.samples(X_dev,X_tags_dev,y_dev,batch, label_encoder), 
                                                       steps= (X_dev.shape[0]/batch)+1)  
            
        
        ###################################################################
        #                TRAINING A DISCRETE MLP
        ###################################################################
        elif args.baseline.lower() == "mlp":

            hidden_size = 100
            batch = 8

            with codecs.open(args.test, encoding="utf-8") as f_dev:
                dev_samples = [[ tuple(l.split("\t")) for l in sentence.split("\n")] for sentence in f_dev.read().split("\n\n")
                                if sentence != ""]     

            with codecs.open(args.train, encoding="utf-8") as f_train:
                train_samples = [[ tuple(l.split("\t")) for l in sentence.split("\n")] for sentence in f_train.read().split("\n\n")
                                 if sentence != ""]
                
            print "Len dev samples", len(dev_samples)
            print "Len train amples", len(train_samples)
            X_train, y_train = PerceptronTagger.builder().transform(train_samples, args.prev_context, args.next_context)
            X_dev, y_dev = PerceptronTagger.builder().transform(dev_samples, args.prev_context, args.next_context)
            
            # Fit our DictVectorizer with our set of features
            dict_vectorizer = DictVectorizer(sparse=True)
            dict_vectorizer.fit(X_train + X_dev)       
            X_train = dict_vectorizer.transform(X_train)
            X_dev = dict_vectorizer.transform(X_dev)    
            label_encoder = LabelEncoder()
            label_encoder.fit(y_train + y_dev)   
            y_train = label_encoder.transform(y_train)
            y_dev = label_encoder.transform(y_dev)
            y_train = np_utils.to_categorical(y_train,num_classes=len(label_encoder.classes_))
            y_dev = np_utils.to_categorical(y_dev,num_classes=len(label_encoder.classes_))
            
            with codecs.open(args.model+".mlp.features","wb") as f:
                pickle.dump((dict_vectorizer, hidden_size, args.prev_context, args.next_context),f)

            with codecs.open(args.model+".mlp.labels","wb") as f:
                pickle.dump(label_encoder,f)
                        
            checkpoint = keras.callbacks.ModelCheckpoint(args.model+".mlp.hdf5", save_best_only=True)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
            mlp_tagger = PerceptronTagger(X_train.shape[1],hidden_size, y_train.shape[1])
            mlp_tagger.model.fit_generator(mlp_tagger.samples(X_train,y_train, batch),
                                 validation_data=mlp_tagger.samples(X_dev,y_dev,batch),
                                 steps_per_epoch=(X_train.shape[0]/batch)+1, epochs=30,  verbose=1,
                                 validation_steps=(X_dev.shape[0]/batch)+1,
                                 callbacks=[checkpoint,early_stopping])

            print mlp_tagger.model.evaluate_generator(mlp_tagger.samples(X_dev,y_dev,batch), steps= (X_dev.shape[0]/batch)+1)

                
        ###################################################################
        #        TRAINING A CONDITIONAL RANDOM FIELDS MODEL
        ###################################################################
        elif args.baseline.lower() == "crf":
            
            crf = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=20,
                all_possible_transitions=False,
                model_filename=args.model+".crf",
            )

            with codecs.open(args.test, encoding="utf-8") as f_dev:
                dev_samples = [[l.split("\t") for l in sentence.split("\n")] for sentence in f_dev.read().split("\n\n")
                                if sentence != ""]     

            with codecs.open(args.train, encoding="utf-8") as f_train:
                train_samples = [[l.split("\t") for l in sentence.split("\n")] for sentence in f_train.read().split("\n\n")
                                 if sentence != ""]
            
            X_train = [sent2features(s,args.prev_context, args.next_context) for s in train_samples]
            y_train = [sent2labels(s) for s in train_samples]
            X_dev = [sent2features(s,args.prev_context, args.next_context) for s in dev_samples]
            y_dev = [sent2labels(s) for s in dev_samples]
            crf.fit(X_train, y_train)      
            y_pred = crf.predict(X_dev)
            print "F-score",flat_f1_score(y_dev, y_pred, average='weighted') 
            print "Accuracy:", crf.score(X_dev, y_dev)  
            with codecs.open(args.model+".crf.pickle","wb") as f:
                pickle.dump((crf, args.prev_context, args.next_context), f)
            
        else:
            raise NotImplementedError
                
    else:
        raise NotImplementedError
    
    