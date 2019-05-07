import tempfile
import subprocess
import warnings
import os
import codecs

import tempfile
import copy
import sys
import time
from tree import SeqTree, RelativeLevelTreeEncoder
from logging import warning




def rebuild_input_sentence(lines):
    
    if len(lines[0].split("\t")) > 3:
                    
        sentence = [ (l.split("\t")[0],  
                      l.split("\t")[1]+"##"+"|".join(feat for feat in l.split("\t")[2:-1]
                                                     if feat != "-")+"##") 
                                    
                                for l in lines]                    
    else:
        sentence = [tuple(l.split("\t")[0:2]) for l in lines]
    return sentence
    


"""
Returns a labels as a tuple of 3 elements: (level,label,leaf_unary_chain)
"""
def split_label(label, split_char):
    if label in ["-BOS-","-EOS-","NONE"]:
        return (label,"-EMPTY-","-EMPTY-")
    if len(label.split(split_char)) == 2:
        return (label.split(split_char)[0], label.split(split_char)[1],"-EMPTY-")
    
    return tuple(label.split(split_char))



"""
Transforms a list of list into a single list
"""
def flat_list(l):
    flat_l = []
    for sublist in l:
        for item in sublist:
            flat_l.append(item)
    return flat_l


"""
Determines if a stringx can be converted into a string
"""
def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False

"""
Auxiliar function to compute the accuracy in an homogeneous way respect
to the enriched approach and the .seq_lu format
"""
def get_enriched_labels_for_retagger(preds,unary_preds):
 
    #TODO: Update for SPRML corpus
    warning("The model will not work well if + is not an unique joint char for collapsed branches (update for SPRML)")
    new_preds = []
    for zpreds, zunaries in zip(preds, unary_preds):
        aux = []
        for zpred, zunary in zip(zpreds,zunaries):
            if "+" in zunary and zpred not in ["-EOS-","NONE","-BOS-"]:
                
                if zpred == "ROOT":
                    new_zpred = "+".join(zunary.split("+")[:-1])
                else:
                    new_zpred = zpred+"_"+"+".join(zunary.split("+")[:-1])
            else:
                new_zpred = zpred
            aux.append(new_zpred)
        new_preds.append(aux)
    return new_preds


"""
Transforms a list of sentences and predictions (labels) into parenthesized trees
@param sentences: A list of list of (word,postag)
@param labels: A list of list of predictions
@return A list of parenthesized trees
"""
#NEWJOINT
def sequence_to_parenthesis(sentences,labels,join_char="~", split_char="@"):
#def sequence_to_parenthesis(sentences,labels,join_char="+"):
    parenthesized_trees = []  
    relative_encoder = RelativeLevelTreeEncoder(join_char=join_char, split_char=split_char)
    
    f_max_in_common = SeqTree.maxincommon_to_tree
    f_uncollapse = relative_encoder.uncollapse
    
    total_posprocessing_time = 0
    for noutput, output in enumerate(labels):       
        if output != "": #We reached the end-of-file
            init_parenthesized_time = time.time()
            sentence = []
            preds = []
            for ((word,postag), pred) in zip(sentences[noutput][1:-1],output[1:-1]):

                if len(pred.split(split_char))==3: #and "+" in pred.split("_")[2]:
                    sentence.append((word,pred.split(split_char)[2]+join_char+postag))             
                              
                else:
                    sentence.append((word,postag)) 
                        
                preds.append(pred)
            tree = f_max_in_common(preds, sentence, relative_encoder)
                        
            #Removing empty label from root
            if tree.label() == SeqTree.EMPTY_LABEL:
                
                #If a node has more than two children
                #it means that the constituent should have been filled.
                if len(tree) > 1:
                    print "WARNING: ROOT empty node with more than one child"
                else:
                    while (tree.label() == SeqTree.EMPTY_LABEL) and len(tree) == 1:
                        tree = tree[0]

            #Uncollapsing the root. Rare needed
            if join_char in tree.label():
                aux = SeqTree(tree.label().split(join_char)[0],[])
                aux.append(SeqTree(join_char.join(tree.label().split(join_char)[1:]), tree ))
                tree = aux

            tree = f_uncollapse(tree)
            

            total_posprocessing_time+= time.time()-init_parenthesized_time
            #To avoid problems when dumping the parenthesized tree to a file
            aux = tree.pformat(margin=100000000)
            
            if aux.startswith("( ("): #Ad-hoc workarounf for sentences of length 1 in German SPRML
                aux = aux[2:-1]
            parenthesized_trees.append(aux)

    return parenthesized_trees 




                
