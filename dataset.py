'''
It receives as input the splits of a constituent treebank (each of them in one file, 
and each sample of the file represented in a one-line format) and transforms it into
a sequence of labels, one per word, in a TSV format.
'''

from argparse import ArgumentParser
from tree import SeqTree, RelativeLevelTreeEncoder
from nltk.corpus import treebank

import nltk
import argparse
import os
import codecs
import utils
import random


BOS = "-BOS-"
EOS = "-EOS-"
EMPTY = "-EMPTY-"



def get_feats_dict(sequences, feats_dict):
    
    idx = len(feats_dict)+1
    for sequence in sequences:
        for token in sequence:
            #The first element seems to be a summary of the coarse-grained tag plus the lemma
            if "##" in token[1]:
                
                for feat in token[1].split("##")[1].split("|"):
                    feat_split = feat.split("=")
                
                    if not feat_split[0] in feats_dict:
            
                        feats_dict[feat_split[0]] = idx
                        idx+=1


"""
Intended for morphologically rich corpora (e.g. SPRML)
"""
#TODO: Needed to set a NONE label if a FEAt is not available for a word 
def feats_from_tag(tag, tag_split_symbol, feats_dict):
    
    feats = ["-"]*(len(feats_dict)+1)
    

    feats[0] = tag.split("##")[0]
    
    if "##" in tag:
    
        for feat in tag.split("##")[1].split(tag_split_symbol):
            
            feat_split = feat.split("=")
            feats[feats_dict[feat_split[0]]] = feat

    return feats

"""
Transforms a constituent treebank (parenthesized trees, one line format) into a sequence labeling format
"""
def transform_split(path, binarized,dummy_tokens, root_label,encode_unary_leaf, abs_top,
                    abs_neg_gap, join_char,split_char):
           
    with codecs.open(path,encoding="utf-8") as f:
        trees = f.readlines()
    
    sequences = []
    sequences_for_leaf_unaries = []
    for tree in trees:
  
        tree = SeqTree.fromstring(tree, remove_empty_top_bracketing=True)
        tree.set_encoding(RelativeLevelTreeEncoder(join_char=join_char, split_char=split_char))
        words = tree.leaves()

        tags = [s.label() for s in tree.subtrees(lambda t: t.height() == 2)]
            
        tree.collapse_unary(collapsePOS=True, collapseRoot=True, joinChar=join_char)    
        unary_sequence =  [s.label() for s in tree.subtrees(lambda t: t.height() == 2)] 

        
        gold = [(w,t,g) for w,t,g in zip(words, tags, tree.to_maxincommon_sequence(is_binary=binarized,
                                                                                   #dummy_tokens=dummy_tokens,
                                                                                   root_label=root_label,
                                                                                   encode_unary_leaf=encode_unary_leaf,
                                                                                   abs_top=abs_top,
                                                                                   abs_neg_gap=abs_neg_gap))]
        if dummy_tokens:
            gold.insert(0, (BOS, BOS, BOS) )
            gold.append( (EOS, EOS, EOS) )
        sequences.append(gold)
        
        gold_unary_leaves = []
        gold_unary_leaves = [(BOS, BOS, BOS) ]
        gold_unary_leaves.extend( [(w,t, _set_tag_for_leaf_unary_chain(unary, join_char=join_char) ) for w,t,unary in zip(words, tags, unary_sequence)] )
        gold_unary_leaves.append((EOS, EOS, EOS))
        sequences_for_leaf_unaries.append(gold_unary_leaves)

            
    return sequences, sequences_for_leaf_unaries


def _set_tag_for_leaf_unary_chain(leaf_unary_chain, join_char="+"):

    if join_char in leaf_unary_chain:
        return join_char.join(leaf_unary_chain.split(join_char)[:-1]) #[:-1] not to take the PoS tag
    else:
        return EMPTY    



def write_linearized_trees(path_dest, sequences, feats_dict):
    
    with codecs.open(path_dest,"w",encoding="utf-8") as f:
        for sentence in sequences:
             
            for word, postag,gold in sentence:
                if (feats_dict == {}):
                    f.write(u"\t".join([word,unicode(postag),unicode(gold)])+u"\n")
                else:
                    feats = u"\t".join(feats_from_tag(unicode(postag), "|", feats_dict))
                    f.write(u"\t".join([word,feats,unicode(gold)])+u"\n")
            f.write(u"\n")    


if __name__ == '__main__':
    
        
    parser = ArgumentParser()
    parser.add_argument("--train", dest="train", help="Path to the parenthesized training file",default=None, required=True)
    parser.add_argument("--dev", dest="dev", help="Path to the parenthesized development file",default=None, required=True)
    parser.add_argument("--test", dest="test", help="Path to the parenthesized test file",default=None, required=True)
    parser.add_argument("--treebank", dest="treebank", help = "Name of the treebank", required=True)
    
    parser.add_argument("--output", dest="output", 
                        help="Path to the output directory to store the dataset", default=None, required=True)
    parser.add_argument("--encode_unaries", action="store_true", dest="encode_unaries", help="Activate this option to encode the leaf unary chains as a part of the label", required=True)
    parser.add_argument("--os", action="store_true",help="Activate this option to add both a dummy beggining- (-BOS-) and an end-of-sentence (-EOS-) token to every sentence")
    parser.add_argument("--root_label", action="store_true", help="Activate this option to add a simplified root label to the nodes that are directly mapped to the root of the constituent tree",
                        default=False)
    parser.add_argument("--abs_top", dest="abs_top", type=int,
                        help="To indicate which levels should be stored in absolute scale. This feature also depends on the value assigned to --abs_neg_gap",
                        default=None)
    parser.add_argument("--abs_neg_gap", type=int, 
                        dest="abs_neg_gap",
                        default=None,
                        help="To indicate that only relative levels that encode a gap < -abs_neg_gap should be codified in an absolute scale")
    parser.add_argument("--join_char",type=str,dest="join_char",
                        default="~",
                        help="Symbol used to to collapse branches")
    parser.add_argument("--split_char",type=str,dest="split_char",
                        default="@",
                        help="Symbol used to to collapse branches")
    parser.add_argument("--split_tags", action="store_true",
                        help="To create various features from the tag")
    parser.add_argument("--split_tag_symbol", type=str, default="|")
    #TODO: The binarized options was still not been tested.
    parser.add_argument("--binarized", action="store_true", help="Activate this options if you first want to binarize the constituent trees [NOT TESTED AT THE MOMENT]", default=False)
    
    args = parser.parse_args()
    
    transform = transform_split
    
    if args.encode_unaries:
        ext = "seq_lu"
    else:
        ext = "seq"
        ext_unary = "lu"
        
        
    train_sequences, train_leaf_unary_chains = transform(args.train, args.binarized, args.os, args.root_label, 
                                                         args.encode_unaries, args.abs_top, args.abs_neg_gap,
                                                         args.join_char, args.split_char) 
    
    dev_sequences, dev_leaf_unary_chains = transform(args.dev, args.binarized, args.os, args.root_label, 
                                                     args.encode_unaries, args.abs_top, args.abs_neg_gap,
                                                     args.join_char, args.split_char) 
    
    feats_dict = {}
    if args.split_tags:
        get_feats_dict(train_sequences, feats_dict)
        get_feats_dict(dev_sequences, feats_dict)

    
    write_linearized_trees("/".join([args.output, args.treebank+"-train."+ext]), train_sequences,
                           feats_dict)
    
    write_linearized_trees("/".join([args.output, args.treebank+"-dev."+ext]), dev_sequences, 
                           feats_dict)
              
    test_sequences, test_leaf_unary_chains = transform(args.test, args.binarized, args.os, args.root_label, 
                                                       args.encode_unaries, args.abs_top, args.abs_neg_gap,
                                                       args.join_char, args.split_char) 
    
    write_linearized_trees("/".join([args.output, args.treebank+"-test."+ext]), test_sequences, 
                           feats_dict)
    

    
