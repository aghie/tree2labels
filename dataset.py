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


"""
Transforms a constituent treebank (parenthesized trees, one line format) into a sequence labeling format
"""
def transform_split(path, binarized,dummy_tokens, root_label,encode_unary_leaf):
           
    with codecs.open(path,encoding="utf-8") as f:
        trees = f.readlines()
    
    sequences = []
    sequences_for_leaf_unaries = []
    for tree in trees:
  
        tree = SeqTree.fromstring(tree, remove_empty_top_bracketing=True)
        #tree = SeqTree.fromstring(tree.pformat(), remove_empty_top_bracketing=True)
        tree.set_encoding(RelativeLevelTreeEncoder())
        words = tree.leaves()
        tags = [s.label() for s in tree.subtrees(lambda t: t.height() == 2)]
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)    
        unary_sequence =  [s.label() for s in tree.subtrees(lambda t: t.height() == 2)] 
        
        gold = [(w,t,g) for w,t,g in zip(words, tags, tree.to_maxincommon_sequence(is_binary=binarized,
                                                                                   #dummy_tokens=dummy_tokens,
                                                                                   root_label=root_label,
                                                                                   encode_unary_leaf=encode_unary_leaf))]
        if dummy_tokens:
            gold.insert(0, (BOS, BOS, BOS) )
            gold.append( (EOS, EOS, EOS) )
        sequences.append(gold)
        
        gold_unary_leaves = []
        gold_unary_leaves = [(BOS, BOS, BOS) ]
        gold_unary_leaves.extend( [(w,t, _set_tag_for_leaf_unary_chain(unary) ) for w,t,unary in zip(words, tags, unary_sequence)] )
        gold_unary_leaves.append((EOS, EOS, EOS))
        sequences_for_leaf_unaries.append(gold_unary_leaves)

            
    return sequences, sequences_for_leaf_unaries


def _set_tag_for_leaf_unary_chain(leaf_unary_chain):
    
    if "+" in leaf_unary_chain:
        return "+".join(leaf_unary_chain.split("+")[:-1]) #[:-1] not to take the PoS tag
    else:
        return EMPTY


def write_linearized_trees(path_dest, sequences):
    
    with codecs.open(path_dest,"w",encoding="utf-8") as f:
        for sentence in sequences:
             
            for word, postag,gold in sentence:
                f.write(u"\t".join([word,unicode(postag),unicode(gold)])+u"\n")
            f.write(u"\n")    


if __name__ == '__main__':
    
        
    parser = ArgumentParser()
    parser.add_argument("--train", dest="train", help="Path to the parenthesized training file",default=None, required=True)
    parser.add_argument("--dev", dest="dev", help="Path to the parenthesized development file",default=None, required=True)
    parser.add_argument("--test", dest="test", help="Path to the parenthesized test file",default=None, required=True)
    parser.add_argument("--treebank", dest="treebank", help = "Name of the treebank", required=True)
    
    parser.add_argument("--output", dest="output", 
                        help="Path to the output directory to store the dataset", default=None, required=True)
    parser.add_argument("--encode_unaries", action="store_true", dest="encode_unaries", help="Activate this option to encode the leaf unary chains as a part of the label")
    parser.add_argument("--os", action="store_true",help="Activate this option to add both a dummy beggining- (-BOS-) and an end-of-sentence (-EOS-) token to every sentence")
    parser.add_argument("--root_label", action="store_true", help="Activate this option to add a simplified root label to the nodes that are directly mapped to the root of the constituent tree",
                        default=False)
    #TODO: The binarized options was still not been tested.
    parser.add_argument("--binarized", action="store_true", help="Activate this options if you first want to binarize the constituent trees [NOT TESTED AT THE MOMENT]", default=False)
    
    args = parser.parse_args()
    
    transform = transform_split
    
    if args.encode_unaries:
        ext = "seq_lu"
    else:
        ext = "seq"
        ext_unary = "lu"
        
    train_sequences, train_leaf_unary_chains = transform(args.train, args.binarized, args.os, args.root_label, args.encode_unaries) 
    write_linearized_trees("/".join([args.output, args.treebank+"-train."+ext]), train_sequences)
    if not args.encode_unaries:
        write_linearized_trees("/".join([args.output, args.treebank+"-train."+ext_unary]), train_leaf_unary_chains)

    dev_sequences, dev_leaf_unary_chains = transform(args.dev, args.binarized, args.os, args.root_label, args.encode_unaries) 
    write_linearized_trees("/".join([args.output, args.treebank+"-dev."+ext]), dev_sequences)
    if not args.encode_unaries:
        write_linearized_trees("/".join([args.output, args.treebank+"-dev."+ext_unary]), dev_leaf_unary_chains)
              
    test_sequences, test_leaf_unary_chains = transform(args.test, args.binarized, args.os, args.root_label, args.encode_unaries) 
    write_linearized_trees("/".join([args.output, args.treebank+"-test."+ext]), test_sequences)
    if not args.encode_unaries:
        write_linearized_trees("/".join([args.output, args.treebank+"-test."+ext_unary]), test_leaf_unary_chains)

    