#!/usr/bin/env python

# (C) 2006, 2008, 2013 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior written
# permission of the copyright holder.
#
# Author: Joachim Wagner
#
# Based on chparsereader.py
#
# Thanks to Djame Seddah for comments and suggestions. (and some additions)
#
# Permission is granted to participants and organisers of the SPMRL 2013 shared
# task to use this code within the shared task and for writing up results and
# further analysis to a maximum of 2 publications per participating team after
# the shared task.
#
# For using this software outside the SPMRL 2013 competition or for anything
# other than post-workshop aftermaths, please watch https://github.com/CNGLdlab
# for a public release and license terms (in progress; may take a few months).
#
# If you use this module in your research, please cite
#
#     Joachim Wagner (2012): Detecting Grammatical Errors with Treebank-
#     Induced, Probabilistic Parsers. PhD Thesis, Dublin City University,
#     Dublin, Ireland. http://doras.dcu.ie/16776/

import math
import string
import sys
import types

ch2term = {
    '-LCB-': '{',        '-RCB-': '}',
    '-LRB-': '(',        '-RRB-': ')',
    '-LSB-': '[',        '-RSB-': ']',
    '\\/':   '/',        '\\|':   '|',
}

opt_labels_to_delete = ['ROOT','TOP','S1','VROOT'] # Djame: this should be read from an evalb parameter file (DELETE_LABEL ROOT et..)
 
# note there's a bug, when there's one more enclosing parameter than the gold, exact match is never incremented
#  evalb got rid of that one ( )
# for now, I had to stripp off that parenthesis  (all baseline results from us on Polish)

def getTree(chrepr, stop_markers = [], index = 0):
    global ch2term
    if not chrepr:
        return None
    if not index:
        if chrepr[0] == '(':
            chrepr = string.rstrip(chrepr)
            tree, index = getTree(chrepr, stop_markers, 1)
            if index < len(chrepr):
                raise ValueError, 'trailing data after end of tree repr'
            return tree
        else:
            raise ValueError, 'left parenthesis expected at index of tree repr'
    category = ''
    while 1:
        try:
            next = chrepr[index]
            index = index + 1
        except IndexError:
            next = ''
        if not next or next == ')':
            sys.stderr.write('*** error in tree data: ')
            sys.stderr.write(`chrepr`)
            sys.stderr.write('***\n')
            raise ValueError, 'no subtree follows category at position %d in tree repr' %(index-1)
        if next == ' ':
            break
        if next == '(':
            # workaround for '((' at start of PTB sentences
            index = index - 1
            break
        category = category + next

    children = []
    terminal = ''
    inParenthesis = 0
    while 1:
        try:
            next = chrepr[index]
            index = index + 1
        except IndexError:
            next = ''
        if not next:
            raise ValueError, 'premature end of string in tree repr'
        if terminal and inParenthesis:
            # normal case: token '712(4)'
            if next == ')':
                inParenthesis = 0
                if chrepr[index:index+2] != ' (':
                    # right context is normal (either rest of
                    # terminal or more closing brackets
                    terminal = terminal + next
                    continue
                # else the ')' cannot have been part
                # of the last terminal as it cannot have
                # a right sister, i.e. ' ('
            else:
                terminal = terminal + next
                continue
        if terminal and next == '(':
            terminal = terminal + next
            inParenthesis = 1
            sys.stderr.write('Warning: parenthesis in token workaround\r')
            continue
        if terminal and next == ' ':  #  in ('(', ' '):
            children.append(terminal)
            terminal = ''
            sys.stderr.write('Warning: found terminal with right sister\n')
            if next == ' ':
                continue
        if next in string.whitespace:
            continue
        if next == '(':
            # recursion
            child, index = getTree(chrepr, stop_markers, index)
            children.append(child)
            continue
        if next == ')':
            if terminal:
                children.append(terminal)
            try:
                next = chrepr[index]
            except IndexError:
                next = ''
            if next == ' ':
                index = index + 1
            height = 0
            width  = 0
            inodes = 1
            terminals = []
            preterm = []
            childindex = 0
            if category == '-NONE-':
                # prune subtree of null element
                children = []
            for child in children:
                if type(child) == types.TupleType:
                    newHeight = child[0][1] + 1
                    if newHeight > height:
                        height = newHeight
                    width  = width  + child[0][2]
                    inodes = inodes + child[0][3]
                    terminals = terminals + child[0][4]
                    preterm = preterm + child[0][5]
                else:
                    width = width + 1
                    for key in ch2term.keys():
                        child = child.replace(key, ch2term[key])
                    terminals.append(child)
                    preterm.append('%d:%s %s' %(childindex, category, child))
                childindex = childindex + 1
            for stop_marker in stop_markers:
                # prune category labels to stop marker
                position = category.find(stop_marker)
                if position >= 0:
                    category = category[:position]
                #DJAME : ici insert code to replace category by dummy if it matches the list of label to skip
                if category in opt_labels_to_delete:
                    #sys.stderr.write("cat: "+category+" replaced with __DUMMY__\n")
                    category = ''

            # finish recursion step
            return (
                (   (category, height, width, inodes, terminals, preterm),
                    children
                ),
                index
            )
        # main loop
        terminal = terminal + next


def getProductions(tree, levels = 1):
    productions = p_getProductions(tree, levels, ())
    return ' '.join(productions)

def p_getProductions(tree, levels, ancestors):
    if not tree:
        return ()
    elif type(tree) == types.StringType:
        return ()
    cat, height, width, inodes, terminals, preterm = tree[0]
    children = tree[1]
    retval = ()
    ancestors = (cat,)+ancestors
    if len(children) == 1 and type(children[0]) == types.StringType:
        # exclude terminal rules
        return retval
    ancestors = ancestors[:levels]
    childcats = map(lambda x: x[0][0], children)
    retval = ('%s->%s' %('^'.join(ancestors), '_'.join(childcats)),)
    for subtree in children:
        productions = p_getProductions(subtree, levels, ancestors)
        retval = retval + productions
    return retval


def getConstituents(tree):
    clist, width = p_getConstituents(tree, 0)
    return ' '.join(clist)

def p_getConstituents(tree, index):
    if not tree:
        return ((), 0)
    elif type(tree) == types.StringType:
        return ((), 1)
    cat, height, width, inodes, terminals, preterm = tree[0]
    retval = ('%s(%d,%d)' %(cat, index+1, index + width),)
    children = tree[1]
    if len(children) == 1 and type(children[0]) == types.StringType:
        # exclude preterms
        return ((), 1)
    for subtree in children:
        constituents, subwidth = p_getConstituents(subtree, index)
        retval = retval + constituents
        index = index + subwidth
    return (retval, width)

def expandConstituents(constituents):
    return frozenset(constituents.split())

def parseval(goldConstituents, testConstituents):
    correct = len(goldConstituents & testConstituents)
    try:
        precision = correct / float(len(testConstituents))
    except ZeroDivisionError:
        precision = 1.0
    try:
        recall = correct / float(len(goldConstituents))
    except ZeroDivisionError:
        recall = 1.0
    try:
        fscore = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        fscore = 0.0
    return precision, recall, fscore


def getLineages(tree):
    cat, height, width, inodes, terminals, preterm = tree[0]
    llist = p_getLineages(tree, 0, [(cat, 0, width-1)])
    return '\n'.join(llist)

def p_path2lineage(index, pathToRoot):
    #return ' '.join(map(repr, pathToRoot))
    tlist = []
    for (cat, spanStart, spanEnd) in pathToRoot[1:]:
        if spanStart is not None and spanStart == index:
            tlist.append('[')
        tlist.append(cat)
        if spanEnd is not None and spanEnd == index:
            tlist.append(']')
    return ' '.join(tlist)

def p_getLineages(tree, index, pathToRoot):
    if not tree or type(tree) == types.StringType:
        return [p_path2lineage(index, pathToRoot)]
    cat, height, width, inodes, terminals, preterm = tree[0]
    children = tree[1]
    numChildren = len(children)
    if numChildren == 1 and type(children[0]) == types.StringType:
        # exclude preterms
        return [p_path2lineage(index, pathToRoot)]
    retval = []
    for daughterIndex in range(numChildren):
        subtree = children[daughterIndex]
        if type(subtree) == types.StringType:
            # semi-preterm or preterm with multiple daughters
            daughterLineages = [p_path2lineage(index, pathToRoot)]
            retval = retval + daughterLineages
            index = index + 1
            continue
        dcat, dheight, dwidth, dinodes, dterminals, dpreterm = subtree[0]
        hspanStart, hspanEnd = None, None
        if numChildren > 1 and daughterIndex == 0:
            hspanEnd = index + dwidth - 1
        elif numChildren > 1 and daughterIndex == numChildren - 1:
            hspanStart = index
        elif numChildren > 1:
            hspanStart = index
            hspanEnd = index + dwidth - 1
        daughterLineages = p_getLineages(
            subtree,
            index,
            [(dcat, hspanStart, hspanEnd)] + pathToRoot
        )
        retval = retval + daughterLineages
        index = index + dwidth
    return retval

def expandLineages(lins):
    retval = []
    for lineage in lins.split('\n'):
        retval.append(tuple(lineage.split()))
    return tuple(retval)

def leafAncestor(goldLineages, testLineages, measure = None, avg = None, costs = None):
    numLineages = len(goldLineages)
    if numLineages != len(testLineages):
        sys.stderr.write('leafAncestor: sentence length missmatch\n')
        return 0.0
    if measure in (None, 'ED', 'EditDistance', 'LevenshteinDistance'):
        measure = p_laLevenshteinDistance
    elif measure == 'LCS':
        measure = p_laLcsDistance
    elif not callable(measure):
        raise ValueError, 'unsupported lineage measure %r' %measure
    sum = 0.0
    if avg == 'minimum':
        sum = 1.0
    for index in range(numLineages):
        goldLineage = goldLineages[index]
        testLineage = testLineages[index]
        if costs:
            score = measure(goldLineage, testLineage, costs[0], costs[1], costs[2])
        else:
            score = measure(goldLineage, testLineage)
        if not avg or avg == 'arithmetric':
            sum = sum + score
        elif avg == 'geometric':
            try:
                sum = sum + math.log(score)
            except:
                sum = '-inf'
        elif avg == 'minimum':
            if score < sum:
                sum = score
    if not avg or avg == 'arithmetric':
        return sum / numLineages
    elif avg == 'geometric':
        try:
            return math.exp(sum/numLineages)
        except:
            return 0.0
    elif avg == 'minimum':
        return sum
    else:
        raise ValueError, 'unsupported lineage avg %r' %avg

def p_laLevenshteinDistance(s, t, costs_match = 0, costs_similar = 1.5, costs_error = 2):
    m = len(s)
    n = len(t)
    if not (m+n):
        return 1.0
    d = {}
    for i in range(m+1):
        d[(i,0)] = i
    for j in range(n+1):
        d[(0,j)] = j
    for i in range(m):
        for j in range(n):
            if s[i] == t[j]:
                costs = costs_match
            elif s and t and s[i][0] == t[j][0]:
                costs = costs_similar
            else:
                costs = costs_error
            d[(i+1, j+1)] = min(
                d[(i,j+1)] + 1,
                d[(i+1,j)] + 1,
                d[(i,j)] + costs
            )
    return 1.0 - d[(m,n)] / float(m+n)

def p_laLcsDistance(x, y):
    """ calculates the length of the longest common subsequence of the
    two arguments with a dynamic programming algorithm """
    # speed up (our data often matches exactly)
    if x == y:
        return 1.0
    if not x or not y:
        return 0.0
    # algorithm from Goodrich and Tamassia 1998 p. 505
    L = {}
    n, m = len(x), len(y)
    for i in range(-1, n):
       L[i, -1] = 0
    for j in range(-1, m):
       L[-1, j] = 0
    for i in range(n):
        for j in range(m):
            if x[i] == y[j]:
                L[i, j] = L[i-1, j-1] + 1
            else:
                L[i, j] = max(L[i-1, j], L[i, j-1])
    commonLength = L[n-1, m-1]
    lenX = len(x)
    lenY = len(y)
    return commonLength / float(min(lenX, lenY))

def tree2chrepr(t):
    if not t:
        return ''
    if type(t) == types.StringType:
        return t
    return '(' + t[0][0] + ' ' + string.join(map(tree2chrepr, t[1])) + ')'

def test():
    s = """(S1 (S (SBAR (IN If) (S (NP (PRP we)) (VP (VBP achieve) (NP (NP (DT a) (NN consensus)) (ADVP (RB here))) (NP (NN tomorrow)) (PP (IN at) (NP (DT the) (NN vote)))))) (, ,) (NP (PRP we)) (VP (MD will) (VP (AUX be) (VP (VBG making) (NP (NN history))))) (. .)))"""
    t = getTree(s)
    print 'getTree', s == tree2chrepr(t)
    # TODO: test more functions


def usage():
    sys.stderr.write('Usage: parse_la.py [options] TEST GOLD\n')
    sys.stderr.write('--add-range   I J  also output stats for sentence length I to J inclusive (0 = no limit; default: 0 0)\n')
    sys.stderr.write('--add-stop-marker S  read category labels up to string S (default: no stop markers / keep all characters)\n')
    sys.stderr.write('--breakdown        show sentence length, score and average so far for each sentence\n')
    sys.stderr.write('--costs-match   F  costs for a label match    (default: 0.0)\n')
    sys.stderr.write('--costs-similar F  costs for a similar label  (default: 0.5 as in the c code and in S&B\'s NLE article)\n')
    sys.stderr.write('--costs-error   F  costs for a label mismatch (default: 2.0)\n')
    sys.stderr.write('--lcs              use LCS (default: Levenshtein distance)\n')
    sys.stderr.write('--workshop-paper   equivalent to --costs-similar 1.5 (value used in the "Beyond Parseval" workshop paper)\n')
    sys.stderr.write('--show-lineages    show lineages for each token (tab-colon-tab separated)\n')
    sys.stderr.write('--simple           equivalent to --costs-similar 1.0 --costs-error 1.0\n')
    sys.stderr.write('-L                 display results in one line format, stop markers are set to [-,#] \n')
    sys.stderr.write('-K                 display results for sentence of lenght <= 70 \n')

def range_applies(ri, rj, sentence_length):
    if ri and ri > sentence_length:
        return False
    if rj and rj < sentence_length:
        return False
    return True
 
def main():
    opt_breakdown     = False
    opt_costs_match   = 0
    opt_costs_similar = 0.5
    opt_costs_error   = 2
    opt_distance      = 'LevenshteinDistance'
    opt_show_lineages = False
    opt_stop_markers  = []
    opt_compact_view  = 0
    
    ranges = [(0, 0)]
    # check for options
    while len(sys.argv) > 1 and sys.argv[1][:1] == '-':
        if sys.argv[1] == '--':
            del sys.argv[1]
            break
        elif sys.argv[1] == '--add-range':
            temp_i = int(sys.argv[2])
            temp_j = int(sys.argv[3])
            if temp_j and temp_j < temp_i:
                raise ValueError, 'J in range must be either 0 or greater or equal I'
            ranges.append((temp_i, temp_j))
            del sys.argv[1]
            del sys.argv[1]
        elif sys.argv[1] == '--add-stop-marker':
            opt_stop_markers.append(sys.argv[2])
            del sys.argv[1]
        elif sys.argv[1] == '--breakdown':
            opt_breakdown = True
        elif sys.argv[1] == '--costs-match':
            opt_costs_match = float(sys.argv[2])
            del sys.argv[1]
        elif sys.argv[1] == '--costs-similar':
            opt_costs_similar = float(sys.argv[2])
            del sys.argv[1]
        elif sys.argv[1] == '--costs-error':
            opt_costs_error = float(sys.argv[2])
            del sys.argv[1]
        elif sys.argv[1] == '--lcs':
            opt_distance = 'LCS'
        elif sys.argv[1] == '--workshop-paper':
            opt_costs_similar = 1.5
        elif sys.argv[1] == '--show-lineages':
            opt_show_lineages = True
        elif sys.argv[1] == '--simple':
            opt_costs_similar = 1.0
            opt_costs_error   = 1.0
        elif sys.argv[1] == '-L':
            opt_compact_view = 1
            opt_stop_markers.append("-")
            opt_stop_markers.append("#")
        elif sys.argv[1] == '-K':
            temp_i = 0
            temp_j = 70
            ranges.append((temp_i, temp_j))
            opt_stop_markers.append("-")
            opt_stop_markers.append("#")
        elif sys.argv[1] == '-X':
            opt_do_nothing = 1        
        elif sys.argv[1][:3] in ('--h', '-h', '-he'):
            usage()
            sys.exit(0)
        else:
            sys.stderr.write('unknown option %s\n' %`sys.argv[1]`)
            sys.exit(1)
        del sys.argv[1]
    if len(sys.argv) != 3:
        # not exactly 2 files provided on command line
        usage()
        sys.exit(0)
        

    

#   argument order inverted
    gold_name = sys.argv[1]    
    test_name = sys.argv[2]

    test_f= open(test_name,'r')
    gold_f= open(gold_name,'r')
	
    test_trees = test_f.readlines()
    gold_trees = gold_f.readlines()
    sys.stderr.write("Processing "+gold_name+" vs "+test_name+" \n")
    if len(gold_trees) != len(test_trees):
        raise ValueError, 'not same number of trees'

    scores = []
    weighted_scores = []
    nb_errors = []
    exact_match = [] #djame
    total = []
    weight_total = []
    for ri,rj in ranges:
        scores.append([])
        weighted_scores.append([])
        nb_errors.append(0)
        exact_match.append(0)
        total.append(0.0)
        weight_total.append(0.0)

    if opt_breakdown or opt_show_lineages:
        sys.stdout.write('Line')
        if opt_breakdown:
            sys.stdout.write('\tLength\tScore')
            for j,(ri,rj) in enumerate(ranges):
                sys.stdout.write('\tS%d-%d' %(ri,rj))
                sys.stdout.write('\tT%d-%d' %(ri,rj))
        sys.stdout.write('\n')
        if opt_show_lineages:
            sys.stdout.write('\tSeq\t%22s\tScore\tTest-Lineage\t  :  \tGold-Lineage\n' %'Token')
            if opt_breakdown:
                sys.stdout.write('\n')
    for i, test_one_tree in enumerate(test_trees):
        # remove trailing whitespace
        test_one_tree = test_one_tree.rstrip()
        gold_one_tree = gold_trees[i].rstrip()
        if not gold_one_tree:
            raise ValueError, 'empty gold tree'
        # need gold details for both branches and show-lineages below
        gold_L = getTree(gold_one_tree, opt_stop_markers)
        # get leaf-ancestor lineages for gold tree
        gold_i = getLineages(gold_L)   # compact format
        gold_i = expandLineages(gold_i)
        try:
            sentence_length = gold_L[0][2]
        except:
            raise NotImplementedError, 'unexpected tree structure %r' %gold_L
        # if test parse not empty
        if test_one_tree \
        and not test_one_tree.startswith('(null)') \
        and not test_one_tree.startswith('(())') \
        and not test_one_tree.startswith('()') \
        and not test_one_tree.startswith('( )'):
            test_L = getTree(test_one_tree, opt_stop_markers)
            # get leaf-ancestor lineages for test tree
            test_i = getLineages(test_L)   # compact format
            test_i = expandLineages(test_i) # ready to use
            # compute score
            lascore = leafAncestor(
                gold_i, test_i, opt_distance, 'arithmetric',
                (opt_costs_match, opt_costs_similar, opt_costs_error)
            )
        else:
            test_i  = None
            lascore = 0.0
            for j,(ri,rj) in enumerate(ranges):
                if range_applies(ri, rj, sentence_length):
                    nb_errors[j] = nb_errors[j] + 1
        # determine weight (micro vs macro average)
        weight = sentence_length
        # add up scores for each length range
        for j,(ri,rj) in enumerate(ranges):
            if range_applies(ri, rj, sentence_length):
                scores[j].append(lascore)
                total[j] = total[j] + 1
                weighted_scores[j].append(lascore * weight)
                weight_total[j] = weight_total[j] + weight
                if lascore == 1.0:
                    #sys.stderr.write("Exact match")
                    exact_match[j] = exact_match[j] +1 #djame
                    
        if opt_breakdown or opt_show_lineages:
            sys.stdout.write('%d' %(i+1))
            if opt_breakdown:
                sys.stdout.write('\t%d\t%.4f' %(sentence_length, lascore))
                for j,(ri,rj) in enumerate(ranges):
                    if range_applies(ri, rj, sentence_length):
                        try:
                            mean = sum(weighted_scores[j]) / float(weight_total[j])
                        except ZeroDivisionError:
                            mean = -1
                        sys.stdout.write('\t%.4f' %mean)
                        try:
                            mean = sum(scores[j]) / float(total[j])
                        except ZeroDivisionError:
                            mean = -1
                        sys.stdout.write('\t%.4f' %mean)
                    else:
                        sys.stdout.write('\t---\t---')
            sys.stdout.write('\n')
            if opt_show_lineages:
                if opt_breakdown:
                    sys.stdout.write('\n')
                for j,gold_lineage in enumerate(gold_i):
                    glin = ' '.join(gold_lineage)
                    if test_i:
                        test_lineage = test_i[j]
                        tlin = ' '.join(test_lineage)
                        # compute score
                        lascore = leafAncestor(
                            [gold_lineage], [test_lineage], opt_distance, 'arithmetric',
                            (opt_costs_match, opt_costs_similar, opt_costs_error)
                        )
                    else:
                        tlin  = '---'
                        lascore = -1
                    token = gold_L[0][4][j]
                    sys.stdout.write('\t%d\t%22s\t%.4f\t%s\t  :  \t%s\n' %(j+1, token, lascore, tlin, glin))
                if opt_breakdown:
                    sys.stdout.write('\n')
            
    # output summary

    if opt_breakdown or opt_show_lineages:
       sys.stdout.write('\n------------------\n')
    if opt_compact_view:
        template = "Acc_macro: %(weighted_mean).4f\tAcc_micro: %(mean).4f\tUnparsed: %(num_errors)d\tSent.: %(num_sentences)d\tEX: %(exact).2f\tRange: (%(range)s)\tfile: "+test_name+"\n"
    else:    
        template = """
    Total Nb of sentences (%(range)s) =\t%(num_sentences)d
    Total Nb of errors    (%(range)s) =\t%(num_errors)d
    Total LeafAncestor    (%(range)s) =\t%(weighted_mean).4f\t(average of sentence scores)
    Total LeafAncestor    (%(range)s) =\t%(mean).4f\t(average of token scores)
    Exact Match           (%(range)s) =\t%(exact).4f\t(num. of exact parses)
    """
    for j,(ri,rj) in enumerate(ranges):
        # description of range
        if ri and rj:
            range = '%d <= length <= %d' %(ri,rj)
        elif ri:
            range = 'length >= %d' %ri
        elif rj:
            range = 'length<=%d' %rj
        else:
            range = 'all'
        # number of sentences in this range
        num_sentences = len(scores[j])
        # overall score for this range
        try:
            weighted_mean = sum(weighted_scores[j]) / float(weight_total[j])
        except ZeroDivisionError:
            weighted_mean = -1
        try:
            mean = sum(scores[j]) / float(total[j])
        except ZeroDivisionError:
            mean = -1
        # number of errors
        num_errors = nb_errors[j]
        exact = (float(exact_match[j]) / float(num_sentences))*100.0
        # output
        sys.stdout.write(template %locals())

if __name__ == '__main__':
    #test()
    main()

