# Constituent Parsing as Sequence Labeling: better, faster, stronger sequence tagging constituent parsers

This repository contains the sources for the paper "Better, Faster, Stronger Sequence Tagging Constituent Parsers", published at NAACL 2019. This is essentially an extension of the paper "Constituent Parsing as Sequence Labeling", where we have included dual encodings, multitask learning, auxiliary tasks and reinforcement learning.

## Prerrequisites

We **assume** and also **highly recommend** that you first **create a virtualenv** (e.g. `virtualenv $HOME/env/tree2labels`), so the requeriments for `tree2labels` do not interfere with other versions of packages that you might need for a different software.

**Software**
- Python 2.7
- NLTK 3.2.5
- numpy 1.14.3
- keras 2.1.0 and tensorflow 1.4.0 with gpu support
- sklearn 0.19.1
- sklearn-crfsuite
- h5py 2.7.1
- torch 0.3.1

**Additional resources** You also might need to download the [pretrained models](http://www.grupolys.org/software/tree2labels-naacl2019-resources/pretrained_models_naacl2019.zip) and/or the [pretrained word embeddings](http://grupolys.org/software/tree2labels-emnlp2018-resources/embeddings-EMNLP2018.zip) used in this work.

The script **`install.sh`** automatically installs the mentioned packages, assuming that you have previously created and activated your virtualenv (tested on Ubuntu 18.04, 64 bits). It also downloads the pretrained models and the pretrained word embeddings used in this work.

## Transforming a constituent treebank into a sequence of labels

`dataset.py` receives as input the splits of a constituent treebank (each of them in one file, and each sample of the file represented in a one-line format) and transforms it into a sequence of labels, one per word, in a TSV format.
```
python dataset.py --train $PATH_TRAINSET --dev $PATH_DEVSET --test $PATH_TESTSET --output $PATH_OUTPUT_DIR --treebank $ANAME --encode_unaries [--os] [--root_label] [--abs_top] [--abs_neg_gap] [--join_char] [--split_char] [--split_tag_symbol]
```
- `--train` refers the path to the parenthesized training set.
- `--dev` refers the path to the parenthesized dev set.
- `--test` refers the path to the parenthesized test set.
- `--treebank` indicates the name that you want to give to your treebank.
- `--os` adds both dummy beginning- and end-of-sentence tokens. 
- `--root_label` uses a special label ROOT instead of an integer number for those words that only have in common the root of the sentence.
- `--encode_unaries` will encode leaf unary chains (check the paper) as a part of the label (see the case of the token Mary, for example). The output will be three files `$ANAME-train.seq_lu`, `$ANAME-dev.seq_lu` and `$ANAME-test.seq_lu` located at `$PATH_OUTPUT_DIR`.
- `--abs_top` to encode the first N top levels of the tree according to a top-down absolute scale. N was set to 3 in our work.
- `--abs_neg_gap` to encode words according to a top-down absolute scale when the difference in the level with respect to the encoded level at the previous work is larger than `abs_neg_gap` (and as long as it's located at one of the `abs_top` levels of the tree). This value was set to 2 in our work. Note that the the scope of this feature is limited by the value of the parameter `--abs_top`.
- `--split_tags` A flag to indicate to split the postags into many features. We recommend to activate this when transforming the SPMRL treebanks, which contain morphologically rich tags that can be split determinalistically (e.g. ADI##lem=joan|AZP=ADI_SIN|ADM=PART|ASP=BURU## in the basque SPMRL treebank).
- `--join_char` The symbol used to collapse unary branches. `~` by default.
- `--split_char` The symbol used to collapse different parts of the same label (singletask version). `@` by default.
- `--split_tag_symbol` The symbol used to split morphologically rich tags (e.g. the provided PoS tags in the SPMRL data). `|` by default.


The input to `dataset.py` must be a raw file where each parenthesized tree is represented in a one-line format. For example, given the file:
> ```
> (S (NP (NNP Mary)) (VP (VBD ate) (NP (DT an) (NN apple))))
> (S (NP (DT The) (NN boy)) (VP (VBZ is) (ADJP (JJ nice))))
> ```

the output will look like this (this example uses the `--os`, `--root_label` and `--encode_unaries` options).

> ```
> -BOS-   -BOS-     -BOS-
>  Mary    NNP      ROOT@S@NP
>  ate     VBD      1@VP
>  an      DT       1@NP
>  apple   NN       NONE
> -EOS-   -EOS-     -EOS-
> 
> -BOS-   -BOS-     -BOS-
>  The     DET      2@NP
>  boy     NN       ROOT@S
>  is      VBZ      1@VP
>  nice    JJ       NONE@ADJP
> -EOS-   -EOS-    -EOS-


In our paper, the command used to obtain the transformed treebanks for PTB and CTB was:
```
python dataset.py --train $PATH_TRAINSET --dev $PATH_DEVSET --test $PATH_TESTSET --output $PATH_OUTPUT_DIR --treebank [PTB|CTB] --encode_unaries --os --abs_top 3 --abs_neg_gap 2 
```

and for the SPMRL treebanks:
```
python dataset.py --train $PATH_TRAINSET --dev $PATH_DEVSET --test $PATH_TESTSET --output $PATH_OUTPUT_DIR --treebank [$A_SPMRL_TREEBANK] --encode_unaries --os --abs_top 3 --abs_neg_gap 2 --split_tags --split_tag_symbol |
```

To create a multitask version fo the corpus we can use the script [encoding2multitask.py](encodings2multitask.py) to obtain an output of the form:

> ```
> -BOS-   -BOS-     -BOS-{}-BOS-{}-BOS-
>  Mary    NNP      ROOT{}S{}NP
>  ate     VBD      1{}VP{}-EMTPY-
>  an      DT       1{}NP{}-EMPTY-
>  apple   NN       NONE{}-EMTPY-{}-EMPTY-
> -EOS-   -EOS-     -EOS-{}-EOS-{}-EOS-
> 
> -BOS-   -BOS-     -BOS-{}-BOS-{}-BOS-
>  The     DET      2{}NP{}-EMTPY-
>  boy     NN       ROOT{}S{}-EMTPY-
>  is      VBZ      1{}VP{}-EMTPY-
>  nice    JJ       NONE{}ADJP{}-EMPTY-
> -EOS-   -EOS-    -EOS-{}-EOS-{}-EOS-


> NOTE: We have removed from the current version the retagging approach, introduced in "Constituent Parsing as Sequence Labeling" at EMNLP 2018. Please, check this [tag](https://github.com/aghie/tree2labels/tree/v1.0) to access to a version of the code that still supports the retagging option.

## Executing and evaluating the pre-trained models

You can download some pretrained models [here](http://www.grupolys.org/software/tree2labels-naacl2019-resources/pretrained_models_naacl2019.zip).

**NCRFpp**: 

```
taskset --cpu-list 1 python run_ncrfpp.py --test $PATH_TEST.seq_lu  --gold $PATH_PARENTHESIZED_TEST_SET --model $PATH_MODEL --status test --gpu $USE_GPU (True|False) --output $PATH_SAVE_OUTPUT  --evalb $EVALB --ncrfpp $NCRFPP [--multitask]
``` 

- `--test` refers the path to the input file (`.seq_lu` format).
- `--gold` refers the path to the file with the parenthesized trees.
- `--model` refers the path to the model and its name to recover its different components.
- `--gpu` True or False to indicate whether to use GPU or CPU.
- `--output` refers the path to store the output.
- `--evalb` refers the path to the official evalb script.
- `--ncrfpp` refers the path to the NCRFpp source code folder.
- `--multitask` A flag to indicate that the model is a MTL model (it is needed to know how to rebuild the label and compute the tree).

See the folder `scripts` to see in detail some examples of how to run these models.

## Training your own models

You can use any sequence labeling approach to train your model, in a similar way that you would address other NLP tasks, such as PoS tagging, NER or chunking.

**NCRFpp:** The *MTL* version of the system used in this work, is attached as a git submodule. Read the [NCRFPP doc](https://github.com/jiesutd/NCRFpp) for a more detailed info of the original system. Some examples for configurations files to train PTB models can be found in `resources/ptb_configs/`. To train the model simply run the `main.py` file located at the NCRFpp source code folder:

```
python main.py --config resources/ptb_configs/a-training-configuration-file
```

#### Fine-tuning

**Auxiliary tasks** can be easily added in the training configuration file. Check out `/resources/ptb_configs/train.ch.multitask.3R.-2.pre_lev.config` and the [MTL NCRFpp](https://github.com/aghie/NCRFpp/blob/naacl-2019/README.md) used in this work for a more detailed explanation.

The hyperparameters for the auxiliary tasks are:

```
main_tasks=3
tasks=4
tasks_weights=1|1|1|0.1
```

where:

- `main_tasks` specifies that the *first* N tasks (corresponding to the *first* N labels) are main tasks (they will be also be computed during testing). 
- `tasks` specifies the number of total tasks, i.e. tasks-main_tasks = M, the number of auxiliary tasks, which won't be computed during testing, they are only used for training).
- `tasks_weights` specifies the weight for the loss obtained for each task. Each task weight is separated by `|`.

For example, if we use as an auxiliary task Named Entity Recognition, together with the MTL setup for constituent parsing, the input format would be something like this:

> ```
> -BOS-   -BOS-     -BOS-{}-BOS-{}-BOS-{}-BOS-
>  Mary    NNP      ROOT{}S{}NP{}PERSON
>  ate     VBD      1{}VP{}-EMTPY-{}OTHER
>  an      DT       1{}NP{}-EMPTY-{}OTHER
>  apple   NN       NONE{}-EMTPY-{}-EMPTY-{}OTHER
> -EOS-   -EOS-     -EOS-{}-EOS-{}-EOS-{}-EOS-
> 
> -BOS-   -BOS-     -BOS-{}-BOS-{}-BOS-{}-BOS-
>  The     DET      2{}NP{}-EMTPY-{}OTHER
>  boy     NN       ROOT{}S{}-EMTPY-{}OTHER
>  is      VBZ      1{}VP{}-EMTPY-{}OTHER
>  nice    JJ       NONE{}ADJP{}-EMPTY-{}OTHER
> -EOS-   -EOS-    -EOS-{}-EOS-{}-EOS-{}-EOS-


**Policy gradient and reinformement learning**

Check `resources/ptb_configs/train.ch.multitask.3R.-2.pre_lev.pg.config` 

As you can see the hyperparameter `status` is set now to `finetune`. We also need to specify the hyperpameters for policy gradient, for example:

```
No_samples=6
pg_variance_reduce=True
variance_reduce_burn_in=999
pg_valsteps=500
entropy_regularisation=True
entropy_reg_coeff=0.01
```

## Acknowledgements

This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150).

## References

- Vilares, David and Abdou, Mostafa and Søgaard, Anders. "Better, Faster, Stronger Sequence Tagging Constituent Parsers". NAACL 2019.
- Gómez-Rodríguez, Carlos and Vilares, David. "Constituent Parsing as Sequence Labeling". EMNLP 2018.


