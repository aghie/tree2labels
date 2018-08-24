# Constituent Parsing as Sequence Labeling

This repository contains the source code from the paper "Constituent Parsing as Sequence Labeling". 

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

**Additional resources** You also might need to download the [pretrained models](http://grupolys.org/software/tree2labels-emnlp2018-resources/models-EMNLP2018.zip) and/or the [pretrained word embeddings](http://grupolys.org/software/tree2labels-emnlp2018-resources/embeddings-EMNLP2018.zip) used in this work.

The script **`install.sh`** automatically installs the mentioned packages, assuming that you have previously created and activated your virtualenv (tested on Ubuntu 16.04, 64 bits). It also downloads the pretrained models and the pretrained word embeddings used in this work.

## Transforming a constituent treebank into a sequence of labels

`dataset.py` receives as input the splits of a constituent treebank (each of them in one file, and each sample of the file represented in a one-line format) and transforms it into a sequence of labels, one per word, in a TSV format.
```
python dataset.py --train $PATH_TRAINSET --dev $PATH_DEVSET --test $PATH_TESTSET --output $PATH_OUTPUT_DIR --treebank $ANAME [--os] [--root_label] [--encode_unaries]
```
- `--train` refers the path to the parenthesized training set
- `--dev` refers the path to the parenthesized dev set
- `--test` refers the path to the parenthesized test set
- `--treebank` indicates the name that you want to give to your treebank.
- `--os` adds both dummy beginning- and end-of-sentence tokens. 
- `--root_label` uses a special label ROOT instead of an integer number for those words that only have in common the root of the sentence.
- `--encode_unaries` will encode leaf unary chains (check the paper) as a part of the label (see the case of the token Mary, for example). The output will be three files `$ANAME-train.seq_lu`, `$ANAME-dev.seq_lu` and `$ANAME-test.seq_lu` located at `$PATH_OUTPUT_DIR`

> NOTE: The current version uses a computation trick where sentences of length one with leaf unary chains, are encoded as ROOT_LEAF-UNARY-CHAIN instead as NONE_LEAF-UNARY-CHAIN if `--root_label` is activated.

The input to `dataset.py` must be a raw file where each parenthesized tree is represented in a one-line format. For example, given the file:
> ```
> (S (NP (NNP Mary)) (VP (VBD ate) (NP (DT an) (NN apple))))
> (S (NP (DT The) (NN boy)) (VP (VBZ is) (ADJP (JJ nice))))
> ```

the output will look like this (this example uses the `--os`, `--root_label` and `--encode_unaries` options).

> ```
> -BOS-   -BOS-     -BOS-
>  Mary    NNP      ROOT_S_NP
>  ate     VBD      1_VP
>  an      DT       1_NP
>  apple   NN       NONE
> -EOS-   -EOS-     -EOS-
> 
> -BOS-   -BOS-     -BOS-
>  The     DET      2_NP
>  boy     NN       ROOT_S
>  is      VBZ      1_VP
>  nice    JJ       NONE_ADJP
> -EOS-   -EOS-    -EOS-


`dataset.py` allows to create a different version of the sequential dataset where the leaf unary chains are not encoded as a part of the label. To do this, we only need to remove the `--encode_unaries` option from the previous command. The output will be now stored in two separated files with the extensions `.seq` and `.lu`. The `.lu` file maps each (word,postag) to a leaf unary chain (if any) meanwhile the `.seq` file encodes is similar to the `.seq_lu` file, but without encoding the leaf unary chains as a part of the label. Therefore, two classiffiers will be needed to properly solve the task.

To address constituent parsing according to this retagging strategy, in our work we first trained a sequential model that learns to identify leaf unary chains (trained on the `.lu` files). We then run that model on the same files (see next section) and merge the output file with the postags of the corresponding `.seq` files. This can be done with the script `dataset_retagged.py`, obtaining as an output a file that we will be naming with the extension `.seq_r`. 

```
python dataset_retagged.py --dataset $PATH_SPLIT_SEQ --predicted_lu $PATH_PREDICTED_LU_OUTPUT --output $PATH_SAVE_OUTPUT.seq_r
``` 
- `--dataset`refers the path to a split in .seq format
- `--predicted_lu` refers the path with the predictions of the leaf unary chains by a given model, for the same split (make sure this file is in a TSV format)
- `--output` refers the path where to store the merged file (the `.seq_r`file)

The `.seq_r`will be used to train the second sequential model (the one used to predict *only* the common ancestors and the lowest common constituent between w_t and w_(t+1), but not any leaf unary chain).

## Executing and evaluating the pre-trained models

Download the pre-trained models [here](http://grupolys.org/software/tree2labels-emnlp2018-resources/models-EMNLP2018.zip).

We include pre-trained models based on three baselines: (1) Conditional Random Fields (CRF) (2) a one-hot vector multilayered perceptron (MLP) and (3) an MLP that uses word and postag embeddings as inputs (EMLP). We also (4) release more accurate models based on [NCRFpp++](https://github.com/jiesutd/NCRFpp), a recent neural sequence labeling framework by Yang & Zhang (2018).

**Baselines**

```
taskset --cpu-list 1 python baselines.py --test $PATH_TEST.seq_lu  --gold $PATH_PARENTHESIZED_TEST_SET --model $PATH_MODEL --baseline (crf|mlp|emlp) --status test --gpu (True|False) --output_decode $PATH_SAVE_OUTPUT [--retagger] --evalb $EVALB
```
- `--test` refers the path to the input file (use the `.seq_lu` file).
- `--gold` refers the path to the file with the parenthesized trees
- `--model` refers the path to the model and its name to recover its different components.
- `--baseline` the type of the model.
- `--gpu` True or False to indicate whether to use GPU or CPU.
- `--output_decode` refers the path to store the output.
- `--retagger` is used when the output is obtained by an architecture that first uses a model trained on the `.lu` dataset (to predict the leaf unary changes) and then that output is merged with the original postags and fed to a second model trained on the `.seq_r` dataset.
- `--evalb` refers the path to the official evalb script.

The scripts `scripts/run_baselines.sh` and `scripts/run_baselines_ch.sh` show how to run a number of baselines, trained on the PTB and CTB treebanks, using `baselines.py`

> NOTE: You might want to only execute a model to detect leaf unary chains (those trained on the `.lu` dataset) and save its output to later create your own `.seq_r` file. To do so, you can reuse the same script as follows:
```
 taskset --cpu-list 1 python baselines.py --test $PATH_TEST.seq_lu  --gold  $PATH_PARENTHESIZED_TEST_SET --model $PATH_MODEL --baseline (crf|mlp|emlp) --status test --gpu (True|False) --unary --output_unary $PATH_SAVE_OUTPUT 
```
> where `--unary` indicates that we are executing a model that can only detect leaf unary chains and `--output_unary` is the path to save the output of this model.


**NCRFpp++**: 

```
taskset --cpu-list 1 python run_ncrfpp.py --test $PATH_TEST.seq_lu  --gold $PATH_PARENTHESIZED_TEST_SET --model $PATH_MODEL --status test --gpu $USE_GPU (True|False) --output $PATH_SAVE_OUTPUT [--retagger] --evalb $EVALB --ncrfpp $NCRFPP
``` 

- `--test` refers the path to the input file (`.seq_lu` format).
- `--gold` refers the path to the file with the parenthesized trees.
- `--model` refers the path to the model and its name to recover its different components.
- `--gpu` True or False to indicate whether to use GPU or CPU.
- `--output` refers the path to store the output.
- `--retagger` is used when the output is obtained by an architecture that first uses a model trained on the `.lu` dataset (to predict the leaf unary changes) and the that output is merged with the original postags and fed to a second model trained on the `.seq_r` dataset.
- `--evalb` refers the path to the official evalb script.
- `--ncrfpp` refers the path to the NCRFpp source code folder.

The scripts `scripts/run_ncrfpp.sh` and `scripts/run_ncrfpp_ch.sh` show how to run a number of baselines, , trained on the PTB and CTB treebanks, using `run_ncrfpp.py`.

> NOTE: Again, you might want to execute the NCRFpp model to detect unary chains. To do so, you have to execute the following command (check a decode-configuration-file as example and update the paths accordingly).
```
python NCRFpp/main.py --config resources/ncrfpp_config/decode_config/an-unary-decode-configuration-file
```

## Training your own models

You can use any sequence labeling approach to train your model, in a similar way that you would address other NLP tasks, such as PoS tagging, NER or chunking.

**Baselines:** To train a CRF/MLP/EMLP baseline like the ones used in this work:
```
python baselines/baselines.py --train $PATH_TRAINSET  --test $PATH_DEVSET --model $PATH_MODEL --baseline (crf|mlp|emlp) --status train --next_context SIZE_NEXT_CONTEXT --prev_context SIZE_PREV_CONTEXT
```
For example, if you want to train an MLP with word and postag embeddings that analyzes a snnipet containing the two previous and next words:
```
python baselines/baselines.py --train $PATH_TRAINSET  --test $PATH_DEVSET --model /tmp/my-model --baseline emlp --status train --next_context 2 --prev_context 2
```

> NOTE: `$PATH_TRAINSET` and `$PATH_DEVSET` will be `.seq_lu`, `.lu` or `.seq_r` files, depending on what classifier you want to train. 

**NCRFpp++:** The version of the system used in this work, is attached as a git submodule. Read the [NCRFPP doc](https://github.com/jiesutd/NCRFpp) for a more detailed info of the system itself. An example of a configuration file used to train the different models can be found in `resources/ncrfpp_config/train_enriched`. To train the model simply run the `main.py` file located at the NCRFpp source code folder:

```
python main.py --config resources/NCRF_config/train_enriched/a-training-configuration-file
```

## References

Gómez-Rodríguez, Carlos and Vilares, David. "Constituent Parsing as Sequence Labeling". To appear in EMNLP 2018. 

## Contact
If you have any suggestion, inquiry or bug to report, please contact david.vilares@udc.es
