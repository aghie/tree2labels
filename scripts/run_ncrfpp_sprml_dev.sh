#Update this path to your virtual environment
#HOME=/home/david.vilares/
#source $HOME/env/tree2labels/bin/activate

HOME_NCRFpp=../NCRFpp/
TEST_NAME="dev"

USE_GPU=True
EVALB=../EVAL_SPRML/evalb_spmrl2013.final/evalb_spmrl
EVALB_PARAM=../EVAL_SPRML/evalb_spmrl2013.final/spmrl.prm
OUTPUT=../outputs_dev/
#MODELS=/home/david.vilares/Escritorio/CPH/NCRFpp_multitask1.6/parsing_models/
MODELS=../parsing_models/
NCRFPP=$HOME_NCRFpp
LOGS=../log_dev_ptb/
#MULTITASK=True


: '

###############################################
#						   BASQUE MODELS
###############################################


INPUT=../sample_data/cp_datasets/basque/basque-$TEST_NAME.seq_lu
PARENTHESIZED_TEST_PATH=../sample_data/cp_datasets/BASQUE_pred_tags/$TEST_NAME.Basque.pred.ptb

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/basque/basque.ch.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/basque.ch.emnlp2018.f.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/basque.ch.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/basque/basque.ch.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/basque.ch.emnlp2018.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/basque.ch.emnlp2018.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/basque/basque.ch.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/basque.ch.multitask.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/basque.ch.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/basque/basque.ch.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/basque.ch.multitask.3R.-2.dis.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/basque.ch.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/basque/basque.ch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/basque.ch.multitask.3R.-2.pre_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/basque.ch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/basque/basque.ch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/basque.ch.multitask.3R.-2.nex_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/basque.ch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/basque/basque.ch.RL.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/basque.ch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/basque.ch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/basque/basque.noch.RL.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/basque.noch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/basque.noch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1


################################################
#								FRENCH MODELS
################################################

INPUT=../sample_data/cp_datasets/french/french-$TEST_NAME.seq_lu
PARENTHESIZED_TEST_PATH=../sample_data/cp_datasets/FRENCH_pred_tags/$TEST_NAME.French.pred.ptb

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/french/french.ch.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/french.ch.emnlp2018.f.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/french.ch.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/french/french.ch.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/french.ch.emnlp2018.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/french.ch.emnlp2018.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/french/french.ch.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/french.ch.multitask.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/french.ch.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/french/french.ch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/french.ch.multitask.3R.-2.pre_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/french.ch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/french/french.ch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/french.ch.multitask.3R.-2.nex_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/french.ch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/french/french.ch.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/french.ch.multitask.3R.-2.dis.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/french.ch.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/french/french.ch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/french.ch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/french.ch.RL.multitask.3R.-2.dis$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/french/french.noch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/french.noch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/french.noch.RL.multitask.3R.-2.dis$TEST_NAME.$USE_GPU.log 2>&1



################################################
#								GERMAN MODELS
################################################

INPUT=../sample_data/cp_datasets/german/german-$TEST_NAME.seq_lu
PARENTHESIZED_TEST_PATH=../sample_data/cp_datasets/GERMAN_pred_tags/$TEST_NAME.German.pred.ptb



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/german/german.ch.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/german.ch.emnlp2018.f.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/german.ch.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/german/german.ch.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/german.ch.emnlp2018.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/german.ch.emnlp2018.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/german/german.ch.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/german.ch.multitask.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/german.ch.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/german/german.ch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/german.ch.multitask.3R.-2.pre_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP  > $LOGS/german.ch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/german/german.ch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/german.ch.multitask.3R.-2.nex_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/german.ch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/german/german.ch.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/german.ch.multitask.3R.-2.dis.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/german.ch.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/german/german.ch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/german.ch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/german.ch.RL.multitask.3R.-2.dis,$TEST_NAME.$USE_GPU.log 2>&1



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/german/german.noch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/german.noch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/german.noch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1

'

################################################
#								HEBREW MODELS
################################################

INPUT=../sample_data/cp_datasets/hebrew/hebrew-$TEST_NAME.seq_lu
PARENTHESIZED_TEST_PATH=../sample_data/cp_datasets/HEBREW_pred_tags/$TEST_NAME.Hebrew.pred.ptb




taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hebrew/hebrew.ch.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hebrew.ch.emnlp2018.f.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/hebrew.ch.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hebrew/hebrew.ch.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hebrew.ch.emnlp2018.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP  > $LOGS/hebrew.ch.emnlp2018.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hebrew/hebrew.ch.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hebrew.ch.multitask.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hebrew.ch.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hebrew/hebrew.ch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hebrew.ch.multitask.3R.-2.pre_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hebrew.ch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hebrew/hebrew.ch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hebrew.ch.multitask.3R.-2.nex_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hebrew.ch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hebrew/hebrew.ch.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hebrew.ch.multitask.3R.-2.dis.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hebrew.ch.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hebrew/hebrew.ch.RL.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hebrew.ch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hebrew.ch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hebrew/hebrew.noch.RL.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hebrew.noch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hebrew.noch.RL.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1


: '

################################################
#								HUNGARIAN MODELS
################################################

INPUT=../sample_data/cp_datasets/hungarian/hungarian-$TEST_NAME.seq_lu
PARENTHESIZED_TEST_PATH=../sample_data/cp_datasets/HUNGARIAN_pred_tags/$TEST_NAME.Hungarian.pred.ptb


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hungarian/hungarian.ch.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hungarian.ch.emnlp2018.f.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/hungarian.ch.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hungarian/hungarian.ch.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hungarian.ch.emnlp2018.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/hungarian.ch.emnlp2018.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hungarian/hungarian.ch.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hungarian.ch.multitask.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hungarian.ch.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hungarian/hungarian.ch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hungarian.ch.multitask.3R.-2.pre_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hungarian.ch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hungarian/hungarian.ch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hungarian.ch.multitask.3R.-2.nex_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hungarian.ch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hungarian/hungarian.ch.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hungarian.ch.multitask.3R.-2.dis.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hungarian.ch.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hungarian/hungarian.ch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hungarian.ch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hungarian.ch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/hungarian/hungarian.noch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/hungarian.noch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/hungarian.noch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1



################################################
#								KOREAN MODELS
################################################


INPUT=../sample_data/cp_datasets/korean/korean-$TEST_NAME.seq_lu
PARENTHESIZED_TEST_PATH=../sample_data/cp_datasets/KOREAN_pred_tags/$TEST_NAME.Korean.pred.ptb


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/korean/korean.ch.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/korean.ch.emnlp2018.f.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/korean.ch.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/korean/korean.ch.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/korean.ch.emnlp2018.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/korean.ch.emnlp2018.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/korean/korean.ch.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/korean.ch.multitask.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/korean.ch.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/korean/korean.ch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/korean.ch.multitask.3R.-2.pre_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/korean.ch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/korean/korean.ch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/korean.ch.multitask.3R.-2.nex_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/korean.ch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/korean/korean.ch.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/korean.ch.multitask.3R.-2.dis.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/korean.ch.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/korean/korean.ch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/korean.ch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/korean.ch.RL.multitask.3R.-2.dis$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/korean/korean.noch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/korean.noch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/korean.noch.RL.multitask.3R.-2.dis$TEST_NAME.$USE_GPU.log 2>&1



################################################
#								POLISH MODELS
################################################



INPUT=../sample_data/cp_datasets/polish/polish-$TEST_NAME.seq_lu
PARENTHESIZED_TEST_PATH=../sample_data/cp_datasets/POLISH_pred_tags/$TEST_NAME.Polish.pred.ptb



taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/polish/polish.ch.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/polish.ch.emnlp2018.f.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/polish.ch.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/polish/polish.ch.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/polish.ch.emnlp2018.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/polish.ch.emnlp2018.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/polish/polish.ch.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/polish.ch.multitask.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/polish.ch.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/polish/polish.ch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/polish.ch.multitask.3R.-2.pre_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/polish.ch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/polish/polish.ch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/polish.ch.multitask.3R.-2.nex_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/polish.ch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/polish/polish.ch.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/polish.ch.multitask.3R.-2.dis.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/polish.ch.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/polish/polish.ch.RL.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/polish.ch.RL.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/polish.ch.RL.multitask.3R.-2.pre_lev$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/polish/polish.noch.RL.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/polish.noch.RL.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/polish.noch.RL.multitask.3R.-2.pre_lev$TEST_NAME.$USE_GPU.log 2>&1


################################################
#								SWEDISH MODELS
################################################


INPUT=../sample_data/cp_datasets/swedish/swedish-$TEST_NAME.seq_lu
PARENTHESIZED_TEST_PATH=../sample_data/cp_datasets/SWEDISH_pred_tags/$TEST_NAME.swedish.pred.ptb


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/swedish/swedish.ch.emnlp2018.f \
--status test \
--gpu $USE_GPU \
--output $OUTPUT/swedish.ch.emnlp2018.f.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/swedish.ch.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/swedish/swedish.ch.emnlp2018.3R.-2 \
--status test \
--gpu $USE_GPU \
--output $OUTPUT/swedish.ch.emnlp2018.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--ncrfpp $NCRFPP > $LOGS/swedish.ch.emnlp2018.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/swedish/swedish.ch.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/swedish.ch.multitask.3R.-2.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/swedish.ch.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/swedish/swedish.ch.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/swedish.ch.multitask.3R.-2.pre_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/swedish.ch.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/swedish/swedish.ch.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/swedish.ch.multitask.3R.-2.nex_lev.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/swedish.ch.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/swedish/swedish.ch.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/swedish.ch.multitask.3R.-2.dis.$TEST_NAME.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/swedish.ch.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/swedish/swedish.ch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/swedish.ch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/swedish.ch.RL.multitask.3R.-2.dis$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $PARENTHESIZED_TEST_PATH \
--model $MODELS/swedish/swedish.noch.RL.multitask.3R.-2.dis \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/swedish.noch.RL.multitask.3R.-2.dis.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--evalb_param $EVALB_PARAM \
--multitask \
--ncrfpp $NCRFPP > $LOGS/swedish.noch.RL.multitask.3R.-2.dis$TEST_NAME.$USE_GPU.log 2>&1

'
