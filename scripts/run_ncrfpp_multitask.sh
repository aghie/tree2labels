#Update this path to your virtual environment
#source $HOME/env/tree2labels/bin/activate

HOME_NCRFpp=../NCRFpp/

TEST_NAME="test"
INPUT=../sample_data/cp_datasets/ptb/ptb-$TEST_NAME.seq_lu
TEST_PATH=../sample_data/cp_datasets/PTB_pred_tags/$TEST_NAME.trees
USE_GPU=False
EVALB=../EVALB/evalb
OUTPUT=../outputs/
MODELS=../pretrained_models_naacl2019/ptb/
NCRFPP=$HOME_NCRFpp
LOGS=../logs/

################################################
#								BASIC MODELS
################################################

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/emnlp2018.f.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.emnlp2018.f.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/emnlp2018.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/ptb.emnlp2018.f.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


###################################################
# + MULTITASK LEARNING
###################################################

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/ptb.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


###################################################
# + BEST AUX TASK
###################################################

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ptb.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/ptb.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1 


###################################################
# + REINFORCEMENT LEARNING
###################################################

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb.RL.multitask.3R.-2.pre_lev \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb.RL.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.txt  \
--evalb $EVALB \
--multitask \
--ncrfpp $NCRFPP #> $LOGS/ptb.RL.multitask.3R.-2.pre_lev.$TEST_NAME.$USE_GPU.log 2>&1


