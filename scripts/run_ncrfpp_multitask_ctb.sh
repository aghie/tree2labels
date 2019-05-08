#HOME=/home/david.vilares/
#Update this path to your virtual environment
#source $HOME/env/tree2labels/bin/activate

HOME_NCRFpp=../NCRFpp/
TEST_NAME="test"
INPUT=../sample_data/cp_datasets/ctb/ctb-$TEST_NAME.seq_lu
#INPUT=$HOME/Escritorio/dataset/ptb/ptb-$TEST_NAME.seq_lu
TEST_PATH=../sample_data/cp_datasets/CTB_pred_tags/$TEST_NAME"_ch.trees"
USE_GPU=False
EVALB=../EVALB/evalb
OUTPUT=../outputs/
MODELS=../pretrained_models_naacl2019/ctb/
NCRFPP=$HOME_NCRFpp
LOGS=../logs/
#MULTITASK=True

################################################
#								BASIC MODELS
################################################


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.emnlp2018.f \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb.emnlp2018.f.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/ctb.emnlp2018.f.$TEST_NAME.$USE_GPU.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.emnlp2018.3R.-2 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb.emnlp2018.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/ctb.emnlp2018.f.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1


###################################################
# +MTL
###################################################


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2 \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/ctb.multitask.3R.-2.$TEST_NAME.$USE_GPU.log 2>&1



################################################
# + BEST AUX TASK
################################################

#NEXT X

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.multitask.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/ctb.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1 


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb.RL.multitask.3R.-2.nex_lev \
--status test  \
--gpu $USE_GPU \
--multitask \
--output $OUTPUT/ctb.RL.multitask.$TEST_NAME.$USE_GPU.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP #> $LOGS/ctb.multitask.3R.-2.nex_lev.$TEST_NAME.$USE_GPU.log 2>&1 





