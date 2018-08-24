pip install nltk==3.2.5
pip install numpy==1.14.3
pip install keras==2.1.0
pip install --upgrade tensorflow-gpu==1.4.0
pip install scikit-learn==0.19.1
pip install sklearn-crfsuite
pip install h5py==2.7.1
pip install torch==0.3.1

echo "Downloading external embeddings"
wget http://grupolys.org/software/tree2labels-emnlp2018-resources/embeddings-EMNLP2018.zip .
unzip embeddings-EMNLP2018 -d ./resources/

echo "Downloading pretrained models"
wget http://grupolys.org/software/tree2labels-emnlp2018-resources/models-EMNLP2018.zip .
unzip models-EMNLP2018.zip
