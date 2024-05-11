current_directory=$(pwd)

mkdir -p ${current_directory}/data

#train
wget -nc -O "${current_directory}/data/ECPred40_train.json" "https://minio.lab.sspcloud.fr/gamer35/KEGG_db/Deep_EC_datasets/ECPred40_train.json"

#validation
wget -nc -O "${current_directory}/data/ECPred40_valid.json" "https://minio.lab.sspcloud.fr/gamer35/KEGG_db/Deep_EC_datasets/ECPred40_valid.json"

#test
wget -nc -O "${current_directory}/data/ECPred40_test.json" "https://minio.lab.sspcloud.fr/gamer35/KEGG_db/Deep_EC_datasets/ECPred40_test.json"


pip install polars pandas scikit-learn transformers datasets
pip install accelerate -U