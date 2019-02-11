python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export TFHUB_CACHE_DIR=data/module

mkdir data/module

# Lite model (25 MB)
wget 'https://tfhub.dev/google/universal-sentence-encoder-lite/2?tf-hub-format=compressed' -O ${TFHUB_CACHE_DIR}/module_lite.tar.gz
cd ${TFHUB_CACHE_DIR}
mkdir -p universal-sentence-encoder-lite-2 && tar -zxvf module_lite.tar.gz -C universal-sentence-encoder-lite-2