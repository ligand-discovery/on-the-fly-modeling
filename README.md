# On the Fly Modeling App
Getting AI/ML models on-the-fly based on primary Ligand Discovery screening data

To run the app, make sure you have the necessary dependencies installed.

## Installation

```bash
# create conda environment
conda create -n onthefly python=3.10
conda activate onthefly

# install tabpfn for cpu usage
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/DhanshreeA/TabPFN.git
pip install TabPFN/.
rm -rf TabPFN

# other dependencies
pip install lolP==0.0.4
pip install streamlit
pip install networkx
pip install python-louvain

# install fragment embedding
git clone https://github.com/ligand-discovery/fragment-embedding.git
pip install -e fragment-embedding/.

# install combine mols
pip install CombineMols
```

## Usage

You can deploy the app using the command `streamlit run app/app.py`.
