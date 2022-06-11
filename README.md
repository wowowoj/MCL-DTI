
# MCL-DTI
MCL-DTI: Using Drug Multimodal Information and Bi-directional Cross-Attention Learning method for Predicting Drug-Target Interaction

##### Dependencies：

- Pytorch
- numpy
- rdkit



##### Using

1. `label.py` generates sample labels for the data
2. `smile_to_image.py` and `smile_to_features.py` generate images of drug molecules and chemical text information, respectively. `smiles_k_gram.py` lets the chemical text be divided into words according to the k-gram method 
3. `protein_k_gram` lets the protein sequences of drug targets be divided into words according to the k-gram method 
4. `main.py` trains MCL-DTI model.