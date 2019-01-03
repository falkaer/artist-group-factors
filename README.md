This project is an attempt to replicate the findings presented in [Transfer Learning of Artist Group Factors to Musical Genre Classification](https://arxiv.org/abs/1805.02043).

This code was written to run with the _medium_ [Free Music Archive dataset](https://github.com/mdeff/fma), which must be extracted to the working directory as the code.

Features and AGFs are extracted from the dataset by running extractor.py and agf.py, and multi-task and single-task models can be trained by running train_mtn.py and train_predictor.py, respectively.

Model architectures can be seen in models.py, and the GradNorm implementation can be seen in the train_mtn.py step method.

predict_slices.py makes predicitons for the last layer of the single-task network, which are used for visualising the internal state of the network before prediction using t-SNE.

The utils.py, tracks.csv and genres.csv files in this repository are from the FMA repository.