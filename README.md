# Training an LSTM with Dropout on wikitext-2 and NYT covid 19 text.
We implement an LSTM with Dropout Regularization and train it on 2 corpora: wikitext-2 and NYT covid 19.

The model architecture is inspired by the paper "Recurrent Neural Network Regularization" by Zaremba et al. (2014), one of the earliest successful implementations of Dropout Regularization on recurrent neural networks.

Paper: [https://arxiv.org/abs/1409.2329](https://arxiv.org/abs/1409.2329)  
Original code, in Lua and Torch: [https://github.com/wojzaremba/lstm](https://github.com/wojzaremba/lstm)

The `src` directory contains four scripts:

+ `lstm.py` contains the model described as in the paper.
+ `main.py` runs the pipeline: initialize dataset -> create model -> train model. 
+ `datasets.py` is used to load data to train the model. We used wikitext-2 and some articles scraped from NYT on covid 19. 
