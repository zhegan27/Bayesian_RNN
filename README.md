# Bayesian RNN

The code for the ACL 2017 paper “[Scalable Bayesian Learning of Recurrent Neural Networks for Language Modeling](https://arxiv.org/pdf/1611.08034.pdf)”

## Dependencies

* Most of the experiments are implemented with Theano.
* The language modeling experiment on PTB is implemented with both Theano and Torch.

## How to use the code

* The data used in our paper can be downloaded [here](https://drive.google.com/open?id=0B1HR6m3IZSO_VFIxR1VyaGREc0k).

* Running the python files can reproduce the results in the paper. 

* For the PTB dataset with successive minibatches, we extend [wojzaremba's lua code](https://github.com/wojzaremba/lstm). For the PTB dataset with random minibatches, we use the provided theano code to run the experiments. 

## Citing Bayesian RNN

Please cite our ACL paper in your publications if it helps your research:


    @inproceedings{BayesianRNN_ACL2017,
      title={Scalable Bayesian Learning of Recurrent Neural Networks for Language Modeling},
      author = {Gan, Zhe and Li, Chunyuan and Chen, Changyou and Pu, Yunchen and Su, Qinliang and Carin, Lawrence},
      booktitle={ACL},
      Year  = {2017}
    }