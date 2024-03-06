# GAE-based_Model_Poisoning_Attack_FL
These codes are about graph autoencoder-based model poisoning attack against federated learning.

Here is a structure of  GAE-based federated learning model poisoning attack:

![Image Alt text.](/readme_pics/GAE-based_attack.png)
The proposed GAE model for generating data-agnostic, malicious local models, where the attacker overhears the benign local model $W_k^t$,  $\forall k$ and applies the GCN-based encoder to create representation $Z^M$ . The output of the encoder, i.e., the feature representations, is input to the decoder for feature reconstruction.


K. Li, J. Zheng, X. Yuan, W. Ni, O. B. Akan and H. V. Poor, "Data-Agnostic Model Poisoning Against Federated Learning: A Graph Autoencoder Approach," in IEEE Transactions on Information Forensics and Security, vol. 19, pp. 3465-3480, 2024, doi: 10.1109/TIFS.2024.3362147. The more details can be found [here](https://ieeexplore.ieee.org/document/10419367)

## Requirements
- Install requirements via  `pip install -r requirements.txt`


## How to run :point_down:
```
python FL_SVM_GAE_Attack_Device_Side.py 
```


## References
1. https://github.com/aswarth123/Federated_Learning_MNIST
2. https://github.com/zfjsail/gae-pytorch

# Citation
```
@article{li2024data,
  title={Data-Agnostic Model Poisoning against Federated Learning: A Graph Autoencoder Approach},
  author={Li, Kai and Zheng, Jingjing and Yuan, Xin and Ni, Wei and Akan, Ozgur B and Poor, H Vincent},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```

