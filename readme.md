### Towards Graph-hop Retrieval and Reasoning in Complex Question Answering over Textual Database





## Prerequisites

- python 3.7 with pytorch (`1.10.0`), transformers(`4.15.0`), tqdm, accelerate, pandas, numpy, sentencepiece, sklearn, networkx
- cuda10/cuda11

#### Installing the GPU driver

```shell script
# preparing environment
sudo apt-get install gcc
sudo apt-get install make
wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run
sudo sh cuda_11.5.1_495.29.05_linux.run
```

#### Installing Conda and Python

```shell script
# preparing environment
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod 777 Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh

conda create -n TreeHop python==3.7
conda activate TreeHop
```

#### Installing Python Libraries

```plain
# preparing environment
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install tqdm transformers sklearn pandas numpy networkx accelerate sentencepiece
```

### 

### Data

We provide the reasontree dataset in the  `./data`



### BiDGR



##### How to train:

```
python train_F.py #Run forward graph retrieval
python train_B.py #Run backward graph retrieval
```

##### How to eval:

```
python eval.py #When the training is over, eval.py file will read the last saved model weight and test the score
```





## Cite

``````
@article{zhu2023towards,
  title={Towards Graph-hop Retrieval and Reasoning in Complex Question Answering over Textual Database},
  author={Zhu, Minjun and Weng, Yixuan and He, Shizhu and Liu, Kang and Zhao, Jun},
  journal={arXiv preprint arXiv:2305.14211},
  year={2023}
}

``````

