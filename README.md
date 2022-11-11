# MVLT
PyTorch implementation of BMVC2022 paper [Masked Vision-Language Transformers for Scene Text Recognition](https://arxiv.org/abs/2211.04785).  

### Dependency

- The code was tested on PyTorch\==1.12.0, timm\==0.3.2
- other requirements: lmdb, pillow, torchvision, tensorboad
## Datasets
- Training Datasets
  - [MJSynth(MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText(ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
  - [Real Datasets](https://github.com/ku21fan/STR-Fewer-Labels/)
- Evaluation Datasets
  - IIIT, SVT, IC13, IC15, SVTP, and CUTE 
 
 We use LMDB of MJ, ST and evaluation datasets by downloaded from [ABINet](https://github.com/FangShancheng/ABINet). Real datasets can be downloaded from [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels/) and we only need training datasets of the real datasets.  
 ```
 #create data directory
 $ cd MVLT
 $ mkdir data
 ```
 
The structure of `data` directory is  

    ```
    data  
    ├── training  
    │   ├── MJ  
    │   │   ├── MJ_test  
    │   │   ├── MJ_train  
    │   │   └── MJ_valid  
    │   ├── ST  
    │   ├── RealLabel
    │   │   ├── 1.SVT  
    │   │   ├── 2.IIIT  
    │   │   ├── 3.IC13
    │   │   ├── 4.IC15
    │   │   ├── 5.COCO
    │   │   ├── 6.RCTW17
    │   │   ├── 7.Uber
    │   │   ├── 8.ArT
    │   │   ├── 9.LSVT
    │   │   ├── 10.MLT19
    │   │   └── 11.ReCTS
    │   └── RealUnlabel
    │       ├── U1.Book32
    │       ├── U2.TextVQA
    │       └── U3.STVQA
    └── evaluation  
        ├── CUTE80  
        ├── IC13_857  
        ├── IC15_1811  
        ├── IIIT5k_3000  
        ├── SVT  
        └── SVTP  
  
    ```   
## Models
You can get the all models from [BaiduNetdisk(passwd:409r)](https://pan.baidu.com/s/19Hfwjmhy1qAzfJebiOkVHA)

## Pretraining
1. pretrain MVLT  (using only synthetic data)
    ```
    bash scripts/run_mvlt_pretrain.sh
    ```
2. pretrain MVLT* (using additional unlabeled real data)
    ```
    bash scripts/run_mvlt_pretrain_ur.sh
    ```
3. fine-tune
    ```
    bash scripts/run_mvlt_finetune.sh OUTPUT_DIR_PATH/checkpoint-xxx.pth
    ```
## Evaluation  
  
```
bash scripts/run_mvlt_test.sh OUTPUT_DIR_PATH/checkpoint-best.pth
```



## Acknowledgements
Our implementation is based on [MAE](https://github.com/facebookresearch/mae), [ABINet](https://github.com/FangShancheng/ABINet), 
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).


## License
This project is under the CC-BY-NC 4.0 license. See LICENSE for details.


  
