## A Light Weight Model for Video Shot Occlusion Detection

This repository contains code and models for our [paper](www.baidu.com):

> A Light Weight Model for Video Shot Occlusion Detection 
> Junhua Liao, Haihan Duan, Wanbin Zhao, Yanbing Yang, Liangyin Chen


### Setup 

1) Download the model weights and place them in the `weights` folder:


Model weights:
- [ICASSP_Model.pth](www.baidu.com)

1) Download the dataset and place them in the `data` folder:


Dataset:
- [OcclusionDataSet-MM20](https://junhua-liao.github.io/Occlusion-Detection/)

  
2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

### Usage 

1) Run occlusion detection model:

    ```shell
    python test.py
    ```

### Citation

Please cite our papers if you use this code or any of the models. 
```
@inproceedings{liao2020occlusion,
  title={Occlusion Detection for Automatic Video Editing},
  author={Liao, Junhua and Duan, Haihan and Li, Xin and Xu, Haoran and Yang, Yanbing and Cai, Wei and Chen, Yanru and Chen, Liangyin},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2255--2263},
  year={2020}
}
```

```
ICASSP 2022 soon
```
