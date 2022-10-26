# MMIN

This repo implements the Consistent Feature aware Missing Modality Imagination Network(CF-MMIN) for the following paper:
"Exploiting modality-consistent feature for robust multimodal emotion recognition with missing modalities" 

# Environment

``` 
python 3.8.0
pytorch >= 1.0.0
```

# Usage

First you should change the data folder path in ```data/config``` and preprocess your data follwing the code in ```preprocess/```.

You can download the preprocessed feature to run the code.

+ For Training MMIN on IEMOCAP:

    First training a model fusion model with all audio, visual and lexical modality as the pretrained encoder.

    ```bash
    bash scripts/CAP_utt_shared.sh AVL [num_of_expr] [GPU_index]
    ```

    Then

    ```bash
    bash scripts/CAP_mmin.sh [num_of_expr] [GPU_index]
    ```


Note that you can run the code with default hyper-parameters defined in shell scripts, for changing these arguments, please refer to options/get_opt.py and the ```modify_commandline_options``` method of each model you choose.

# Download the features
Baidu Yun Link
IEMOCAP A V L modality Features
链接: https://pan.baidu.com/s/1WmuqNlvcs5XzLKfz5i4iqQ 提取码: gn6w 

[comment]: <> (# License)

[comment]: <> (MIT license. )

[comment]: <> (Copyright &#40;c&#41; 2021 AIM3-RUC lab, School of Information, Renmin University of China.)

[comment]: <> (# Citation)

[comment]: <> (If you find our paper and this code usefull, please consider cite)

[comment]: <> (```)

[comment]: <> (@inproceedings{zhao2021missing,)

[comment]: <> (  title={Missing modality imagination network for emotion recognition with uncertain missing modalities},)

[comment]: <> (  author={Zhao, Jinming and Li, Ruichen and Jin, Qin},)

[comment]: <> (  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing &#40;Volume 1: Long Papers&#41;},)

[comment]: <> (  pages={2608--2618},)

[comment]: <> (  year={2021})

[comment]: <> (})

[comment]: <> (```)
