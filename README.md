<p align="center">
    <img src="assets/logo.png" width="300">
</p>

## RWKVIR



## 📑 Contents

- [Visual Results](#visual_results)
- [News](#news)
- [TODO](#todo)
- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)
- [Citation](#cite)


## <a name="todo"></a> ☑️ TODO

- [x] Build the repo
- [] arXiv version
- [] Release code




## <a name="installation"></a> :wrench: Installation
This codebase was tested with the following environment configurations. It may work with other versions.




## Datasets

The datasets used in our training and testing are orgnized as follows: 


| Task                                          |                         Training Set                         |                         Testing Set                          |                        Visual Results                        |
| :-------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| image SR                                      | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) [complete dataset DF2K [download](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link)] | Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1n-7pmwjP0isZBK7w3tx2y8CTastlABx1/view?usp=sharing)] | [Google Drive]() |
| gaussian color image denoising                          | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) [complete dataset DFWB_RGB [download](https://drive.google.com/file/d/1jPgG_URDQZ4kyXaMMXJ8AZ8jEErCdKuM/view?usp=share_link)] | CBSD68 + Kodak24 + McMaster + Urban100  [[download](https://drive.google.com/file/d/1baLpOjNlTCNbREUDAZf9Lso6YCeUOQER/view?usp=sharing)] | [Google Drive]() |
| real image denoising                          | [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) (320 training images) [complete dataset SIDD [download](https://drive.google.com/drive/folders/1L_8ig1P71ikzf8PHGs60V6dZ2xoCixaC?usp=share_link)] | SIDD + DND [[download](https://drive.google.com/file/d/1Vuu0uhm_-PAG-5UPI0bPIaEjSfrSvsTO/view?usp=share_link)] | [Google Drive]() |
| grayscale JPEG compression artifact reduction | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images) [complete dataset DFWB_CAR [download](https://drive.google.com/file/d/1IASyJRsX9CKBE0i5iSJMelIr_a6U5Qcd/view?usp=share_link)] | Classic5 + LIVE1 [[download](https://drive.google.com/file/d/1KJ1ArYxRubRAWP1VgONf6rly1DwiRnzZ/view?usp=sharing)] | [Google Drive]() |



## <a name="training"></a>  :hourglass: Training

### Train on SR
Train classic RWKVIR to converage
batch_size = 8 per GPU (4卡)
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/train/rwkvir/train_RWKVIRv4_SR_x2.yml --launcher pytorch
```
batch_size = 4 per GPU (8卡)
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 basicsr/train.py -opt options/train/rwkvir/train_RWKVIRv4_SR_x2_8.yml --launcher pytorch
```

<!-- 1. Please download the corresponding training datasets and put them in the folder datasets/DF2K. Download the testing datasets and put them in the folder datasets/SR.

2. Follow the instructions below to begin training our model. -->

<!-- ```
# Claissc SR task, cropped input=64×64, 8 GPUs, batch size=4 per GPU
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_SR_x2.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_SR_x3.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_SR_x4.yml --launcher pytorch

# Lightweight SR task, cropped input=64×64, 8 GPUs, batch size=8 per GPU
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_lightSR_x2.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_lightSR_x3.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 basicsr/train.py -opt options/train/train_MambaIR_lightSR_x4.yml --launcher pytorch
``` -->

<!-- 3. Run the script then you can find the generated experimental logs in the folder experiments.

### Train on Gaussian Color Image Denosing


1. Download the corresponding training datasets [here](#datasets) and put them in the folder `./datasets/DFWB_RGB`. Download the testing datasets and put them in the folder `./datasets/ColorDN`.


2. Follow the instructions below to begin training:


```
# train on denosing15
python -m torch.distributed.launch --nproc_per_node=8 --master_port=2414 basicsr/train.py -opt options/train/train_MambaIR_ColorDN_level15.yml --launcher pytorch

# train on denosing25
python -m torch.distributed.launch --nproc_per_node=8 --master_port=2414 basicsr/train.py -opt options/train/train_MambaIR_ColorDN_level25.yml --launcher pytorch

# train on denosing50
python -m torch.distributed.launch --nproc_per_node=8 --master_port=2414 basicsr/train.py -opt options/train/train_MambaIR_ColorDN_level50.yml --launcher pytorch
```


3. Run the script then you can find the generated experimental logs in the folder `./experiments`.




### Train on Real Denoising

1. Please download the corresponding training datasets and put them in the folder datasets/SIDD. Note that we provide both training and validating files, which are already processed.
2. Go to folder 'realDenoising'. Follow the instructions below to train our model.

``` 
# go to the folder
cd realDenoising
# set the new environment (BasicSRv1.2.0), which is the same with Restormer for training.
python setup.py develop --no_cuda_extgf
# train for RealDN task, 8 GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=2414 basicsr/train.py -opt options/train_MambaIR_RealDN.yml --launcher pytorch
Run the script then you can find the generated experimental logs in the folder realDenoising/experiments.
```

3. Remember to go back to the original environment if you finish all the training or testing about real image denoising task. This is a friendly hint in order to prevent confusion in the training environment.
```
# Tips here. Go back to the original environment (BasicSRv1.3.5) after finishing all the training or testing about real image denoising. 
cd ..
python setup.py develop
```


## <a name="testing"></a> :smile: Testing

### Test on SR

1. Please download the corresponding testing datasets and put them in the folder datasets/SR. Download the corresponding models and put them in the folder experiments/pretrained_models.

2. Follow the instructions below to begin testing our MambaIR model.
```
# test for image SR. 
python basicsr/test.py -opt options/test/test_MambaIR_SR_x2.yml
python basicsr/test.py -opt options/test/test_MambaIR_SR_x3.yml
python basicsr/test.py -opt options/test/test_MambaIR_SR_x4.yml
# test for lightweight image SR. 
python basicsr/test.py -opt options/test/test_MambaIR_lightSR_x2.yml
python basicsr/test.py -opt options/test/test_MambaIR_lightSR_x3.yml
python basicsr/test.py -opt options/test/test_MambaIR_lightSR_x4.yml
```


### Test on Gaussian Color Image Denoising
1. Please download the corresponding testing datasets and put them in the folder `datasets/ColorDN`. 

2. Download the corresponding models and put them in the folder `experiments/pretrained_models`.

3. Follow the instructions below to begin testing our model.

```
# test on denosing15
python basicsr/test.py -opt options/test/test_MambaIR_ColorDN_level15.yml

# test on denosing25
python basicsr/test.py -opt options/test/test_MambaIR_ColorDN_level25.yml

# test on denosing50
python basicsr/test.py -opt options/test/test_MambaIR_ColorDN_level50.yml
```



### Test on Real Image Denoising

1. Download the [SIDD test](https://drive.google.com/file/d/11vfqV-lqousZTuAit1Qkqghiv_taY0KZ/view) and [DND test](https://drive.google.com/file/d/1CYCDhaVxYYcXhSfEVDUwkvJDtGxeQ10G/view?usp=sharing). Place them in `datasets/RealDN`.  Download the corresponding models and put them in the folder `experiments/pretrained_models`. 
2. Go to folder 'realDenoising'. Follow the instructions below to test our model. The output is in `realDenoising/results/Real_Denoising`.
    ```bash
    # go to the folder
    cd realDenoising
    # set the new environment (BasicSRv1.2.0), which is the same with Restormer for testing.
    python setup.py develop --no_cuda_ext
    # test MambaIR (training total iterations = 300K) on SSID
    python test_real_denoising_sidd.py
    # test MambaIR (training total iterations = 300K) on DND
    python test_real_denoising_dnd.py
    ```
3. Run the scripts below to reproduce PSNR/SSIM on SIDD. 
   ```bash
   run evaluate_sidd.m
   ```
4. For PSNR/SSIM scores on DND, you can upload the genetated DND mat files to the [online server](https://noise.visinf.tu-darmstadt.de/) and get the results.

5. Remerber to go back to the original environment if you finish all the training or testing about real image denoising task. This is a friendly hint in order to prevent confusion in the training environment.
    ```bash
    # Tips here. Go back to the original environment (BasicSRv1.3.5) after finishing all the training or testing about real image denoising. 
    cd ..
    python setup.py develop
    ```




## <a name="cite"></a> 🥰 Citation

Please cite us if our work is useful for your research.

```
@article{guo2024mambair,
  title={MambaIR: A Simple Baseline for Image Restoration with State-Space Model},
  author={Guo, Hang and Li, Jinmin and Dai, Tao and Ouyang, Zhihao and Ren, Xudong and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2402.15648},
  year={2024}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [ART](https://github.com/gladzhang/ART) ,and [VMamba](https://github.com/MzeroMiko/VMamba). Thanks for their awesome work.

## Contact

If you have any questions, feel free to approach me at cshguo@gmail.com -->

