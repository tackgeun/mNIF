# Generative Neural Fields by Mixtures of Neural Implicit Functions
The official implementation of Generative Neural Fields by Mixtures of Neural Implicit Functions
- [Tackgeun You](https://tackgeun.github.io/), [Mijeong Kim](https://mjmjeong.github.io/), [Jungtaek Kim](https://jungtaekkim.github.io/) and [Bohyung Han](https://cv.snu.ac.kr/index.php/~bhhan/), (**NeurIPS 2023**)

## Requirements
We have tested our codes on the environment below
- `Python 3.8` / `Pytorch 1.10` / `torchvision 0.11.0` / `CUDA 11.3` / `Ubuntu 18.04`  .

Please run the following command to install the necessary dependencies
```bash
pip install -r requirements.txt
```

or you can use provided docker image.
```
docker pull tackgeun/mNIF:init
```

## Dataset Preparation
Here are benchmarks from three modalities adopted in our work.
Extract those zip files in ${ROOT}/datasets.
- [CelebAHQ 128px](https://www.dropbox.com/scl/fi/l4p15ecmnm9k5qnq8kkx3/CelebAHQ.zip?rlkey=xn2tllj539fkizp9rn2xgsq4x&dl=0).
- [ShapeNet 64x64x64](https://www.dropbox.com/scl/fi/lj9uwsw1234jfw2bpupfk/shapenet.zip?rlkey=2bkv8xmc7en6ok6mc5oczmn5b&dl=0)
- [SRN Cars 128px](https://www.dropbox.com/scl/fi/4maypw7idr7yis8cwxw4d/srn_cars_lmdb.zip?rlkey=zdd67iy24t20xwjbn3vyn6o1q&dl=0)


## Pre-trained models
Here are pre-trained models from three modalities in our work.
Extract those zip files in ${ROOT}/experiments.
- [mNIF (S) CelebAHQ 64px](https://www.dropbox.com/scl/fi/iurr0s79glpehvkdezhuw/CelebAHQ-small.zip?rlkey=6spmw045i2on46glke5l6up4k&dl=0)
- [mNIF (S) ShapeNet 64x64x64](https://www.dropbox.com/scl/fi/8ievjsf0jlmbrof0awlb1/ShapeNet-small.zip?rlkey=m7riec4rmek232p6gswo2mubm&dl=0)
- [mNIF (S) SRN Cars 128px](https://www.dropbox.com/scl/fi/t9qk83eewy0uosn9syt5n/SRNCars-small.zip?rlkey=06jw7gvczv2d61agdeo1mt9yf&dl=0)


## Training and Evaluation Commands
Refer to the shell scripts in scripts.
### Training Mixtures of Neural Implicit Functions with Meta-Learning (Image, Voxel)
Training stage 1 mNIF with fast context adaptative via meta learning (CAVIA)
```
sh scripts/train_stage1_small_CelebAHQ.sh
sh scripts/train_stage1_small_ShapeNet.sh
```

### Evaluation and Test-time Adaptation of Mixtures of Neural Implicit Functions with Meta-Learning (Image, Voxel)
CAVIA simultaneously conducts adaptation and evaluation of given samples.
```
sh scripts/test_stage1_small_CelebAHQ.sh
sh scripts/test_stage1_small_ShapeNet.sh
```
For evaluation, remove result path in -r and add a specific model -m=${MODEL_PATH}/metainits/epoch${EPOCH}.pth and add --eval flag.
- It also computes context vectors in latent space, which is saved on ${MODEL_PATH}/contexts/context-epoch${EPOCH}.pth

If out-of-memory occurs during evaluation, reduce the batch size and lr_inner because lr_inner is dependent on batch size currently.
- If the model is trained with batch_size=32 and lr_inner=10.0, batch_size=16 requires lr_inner=5.0

### Training Mixtures of Neural Implicit Functions with Auto-Decoding (NeRF)
Training stage 1 mNIF with auto decoding
```
sh scripts/train_stage1_small_SRNCars.sh
```

### Evaluation of Mixtures of Neural Implicit Functions with Auto-Decoding (NeRF)
Evaluation stage 1 mNIF with auto decoding.
Contrary to CAVIA, auto-decoding procedure already computes context vectors during stage 1 training.
```
sh scripts/test_stage1_small_SRNCars.sh
```


### Training Denoising Diffusion Process
Training latent diffusion model using features acquired from the context adaptation.
Testingi and test-time adaptation of stage 1 model is required for stage 1 model trained with CAVIA.
```
sh scripts/train_stage2_small_CelebAHQ.sh
sh scripts/train_stage2_small_ShapeNet.sh
sh scripts/train_stage2_small_SRNCars.sh
```

### Evaluating diffusion model 
```
sh scripts/test_stage2_small_CelebAHQ.sh
sh scripts/test_stage2_small_ShapeNet.sh
sh scripts/test_stage2_small_SRNCars.sh
```


## Acknowledgement
Our implementation is based on below repositories.
- Datasets
  - CelebAHQ is duplicated dataset from Functa
  - Pre-processing [ShapeNet 64x64x64 IMNet](https://drive.google.com/open?id=158so7dnkQQNFSQTj741S3SUbuIXXRrLn) from [IMNet](https://github.com/czq142857/IM-NET)
  - Pre-processing SRNCars dataset from [PixelNeRF](https://github.com/sxyu/pixel-nerf)
- Mixtures of neural implicit functions
  - [SIREN](https://github.com/vsitzmann/siren)
  - [Functa](https://github.com/deepmind/functa)
  - [PixelNeRF](https://github.com/sxyu/pixel-nerf)
- ShapeNet evaluation
  - [GEM](https://github.com/yilundu/gem)
- Latent diffusion model
  - [ADM](https://github.com/openai/guided-diffusion)
  - [Karlo](https://github.com/kakaobrain/karlo)
  - [HQ-Transformer](https://github.com/kakaobrain/hqtransformer)
