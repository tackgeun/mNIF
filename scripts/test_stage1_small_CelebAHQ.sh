python main_stage1_cavia.py -m=experiments/CelebAHQ-small/LatentMixtureINR-K256L4W64H1024-lr1e-4+1.0-lrschedule-batch32-epoch800/metainits/epoch799.pth --model_type=latent0.0001-mixtureinr-layerwise --hidden_features=1024 --k_mixtures=256 --use_meta_sgd --width=64 --depth=4 --num_inner=3 --lr_outer=1e-4 --lr_inner=1.0 --batch_size=32 --eval
python main_stage1_cavia.py -m=experiments/CelebAHQ-small/LatentMixtureINR-K256L4W64H1024-lr1e-4+1.0-lrschedule-batch32-epoch800/metainits/epoch799.pth --model_type=latent0.0001-mixtureinr-layerwise --hidden_features=1024 --k_mixtures=256 --use_meta_sgd --width=64 --depth=4 --num_inner=3 --lr_outer=1e-4 --lr_inner=1.0 --batch_size=32 --eval --split=test

