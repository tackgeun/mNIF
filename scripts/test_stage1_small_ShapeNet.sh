python main_stage1_cavia.py -m=experiments/ShapeNet-small/LatentMixtureINR-K256L4W64H512-W0-50-subsampling4096-lr1e-4+1.0-lrschedule-batch32-epoch800/metainits/epoch799.pth --model_type=latent0.0001-mixtureinr-layerwise --hidden_features=512 --k_mixtures=256 --use_meta_sgd --width=64 --depth=4 --w0=50 --num_inner=3 --lr_outer=1e-4 --lr_inner=1.0 --batch_size=32 --dataset=shapenet --eval
python main_stage1_cavia.py -m=experiments/ShapeNet-small/LatentMixtureINR-K256L4W64H512-W0-50-subsampling4096-lr1e-4+1.0-lrschedule-batch32-epoch800/metainits/epoch799.pth --model_type=latent0.0001-mixtureinr-layerwise --hidden_features=512 --k_mixtures=256 --use_meta_sgd --width=64 --depth=4 --w0=50 --num_inner=3 --lr_outer=1e-4 --lr_inner=1.0 --batch_size=32 --dataset=shapenet --eval --split=test

