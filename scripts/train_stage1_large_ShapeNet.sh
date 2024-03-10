python main_stage1_cavia.py -r=results/ShapeNet-large/mNIF-stage1-M512L5W128H1024 --model_type=latent0.0001-mixtureinr-layerwise --hidden_features=1024 --k_mixtures=512 --use_meta_sgd --width=128 --depth=5 --num_inner=3 --lr_outer=1e-4 --lr_inner=1.0 --batch_size=32 --dataset=shapenet --save_freq=100 --num_epochs=400 --use_lr_schedule --clip_grad


