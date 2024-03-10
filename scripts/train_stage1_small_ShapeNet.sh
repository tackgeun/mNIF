python main_stage1_cavia.py -r=results/ShapeNet-small/mNIF-stage1-M256L4W64H512 --model_type=latent0.0001-mixtureinr-layerwise --hidden_features=512 --k_mixtures=256 --use_meta_sgd --width=64 --depth=4 --w0=50 --num_inner=3 --lr_outer=1e-4 --lr_inner=1.0 --batch_size=32 --dataset=shapenet --save_freq=100 --num_epochs=800 --use_lr_schedule --clip_grad

