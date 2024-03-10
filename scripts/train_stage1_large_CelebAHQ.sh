python main_stage1_cavia.py -r=experiments/mNIF-stage1-M384L5W128H512 --model_type=latent0.0001-mixtureinr-layerwise --hidden_features=512 --k_mixtures=384 --use_meta_sgd --width=128 --depth=5 --num_inner=3 --lr_outer=1e-4 --lr_inner=1.0 --batch_size=32 --save_freq=100 --num_epochs=800 --use_lr_scheduler

