
# # handcrafted feature normalization first parameters
INPUTS="/path/to/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr"
for BATCH_SIZE in 32; do
    for LR in 1e-4; do
        for DROPOUT in 0.0; do
            for LAYERS in 3; do
                for see in 200; do
                    CUDA_VISIBLE_DEVICES=0 srun python /path/to/benchmark/classification/mlp_pl2_wloss_standardization.py --batch_size $BATCH_SIZE --lr $LR --dropout $DROPOUT --hidden_layers $LAYERS --out_dir "/path/to/output/mlp/new_avg" --epochs 50 --input_embs $INPUTS --scenario concat --hand_crafted 1 --seed $see
                    # echo "testing $path" # testing
                done
            done
        done
    done
done






