


for seed in 42; do
    # python linear_knn.py --input_embs /path/to/embeddings/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr --scenario concat --seed $seed
    # python linear_knn.py --input_embs /path/to/embeddings/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr --scenario concat --seed $seed
    # python linear_knn.py --input_embs /path/to/embeddings/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct --scenario concat --seed $seed
    # python linear_knn.py --input_embs /path/to/embeddings/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True --scenario concat --seed $seed
    python benchmark/classification/linear_knn.py --input_embs /path/to/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr  --hand_crafted True --seed $seed
    # python linear_knn.py --input_embs /path/to/random --scenario concat --seed $seed
done
