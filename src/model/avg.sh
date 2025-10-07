

INPUT_PATHS=(
    # "/path/to/embeddings/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr"
    # "/path/to/embeddings/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr"
    # "/path/to/embeddings/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr"
    # "/path/to/embeddings/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct"
    # "/path/to/embeddings/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True"
    "/path/to/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr"
    "/path/to/embeddings/random"
    # "/path/to/input/hf_csdr1_multiband_raw_lc_subclass_class_str_v2"
)


for input_path in "${INPUT_PATHS[@]}"; do
    # Extract just the dataset name from the full path
    dataset_name=$(basename "$input_path")
    echo "Processing dataset: $dataset_name from path: $input_path"
    python /path/to/model/compute_avg_embeddings.py --dataset "$dataset_name" --batch_size 1000
done