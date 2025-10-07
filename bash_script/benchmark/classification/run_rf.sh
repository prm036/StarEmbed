


python /path/to/benchmark/classification/rf_hpo.py \
 --input-embs /path/to/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr \
 --standardize 1 \
 --hand-crafted 1 \
 --seed 42 \
 --skip-hpo \
 --best-params '{"max_depth": None, "min_samples_split": 10, "n_estimators": 100}' \
 --output-dir /path/to/output/rf/handcrafted_feature/new_avg


python /path/to/benchmark/classification/rf_hpo.py \
 --input-embs /path/to/embeddings/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr \
 --hand-crafted 0 \
 --seed 42 \
 --skip-hpo \
 --best-params '{"max_depth": None, "min_samples_split": 5, "n_estimators": 100}' \
 --output-dir /path/to/output/rf/handcrafted_feature/new_avg_default


# python /path/to/benchmark/classification/rf_hpo.py \
#  --input-embs /path/to/embeddings/hf_csdr1_multiband_raw4_embeddings_astromer_2_gr_sampling_True \
#  --hand-crafted 0 \
#  --seed 42 \
#  --skip-hpo True\
#  --best-params '{"max_depth": 30, "min_samples_split": 10, "n_estimators": 500}' \
#  --output-dir /path/to/output/rf/astromer_2



# hpo
# python /path/to/benchmark/classification/rf_hpo.py \
#  --input-embs /path/to/embeddings/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr \
#  --hand-crafted 0 \
#  --seed 42 \
#  --output-dir /path/to/output/rf/astromer_2 
 
# python /path/to/benchmark/classification/rf_hpo.py \
#  --input-embs /path/to/embeddings/csdr1_raw_embs_moiral_small_trn_val_tst_ctx200_pdt64_psz16_bandgr \
#  --hand-crafted 0 \
#  --seed 42 \
#  --output-dir /path/to/output/rf/moirai

# python /path/to/benchmark/classification/rf_hpo.py \
#  --input-embs /path/to/embeddings/hf_csdr1_multiband_raw4_embeddings_astromer_1_subclass_pad_correct \
#  --hand-crafted 0 \
#  --seed 42 \
#  --output-dir /path/to/output/rf/astromer_1

# python /projects/b1094/StarEmbed/src/benchmark/classification/rf_hpo.py \
#  --input-embs /projects/b1094/StarEmbed/embeddings/embeddings_with_anom/csdr1_raw4_catflags_filtered_embs_chronos_t5_tiny_trn_val_tst_ctx200_bandgr \
#  --hand-crafted 0 \
#  --seed 42 \
#  --output-dir /projects/b1094/StarEmbed/src/output/rf/chronos_tiny 

python /projects/b1094/StarEmbed/src/benchmark/classification/rf_hpo.py \
 --input-embs /projects/p32795/dennis/random \
 --hand-crafted 1 \
 --seed 42 \
 --output-dir /projects/b1094/StarEmbed/src/output/rf/random 


 
