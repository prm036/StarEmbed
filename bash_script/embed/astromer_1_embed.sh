

# to the directory contain weights/macho
cd /path/to/model/astromer_1

python /path/to/model/astromer_1/embed.py \
    --input_path /path/to/input/hf_macho_70-10-20 \
    --output_path /path/to/embeddings/hf_macho_unlabel_embeddings_astromer_1 \
    --model_name macho \
    --bands g r \
    --splits validation \
    --duration 200 \
    --enc_batch 1024 \
    --preproc_procs 8 \  # number of worker for preprocessing the light curve into windows

