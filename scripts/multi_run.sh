CUDA_VISIBLE_DEVICES=1 python3 translate_nllb.py \
--data_split dev \
--input_path data/imdb/ \
--src_lang eng_Latn \
--tgt_lang hau_Latn \
--model_type nllb \
--model_name_or_path facebook/nllb-200-distilled-600M \
--max_seq_length  400 \
--seed 1 \
--do_predict




CUDA_VISIBLE_DEVICES=1 python3 translate_nllb.py \
--data_split dev \
--input_path data/imdb/ \
--src_lang eng_Latn \
--tgt_lang ibo_Latn \
--model_type nllb \
--model_name_or_path facebook/nllb-200-distilled-600M \
--max_seq_length  400 \
--seed 1 \
--do_predict



CUDA_VISIBLE_DEVICES=1 python3 translate_nllb.py \
--input_path data/imdb/ \
--src_lang en \
--tgt_lang fr \
--model_type m2m100 \
--model_name_or_path masakhane/m2m100_418M_en_pcm_rel_news \
--data_split dev \
--max_seq_length 400 \
--seed 1 \
--do_predict