python main.py eval --benchmark perplexity-essay-vn --base_model vinai/PhoGPT-4B-Chat --output results.csv --modification layer_reduction --measure_time True

python main.py train --base_model vinai/PhoGPT-4B-Chat --pruning True --pruning_layer_start 16 --pruning_layer_end 17 --dataset_path datasets\vneconomy\vneconomy.json --eval_after_train True

python main.py prune --base_model vinai/PhoGPT-4B-Chat --pruning_method magnitude --pruning_rate 0.2 --pruning_avg --pruning_mag_norm l1 --benchmark perplexity-essay-vn --eval_base --output pruning_results.csv --measure_time --pruning_rand_data --pruning_n_sample 512 --output_console 

!python main.py prune --base_model vinai/PhoGPT-4B-Chat \
--pruning_method combine --pruning_rate 0.2 --pruning_avg --pruning_mag_norm l1 --pruning_rand_data --pruning_n_sample 512 \
--benchmark perplexity-essay-vn \
--eval_base --output pruning_results.csv --measure_time --pruning_data oscarvi --output_console

!python main.py prune --base_model vinai/PhoGPT-4B-Chat \
--pruning_method combine --pruning_rate 0.2 --pruning_avg --pruning_mag_norm l1 --pruning_rand_data --pruning_n_sample 512 \
--benchmark perplexity-dataset-oscarvi \
--eval_base --output pruning_results.csv --measure_time  --output_console