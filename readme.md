python main.py eval --benchmark perplexity-vn --base_model vinai/PhoGPT-4B-Chat --output results.csv --modification layer_reduction --measure_time True

python main.py train --base_model vinai/PhoGPT-4B-Chat --pruning True --pruning_layer_start 16 --pruning_layer_end 17 --dataset_path datasets\vneconomy\vneconomy.json --eval_after_train True