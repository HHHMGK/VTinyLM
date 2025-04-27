#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./run_experiments.sh [1|2|3]"
    echo "  1: Run basic consecutive method"
    echo "  2: Run each method"
    echo "  3: Run combine method"
    exit 1
fi

# Get the argument
experiment_type=$1
MODEL=$2
# phogpt = "vinai/PhoGPT-4B-Chat" or 
# llama = "meta-llama/Llama-3.2-3B-Instruct"
# Convert model name to full path
if [ "$MODEL" = "phogpt" ]; then
    MODEL="vinai/PhoGPT-4B-Chat"
elif [ "$MODEL" = "llama" ]; then
    MODEL="meta-llama/Llama-3.2-3B-Instruct"
fi


# Method mag grad act combine			 
# Prune layers: 2, 3, 4, 8
# Prune dataset: c4, oscarvi
# Benchmark dataset: essay, oscarvi, c4
# Run combinations 

case $experiment_type in
    1)
        echo "Running basic consecutive pruning experiments..."

        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-dataset-oscarvi \
        --eval_base --output pruning_results.csv --measure_time --output_console
        
        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-dataset-c4 \
        --eval_base --output pruning_results.csv --measure_time --output_console
        
        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-essay-vn \
        --eval_base --output pruning_results.csv --measure_time --output_console
        
        ;;
    2)
        echo "Running each method pruning experiments..."

        # Magnitude, 2 layer, c4, c4
        python main.py prune --base_model vinai/PhoGPT-4B-Chat \
        --pruning_method magnitude --pruning_layer_num 2 --pruning_avg --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 512 \
        --benchmark perplexity-dataset-c4 \
        --eval_base --output pruning_results.csv --measure_time --output_console

        

        
    3)
        echo "Running combined method pruning experiments..."
        
        # Experiment 3.1: Combined method on essay benchmark
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_rate 0.2 --pruning_avg --pruning_mag_norm l1 \
        --benchmark perplexity-essay-vn \
        --pruning_data oscarvi --pruning_n_sample 512 --pruning_rand_data \
        --eval_base --output pruning_combine_essay.csv --measure_time --output_console
        
        # Experiment 3.2: Combined method on Oscar-VI dataset benchmark
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_rate 0.2 --pruning_avg --pruning_mag_norm l1 \
        --benchmark perplexity-dataset-oscarvi \
        --pruning_data oscarvi --pruning_n_sample 512 --pruning_rand_data \
        --eval_base --output pruning_combine_oscarvi.csv --measure_time --output_console
        
        # Experiment 3.3: Combined method with higher pruning rate
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_rate 0.3 --pruning_avg --pruning_mag_norm l1 \
        --benchmark perplexity-essay-vn \
        --pruning_data oscarvi --pruning_n_sample 512 --pruning_rand_data \
        --eval_base --output pruning_combine_higher_rate.csv --measure_time --output_console
        ;;
        
    *)
        echo "Invalid argument: $experiment_type"
        echo "Usage: ./run_experiments.sh [1|2|3]"
        echo "  1: Run magnitude-based pruning experiments"
        echo "  2: Run gradient-based pruning experiments"
        echo "  3: Run combined method pruning experiments"
        exit 1
        ;;
esac

echo "All experiments completed!"