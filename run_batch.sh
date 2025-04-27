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
# Benchmark dataset: essay, oscarvi
# Run combinations 

case $experiment_type in
    1)
        echo "Running basic consecutive pruning experiments..."

        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-dataset-oscarvi \
        --eval_base --output results/pruning_results_xcon.csv --measure_time --output_console
        
        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-dataset-c4 \
        --eval_base --output results/pruning_results_xcon.csv --measure_time --output_console
        
        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_xcon.csv --measure_time --output_console
    
        ;;
    2)
        echo "Running each method pruning experiments..."

        # Magnitude, c4, c4
        # python main.py prune --base_model $MODEL \
        # --pruning_method magnitude --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        # --benchmark perplexity-dataset-c4 \
        # --eval_base --output results/pruning_results_mag.csv --measure_time --output_console
        # with l2 norm
        # python main.py prune --base_model $MODEL \
        # --pruning_method magnitude --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l2 \
        # --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        # --benchmark perplexity-dataset-c4 \
        # --eval_base --output results/pruning_results_mag.csv --measure_time --output_console
        # Magnitude, c4, essay
        python main.py prune --base_model $MODEL \
        --pruning_method magnitude --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_mag.csv --measure_time --output_console

        # Magnitude, oscarvi, oscarvi
        python main.py prune --base_model $MODEL \
        --pruning_method magnitude --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-dataset-oscarvi \
        --eval_base --output results/pruning_results_mag.csv --measure_time --output_console
        # Magnitude, oscarvi, essay 
        python main.py prune --base_model $MODEL \
        --pruning_method magnitude --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_mag.csv --measure_time --output_console \
        
        # Gradient, c4, c4
        # python main.py prune --base_model $MODEL \
        # --pruning_method gradient --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        # --benchmark perplexity-dataset-c4 \
        # --eval_base --output results/pruning_results_grad.csv --measure_time --output_console
        # Gradient, c4, essay
        python main.py prune --base_model $MODEL \
        --pruning_method gradient --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_grad.csv --measure_time --output_console
        # Gradient, oscarvi, oscarvi
        python main.py prune --base_model $MODEL \
        --pruning_method gradient --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-dataset-oscarvi \
        --eval_base --output results/pruning_results_grad.csv --measure_time --output_console
        # Gradient, oscarvi, essay
        python main.py prune --base_model $MODEL \
        --pruning_method gradient --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_grad.csv --measure_time --output_console

        # Activation, c4, c4
        # python main.py prune --base_model $MODEL \
        # --pruning_method activation --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        # --benchmark perplexity-dataset-c4 \
        # --eval_base --output results/pruning_results_act.csv --measure_time --output_console
        # Activation, c4, essay
        python main.py prune --base_model $MODEL \
        --pruning_method activation --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_act.csv --measure_time --output_console

        # Activation, oscarvi, oscarvi
        python main.py prune --base_model $MODEL \
        --pruning_method activation --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-dataset-oscarvi \
        --eval_base --output results/pruning_results_act.csv --measure_time --output_console
        # Activation, oscarvi, essay
        python main.py prune --base_model $MODEL \
        --pruning_method activation --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_act.csv --measure_time --output_console

        ;;        
    3)
        echo "Running combined method pruning experiments..."
        
        # Combined, c4, essay
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_comb.csv --measure_time --output_console

        # Combined, oscarvi, oscarvi
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-dataset-oscarvi \
        --eval_base --output results/pruning_results_comb.csv --measure_time --output_console

        # Combined, oscarvi, essay
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_comb.csv --measure_time --output_console

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