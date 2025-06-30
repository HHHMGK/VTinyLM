#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./run_experiments.sh [1|2|3|all] [phogpt|llama]"
    echo "  1: Run basic consecutive method"
    echo "  2: Run each method"
    echo "  3: Run combine method"
    echo "  all: Run all methods (1, 2, and 3)"
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
elif [ "$MODEL" = "qwen" ]; then
    MODEL="Qwen/Qwen3-4B"
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
        --eval_base --output results/pruning_results_xcon.csv --measure_time --output_console --eval_repeat 5
        
        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-dataset-c4 \
        --eval_base --output results/pruning_results_xcon.csv --measure_time --output_console --eval_repeat 5
        
        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_xcon.csv --measure_time --output_console --eval_repeat 5

        python main.py prune --base_model $MODEL \
        --pruning_method xconsecutive --pruning_layer_num 1 2 3 4 6 \
        --benchmark perplexity-dataset-vnnews \
        --eval_base --output results/pruning_results_xcon.csv --measure_time --output_console --eval_repeat 5
    
        ;;
    2)
        echo "Running each method pruning experiments..."

        # # Magnitude, essay 
        # python main.py prune --base_model $MODEL \
        # --pruning_method magnitude --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --benchmark perplexity-essay-vn \
        # --eval_base --output results/pruning_results_mag.csv --measure_time --output_console --eval_repeat 5
        # # Magnitude, vnnews
        # python main.py prune --base_model $MODEL \
        # --pruning_method magnitude --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --benchmark perplexity-dataset-vnnews \
        # --eval_base --output results/pruning_results_mag.csv --measure_time --output_console --eval_repeat 5

        # Gradient, c4, essay
        python main.py prune --base_model $MODEL \
        --pruning_method gradient --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_grad.csv --measure_time --output_console --eval_repeat 5
        # Gradient, c4, vnnews
        python main.py prune --base_model $MODEL \
        --pruning_method gradient --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-dataset-vnnews \
        --eval_base --output results/pruning_results_grad.csv --measure_time --output_console --eval_repeat 5

        # Gradient, oscarvi, essay
        python main.py prune --base_model $MODEL \
        --pruning_method gradient --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_grad.csv --measure_time --output_console --eval_repeat 5
        # Gradient, oscarvi, vnnews
        python main.py prune --base_model $MODEL \
        --pruning_method gradient --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-dataset-vnnews \
        --eval_base --output results/pruning_results_grad.csv --measure_time --output_console --eval_repeat 5

        # # Activation, c4, essay
        # python main.py prune --base_model $MODEL \
        # --pruning_method activation --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        # --benchmark perplexity-essay-vn \
        # --eval_base --output results/pruning_results_act.csv --measure_time --output_console --eval_repeat 5
        # # Activation, c4, vnnews
        # python main.py prune --base_model $MODEL \
        # --pruning_method activation --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        # --benchmark perplexity-dataset-vnnews \
        # --eval_base --output results/pruning_results_act.csv --measure_time --output_console --eval_repeat 5

        # # Activation, oscarvi, essay
        # python main.py prune --base_model $MODEL \
        # --pruning_method activation --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        # --benchmark perplexity-essay-vn \
        # --eval_base --output results/pruning_results_act.csv --measure_time --output_console --eval_repeat 5
        # # Activation, oscarvi, vnnews
        # python main.py prune --base_model $MODEL \
        # --pruning_method activation --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        # --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        # --benchmark perplexity-dataset-vnnews \
        # --eval_base --output results/pruning_results_act.csv --measure_time --output_console --eval_repeat 5

        ;;        
    3)
        echo "Running combined method pruning experiments..."
        
        # Combined, c4, essay
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_comb.csv --measure_time --output_console --eval_repeat 5
        # Combined, c4, vnnews
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data c4 --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-dataset-vnnews \
        --eval_base --output results/pruning_results_comb.csv --measure_time --output_console --eval_repeat 5

        # Combined, oscarvi, essay
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-essay-vn \
        --eval_base --output results/pruning_results_comb.csv --measure_time --output_console --eval_repeat 5
        # Combined, oscarvi, vnnews
        python main.py prune --base_model $MODEL \
        --pruning_method combine --pruning_layer_num 1 2 3 4 6 --pruning_mag_norm l1 \
        --pruning_data oscarvi --pruning_rand_data --pruning_n_sample 128 \
        --benchmark perplexity-dataset-vnnews \
        --eval_base --output results/pruning_results_comb.csv --measure_time --output_console --eval_repeat 5

        ;;
    all)
        echo "Running ALL experiments (1, 2, and 3)..."
        echo "-----------------------------------------"
        echo "Running Case 1: Basic consecutive pruning..."
        bash run_batch.sh 1 $MODEL
        echo "-----------------------------------------"
        echo "Running Case 2: Each method pruning..."
        bash run_batch.sh 2 $MODEL
        echo "-----------------------------------------"
        echo "Running Case 3: Combined method pruning..."
        bash run_batch.sh 3 $MODEL
        echo "-----------------------------------------"
        echo "All experiment sets completed!"
        ;;
    *)
        echo "Invalid argument: $experiment_type"
        echo "Usage: ./run_experiments.sh [1|2|3|all] [phogpt|llama]"
        echo "  1: Run magnitude-based pruning experiments"
        echo "  2: Run gradient-based pruning experiments"
        echo "  3: Run combined method pruning experiments"
        exit 1
        ;;
esac

echo "All experiments completed!"