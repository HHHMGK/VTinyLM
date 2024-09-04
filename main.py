import argparse
import os
from model import load_model, load_tokenizer, clone_model

# Take arg from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('run_mode', type=str, default='train', choices=['train','test'], help='Mode run mode')
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')

# Import config from config.json

args = parser.parse_args()

if args.run_mode == 'train':
    print('Training')
    print('Config path:', args.config)
    
if args.run_mode == 'test':
    print('Testing')
    print('Config path:', args.config)
    model = load_model()