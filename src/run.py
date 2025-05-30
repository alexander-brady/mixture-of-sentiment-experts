import torch, os, argparse
from transformers import AutoTokenizer

from moe.moe_classifier import MoEClassifier
from moe.trainer import train, predict
from baselines.utils import load_config, load_dataloaders
from baselines.baseline_sentence_transformer import run_sentence_transformer
from baselines.baseline_distil_bert import run_distil_bert
from baselines.baseline_deberta_finetune import train_and_predict


def run_moe():
    '''Run the MoE model training and evaluation pipeline.'''
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MoEClassifier(**config['moe_params']).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config['moe_params']['experts'][0], use_fast=False)
    
    label_map = {
    "negative":0,
    "neutral":1,
    "positive":2,
    }
    reverse_label_map = {v: k for k, v in label_map.items()} if label_map else None
    
    train_loader, val_loader, test_loader = load_dataloaders(
        tokenizer, label_map=label_map, **config['dataloaders']
    )
    
    train(model, train_loader, val_loader, device=device, **config['train'])
    predict(model, test_loader, device=device, label_map=reverse_label_map)
    
    if config.get('model_save_path'):
        os.mkdir(config['model_save_path'], exist_ok=True)
        model.save_pretrained(config['model_save_path'])
        print(f"Model saved to {config['model_save_path']}")


def run_deberta_v3(size):
    if size not in ("base", "large"):
        raise ValueError(f"Invalid model size: {size}. Must be 'base' or 'large'.")
    train_and_predict(size)

    
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["moe", "sentence-transformer", "distil-bert", "deberta-v3-base", "deberta-v3-large"],
        default="moe",
        help="Model to run."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.model == "moe":
        run_moe()
    if args.model == "sentence-transformer":
        run_sentence_transformer()
    elif args.model == "distil-bert":
        run_distil_bert()
    elif args.model == "deberta-v3-base":
        run_deberta_v3("base")
    elif args.model == "deberta-v3-large":
        run_deberta_v3("large")
    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == "__main__":
    parse_args()
    main()
