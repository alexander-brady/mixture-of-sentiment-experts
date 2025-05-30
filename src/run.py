import torch, os, argparse
from transformers import AutoTokenizer

from moe.moe_classifier import MoEClassifier
from moe.trainer import train, predict
from .utils import load_config, load_dataloaders


def run_moe():
    '''Run the MoE model training and evaluation pipeline.'''
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MoEClassifier(**config['moe_params']).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config['experts'][0], use_fast=False)
    
    label_map = config.get('label_map', None)
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
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline models.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["moe"],
        default="moe",
        help="Model to run."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.model == "moe":
        run_moe()
    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == "__main__":
    parse_args()
    main()