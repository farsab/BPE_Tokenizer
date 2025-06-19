import os
import argparse
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

def download_and_prepare_data(output_txt: str):
    """
    Download the AG News dataset and write all training texts to a single file.
    """
    dataset = load_dataset("ag_news", split="train")
    with open(output_txt, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
    print(f"Training data written to {output_txt}")

def train_tokenizer(
    train_file: str,
    vocab_size: int,
    min_frequency: int,
    output_dir: str
):
    """
    Train a Byte-Pair Encoding (BPE) tokenizer on the given text file.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[train_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )
    tokenizer.save_model(output_dir)
    print(f"Tokenizer saved to {output_dir}/")

def demo_tokenizer(tokenizer_dir: str, examples: list):
    """
    Load the trained tokenizer and demonstrate tokenization.
    """
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )
    for text in examples:
        encoded = tokenizer.encode(text)
        print(f"Input: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs:    {encoded.ids}")
        print("-" * 40)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and demo a BPE tokenizer on AG News")
    parser.add_argument(
        "--vocab_size", type=int, default=30_000,
        help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--min_frequency", type=int, default=2,
        help="Minimum frequency for a pair to be merged"
    )
    parser.add_argument(
        "--output_dir", type=str, default="tokenizer",
        help="Directory to save the trained tokenizer"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    train_txt = "ag_news_train.txt"
    download_and_prepare_data(train_txt)
    train_tokenizer(train_txt, args.vocab_size, args.min_frequency, args.output_dir)

    # Demo on a few example sentences
    examples = [
        "Stocks rallied on Wall Street after positive earnings reports.",
        "The new movie has breathtaking visual effects and strong performances.",
        "Local elections are coming up next month; make sure to register."
    ]
    demo_tokenizer(args.output_dir, examples)

if __name__ == "__main__":
    main()
