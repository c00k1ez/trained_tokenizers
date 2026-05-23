import click
import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from pathlib import Path


def dataset_iterator(dataset, batch_size=100):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]


@click.command()
def main():
    base_dir = Path("./trained_tokenizers")

    pretrain_dataset = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1"
    )
    # pretrain_dataset = pretrain_dataset.filter(lambda x: len(x["text"]) > 200)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=32_000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer.train_from_iterator(
        dataset_iterator(pretrain_dataset["train"]), trainer=trainer
    )

    tokenizers_dir = base_dir / "tokenizers"
    if not tokenizers_dir.exists():
        tokenizers_dir.mkdir()

    tokenizer.save(str(tokenizers_dir / "en_tokenizers_wikitext_32000.json"))


if __name__ == "__main__":
    main()
