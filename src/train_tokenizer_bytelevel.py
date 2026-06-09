import click
import datasets
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from pathlib import Path


def dataset_iterator(dataset, batch_size=100):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]


@click.command()
@click.option("--vocab_size", default=8_000, type=int)
def main(vocab_size: int):
    base_dir = Path("./trained_tokenizers")

    pretrain_dataset = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1"
    )

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        dataset_iterator(pretrain_dataset["train"]), trainer=trainer
    )

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    tokenizers_dir = base_dir / "tokenizers"
    if not tokenizers_dir.exists():
        tokenizers_dir.mkdir()

    tokenizer.save(str(tokenizers_dir / f"en_tokenizers_wikitext_{vocab_size}.json"))


if __name__ == "__main__":
    main()
