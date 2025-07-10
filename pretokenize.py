import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from rich.progress import track

import json
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def encode(tokenizer, text, add_special_tokens=False, add_end_of_seq=False):
    text = text + tokenizer.eos_token + "\n" if add_end_of_seq else text
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def preprocess_pretraining(example, tokenizer: AutoTokenizer):
    tokens = encode(tokenizer, example["text"].strip())
    tokens.append(tokenizer.eos_token_id)
    return {"input_ids": tokens}


def load_chunked_data(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    basename = metadata["basename"]
    num_chunks = metadata["num_chunks"]

    output_dir = Path(metadata_path).parent

    for chunk_idx in range(num_chunks):
        input_ids_path = output_dir / f"{basename}_tokenized_chunk_{chunk_idx:04d}.npy"

        input_ids_chunk = np.memmap(input_ids_path, dtype=np.uint32, mode="r")

        label_mask_chunk = None

        yield input_ids_chunk, label_mask_chunk


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if args.eos_token:
        tokenizer.eos_token_id = args.eos_token
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    basename = Path(args.input_jsonl).stem

    log.info(f"Streaming and tokenizing JSONL file: {args.input_jsonl}")
    pool = Pool(processes=args.num_procs)
    process_fn = partial(preprocess_pretraining, tokenizer=tokenizer)

    MAX_CHUNK_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
    TOKENS_PER_CHUNK = MAX_CHUNK_SIZE_BYTES // 4  # 4 bytes per uint32 token

    total_tokens = 0
    chunk_idx = 0
    current_chunk_tokens = 0
    current_chunk_data = []

    created_files = []

    def save_chunk(chunk_data, chunk_tokens, chunk_index):
        if chunk_tokens == 0:
            return

        output_path = str(
            output_dir / f"{basename}_tokenized_chunk_{chunk_index:04d}.npy"
        )

        input_ids_file = np.memmap(
            output_path, dtype=np.uint32, mode="w+", shape=(chunk_tokens,)
        )

        offset = 0
        for ex in chunk_data:
            ex_len = len(ex["input_ids"])
            input_ids_file[offset : offset + ex_len] = ex["input_ids"]
            offset += ex_len

        input_ids_file.flush()

        created_files.append(output_path)

        log.info(
            f"Saved chunk {chunk_index} with {chunk_tokens:,} tokens to {output_path}"
        )

    for result in track(
        pool.imap(process_fn, read_jsonl(args.input_jsonl), chunksize=100),
        description="Tokenizing and chunking",
    ):
        if result is None:
            continue

        result_tokens = len(result["input_ids"])

        if (
            current_chunk_tokens + result_tokens > TOKENS_PER_CHUNK
            and current_chunk_data
        ):
            save_chunk(current_chunk_data, current_chunk_tokens, chunk_idx)

            chunk_idx += 1
            current_chunk_data = []
            current_chunk_tokens = 0

        current_chunk_data.append(result)
        current_chunk_tokens += result_tokens
        total_tokens += result_tokens

    if current_chunk_data:
        save_chunk(current_chunk_data, current_chunk_tokens, chunk_idx)

    metadata = {
        "total_tokens": total_tokens,
        "num_chunks": chunk_idx + 1 if current_chunk_data or chunk_idx > 0 else 0,
        "max_seq_len": args.seq_len,
        "basename": basename,
        "files": created_files,
    }

    metadata_path = output_dir / f"{basename}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Total tokens: {total_tokens:,}")
    log.info(f"Created {metadata['num_chunks']} chunks")
    log.info(f"Metadata saved to: {metadata_path}")
    log.info("Done processing JSONL.")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Tokenize dataset to numpy memory-mapped files")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        help="""Jsonl dataset file""",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="""Directory to save the memory-mapped numpy files to""",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="""Tokenizer path or HuggingFace identifier""",
        required=True,
    )
    parser.add_argument(
        "--eos_token",
        type=int,
        help="""EOS token ID If not set, picks up from tokenizer.""",
        default=None,
    )
    parser.add_argument(
        "--num_procs", type=int, help="""Number of workers.""", default=8
    )
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())