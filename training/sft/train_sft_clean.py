import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.utils.checkpoint  # chunked CE recomputes logits in backward
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_clean_sharegpt_dataset(json_path: str, tokenizer, max_length: int = 8192) -> Dataset:
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} samples from {json_path}")

    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    assistant_tok = tokenizer.encode("assistant", add_special_tokens=False)[0]

    all_input_ids = []
    all_labels = []

    for item in raw_data:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue

        messages = [
            {"role": "system", "content": item.get("system", "You are a professional Text-to-SQL agent specialized in SQLite databases.")},
        ]
        for c in convs:
            role = "user" if c["from"] == "human" else "assistant"
            messages.append({"role": role, "content": c["value"]})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, max_length=max_length, truncation=True, padding=False, return_tensors=None)
        input_ids = tokens["input_ids"]
        labels = [-100] * len(input_ids)

        # Mask labels: only calculate loss for assistant tokens between `<|im_start|>assistant\n` and `<|im_end|>`
        i = 0
        n = len(input_ids)
        while i < n - 2:
            if input_ids[i] == im_start and input_ids[i + 1] == assistant_tok:
                # Skip past "\n"
                start_idx = i + 2
                while start_idx < n and input_ids[start_idx] in [198, 271]: # newline tokens
                    start_idx += 1

                # Find matching im_end
                end_idx = start_idx
                while end_idx < n and input_ids[end_idx] != im_end:
                    end_idx += 1

                # Assign labels for assistant output tokens including im_end
                for j in range(start_idx, min(end_idx + 1, n)):
                    labels[j] = input_ids[j]

                i = end_idx + 1
            else:
                i += 1

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels
    })
    logger.info(f"Processed {len(dataset)} tokenized examples with masked assistant-only labels.")
    return dataset

class ChunkedCETrainer(Trainer):
    """Trainer that computes lm_head + cross-entropy in sequence chunks.

    Why this is needed at 32k and not below
    ---------------------------------------
    ZeRO-3 shards parameters, gradients and optimizer state, but the logits
    tensor is neither a parameter nor sharded: with per-device batch 1 every GPU
    materialises the full logits for its own sequence, and
    ``ForCausalLMLoss`` upcasts them to fp32 first. For Qwen3's 151,936-token
    vocabulary that is

        32768 x 151936 x 4 bytes = 18.55 GiB

    for the tensor alone, plus another of the same shape for its gradient. Job
    377833 died asking for exactly 18.55 GiB with 14.67 GiB free of 79.18 GiB,
    which is why adding GPUs does not help -- the spike is per GPU per
    sequence, so 16 GPUs each still need those 18.55 GiB.

    Chunking the sequence and recomputing each chunk's logits in the backward
    pass (``checkpoint``) keeps one chunk live at a time: at CHUNK=2048 the
    same tensor is 1.16 GiB.

    Numerically this reproduces ``ForCausalLMLoss`` exactly: same right-shift,
    same fp32 upcast, same ``ignore_index``, and the same normalisation --
    ``reduction="sum"`` divided by ``num_items_in_batch`` when the Trainer
    supplies it, mean over valid tokens when it does not. Only the float
    summation order differs, so the 8k and 16k runs stay comparable to a run
    made with this path enabled; ``scripts/probe_chunked_ce.py`` asserts the
    two agree to within float tolerance before any long job is launched.
    """

    def __init__(self, *args, ce_chunk: int = 2048, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_chunk = ce_chunk

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        if return_outputs:
            # Only the training path is chunked; nothing here calls evaluate().
            return super().compute_loss(model, inputs, return_outputs=True,
                                        num_items_in_batch=num_items_in_batch,
                                        **kwargs)
        inputs = dict(inputs)
        labels = inputs.pop("labels")

        # Reach the HF model through DeepSpeed's / DDP's wrappers so the body
        # can be called without lm_head building full-sequence logits.
        core = model
        while hasattr(core, "module"):
            core = core.module
        body, lm_head = core.model, core.lm_head

        out = body(input_ids=inputs["input_ids"],
                   attention_mask=inputs.get("attention_mask"),
                   position_ids=inputs.get("position_ids"),
                   use_cache=False)
        hidden = out[0] if not hasattr(out, "last_hidden_state") else out.last_hidden_state

        # Same shift as ForCausalLMLoss: pad right with ignore_index, drop the
        # first column, so position t is scored against token t+1.
        shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
        h = hidden.reshape(-1, hidden.size(-1))
        y = shift.reshape(-1).to(h.device)

        n_valid = (y != -100).sum()
        denom = num_items_in_batch if num_items_in_batch is not None else n_valid
        if torch.is_tensor(denom):
            denom = denom.to(h.device)

        def chunk_ce(hc, yc):
            logits = lm_head(hc).float()
            if not (yc != -100).any():
                # cross_entropy over an all-ignored chunk is nan; contribute a
                # graph-attached zero instead. Crucially this still CALLS
                # lm_head -- see the loop below for why that matters.
                return logits.sum() * 0.0
            return torch.nn.functional.cross_entropy(
                logits, yc, ignore_index=-100, reduction="sum")

        # Every rank must invoke lm_head the SAME number of times.
        #
        # Under ZeRO-3 the lm_head weight is sharded, so each invocation costs
        # an all-gather in forward and a reduce-scatter of its gradient in
        # backward. Per-device batch size is 1 and the collator pads each
        # sample to a multiple of 8 of its own length, so sequence lengths --
        # and therefore chunk counts -- differ across ranks. A first version
        # looped over the local length and skipped all-ignored chunks; ranks
        # then issued different numbers of collectives and hung in
        # _REDUCE_SCATTER_BASE with NumelIn=777912320, which is exactly
        # 151936 x 5120, the lm_head weight.
        #
        # So take the max chunk count across ranks and loop that many times;
        # ranks past their own length pass a one-token all-ignored slice, which
        # still calls lm_head (aligning the collectives) and adds zero.
        n_chunks = (h.size(0) + self.ce_chunk - 1) // self.ce_chunk
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([n_chunks], device=h.device, dtype=torch.long)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            n_chunks = int(t.item())

        dummy_y = torch.full((1,), -100, dtype=y.dtype, device=y.device)
        total = None
        for i in range(n_chunks):
            start = i * self.ce_chunk
            if start < h.size(0):
                hc, yc = h[start:start + self.ce_chunk], y[start:start + self.ce_chunk]
            else:
                hc, yc = h[:1], dummy_y
            part = torch.utils.checkpoint.checkpoint(
                chunk_ce, hc, yc, use_reentrant=False)
            total = part if total is None else total + part

        if total is None:
            return hidden.sum() * 0.0
        return total / denom


def main():
    required = ("MODEL_PATH", "DATA_PATH", "OUTPUT_DIR")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing required environment variables: {', '.join(missing)}")
    model_path = os.environ["MODEL_PATH"]
    data_path = os.environ["DATA_PATH"]
    output_dir = os.environ["OUTPUT_DIR"]
    deepspeed_config = os.environ.get("DEEPSPEED_CONFIG", "")
    num_epochs = float(os.environ.get("NUM_EPOCHS", "3.0"))
    lr = float(os.environ.get("LR", "1e-5"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "2"))
    max_length = int(os.environ.get("MAX_LENGTH", "8192"))
    max_steps = int(os.environ.get("MAX_STEPS", "-1"))  # -1 = disabled
    ce_chunk = int(os.environ.get("CE_CHUNK", "0"))  # 0 = stock HF loss
    # A ZeRO-3 mid-training checkpoint carries fp32 optimizer state: roughly
    # 45G for an 8B model and 79G for a 14B model. Runs that only need final
    # weights can drop optimizer state, reducing checkpoints to the bf16 model
    # (roughly 16G / 28G). This forfeits mid-run resume, so it is opt-in.
    save_only_model = os.environ.get("SAVE_ONLY_MODEL", "0") == "1"
    # Liger's fused_linear_cross_entropy is the preferred fix for the 32k
    # logits spike: it fuses lm_head with the CE and never materialises the
    # full logits, and unlike a chunked loop it calls lm_head exactly once, so
    # no ZeRO-3 collective can desynchronise across ranks. Requires
    # PYTHONPATH to include the --target install (see the 32k sbatch).
    use_liger = os.environ.get("USE_LIGER", "0") == "1"
    save_total_limit = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))

    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    train_dataset = load_clean_sharegpt_dataset(data_path, tokenizer, max_length=max_length)
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,  # -1 = disabled; set via MAX_STEPS env for OOM probes
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        save_only_model=save_only_model,
        bf16=True,
        fp16=False,
        deepspeed=deepspeed_config if os.path.exists(deepspeed_config) else None,
        use_liger_kernel=use_liger,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    if use_liger and ce_chunk > 0:
        raise SystemExit("set either USE_LIGER=1 or CE_CHUNK>0, not both: "
                         "both replace the loss and the result would be ambiguous")
    if ce_chunk > 0:
        logger.info(f"chunked cross-entropy ON, chunk={ce_chunk} "
                    f"(fp32 logits per chunk: "
                    f"{ce_chunk * model.config.vocab_size * 4 / 2**30:.2f} GiB "
                    f"vs {max_length * model.config.vocab_size * 4 / 2**30:.2f} GiB unchunked)")
        trainer = ChunkedCETrainer(ce_chunk=ce_chunk, **trainer_kwargs)
    else:
        logger.info("stock HF loss (CE_CHUNK unset)")
        trainer = Trainer(**trainer_kwargs)
    logger.info(f"save_only_model={save_only_model} save_total_limit={save_total_limit} use_liger={use_liger}")

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training and saving complete!")

if __name__ == "__main__":
    main()
