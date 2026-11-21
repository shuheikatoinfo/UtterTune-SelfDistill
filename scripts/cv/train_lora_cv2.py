"""Data-free self-distill LoRA training for CV2 (UtterTune notation), ported to the MAIN CosyVoice
repo so it runs on the RTX 5090 (CosyVoice/.venv, torch cu128; the UtterTune venv's torch 2.3.1 is
sm_120-incompatible). Logic mirrors UtterTune scripts/cv2/train.py: register <PHON_START>/<PHON_END>,
LoRA on q/k/v/o_proj, unfreeze only the 2 new embed rows (grad-masked), CE loss via Qwen2LM.forward,
save adapter + embed_patch.safetensors. Trains on our self-distilled manifest (phon_text -> tokens).
"""
import sys, os, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
from pathlib import Path

CV_ROOT = (os.environ.get("CV_ROOT") or paths.COSYVOICE_DIR)
sys.path.insert(0, CV_ROOT)
sys.path.insert(0, os.path.join(CV_ROOT, "third_party/Matcha-TTS"))

import numpy as np, torch
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--base", default=f"{CV_ROOT}/pretrained_models/CosyVoice2-0.5B")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=64)
ap.add_argument("--val_ratio", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed)
rng = torch.Generator().manual_seed(args.seed)

cv2 = CosyVoice2(model_dir=args.base, fp16=False)
base_model = cv2.model.llm  # Qwen2LM

tok = get_qwen_tokenizer(token_path=f"{args.base}/CosyVoice-BlankEN", skip_special_tokens=True)
NEW = ["<PHON_START>", "<PHON_END>"]
tok.tokenizer.add_special_tokens({"additional_special_tokens": NEW})
tok.special_tokens["additional_special_tokens"].extend(
    [t for t in NEW if t not in tok.special_tokens["additional_special_tokens"]])
base_model.llm.model.resize_token_embeddings(len(tok.tokenizer))
new_ids = tok.tokenizer.convert_tokens_to_ids(NEW)
print(f"added {NEW} -> ids {new_ids}, vocab {len(tok.tokenizer)}", flush=True)

llm = get_peft_model(base_model, LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

emb = llm.base_model.llm.model.get_input_embeddings()
emb.weight.requires_grad_(True)
existing_mask = torch.ones(emb.num_embeddings, dtype=torch.bool, device=emb.weight.device)
existing_mask[new_ids] = False  # only the 2 new rows keep gradient
def mask_grad(grad):
    grad[existing_mask] = 0
    return grad
emb.weight.register_hook(mask_grad)

cv2.model.llm = llm
cv2.frontend.tokenizer = tok  # same lru-cached tok, but set explicitly


class TSV(Dataset):
    def __init__(self, path):
        self.rows = []
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            _, txt, npy, wav = ln.split("\t")
            self.rows.append((txt, torch.from_numpy(np.load(npy)).long()))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def collate(batch):
    texts, sp = zip(*batch)
    tl = [tok.encode(t, allowed_special=tok.special_tokens) for t in texts]
    pad_id = tok.encode("<|endoftext|>", allowed_special=tok.special_tokens)[0]
    L = max(len(x) for x in tl)
    tt = torch.full((len(tl), L), pad_id, dtype=torch.long)
    for i, ids in enumerate(tl):
        tt[i, :len(ids)] = torch.tensor(ids)
    return {"text_token": tt, "text_token_len": torch.tensor([len(x) for x in tl], dtype=torch.int32),
            "speech_token": pad_sequence(sp, batch_first=True, padding_value=0),
            "speech_token_len": torch.tensor([t.size(0) for t in sp], dtype=torch.int32)}


class CV2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        d = model(inputs, self.args.device)
        return (d["loss"], d) if return_outputs else d["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        with torch.no_grad():
            d = model(inputs, self.args.device)
        return (d["loss"], None, None)


ds = TSV(args.manifest)
n_val = max(1, int(len(ds) * args.val_ratio))
train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=rng)
print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

trainer = CV2Trainer(
    model=llm,
    args=TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
        max_steps=args.steps, learning_rate=args.lr, adam_beta1=0.9, adam_beta2=0.98,
        warmup_ratio=0.05, lr_scheduler_type="cosine", max_grad_norm=1.0, fp16=True,
        logging_steps=25, eval_strategy="steps", eval_steps=max(50, args.steps // 4),
        save_steps=args.steps, seed=args.seed, report_to="none"),
    train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
trainer.train()
llm.save_pretrained(args.out)
save_file({"embed_rows": emb.weight.detach().cpu()[new_ids]}, str(Path(args.out) / "embed_patch.safetensors"))
print(f"SAVED adapter + embed_patch -> {args.out}", flush=True)
