"""Data-free UtterTune LoRA on Sarashina — clean-room rebuild.

Copy of the recovered original (durable: $DATA/code/train_lora.py;
this repo's scripts/train_lora.py is the later revision of the same file adding --alpha,
which the saved accent adapters require: lora_faithful2_tf457 has lora_alpha=64 = 4*rank,
i.e. was trained with --alpha 64). Changes vs. that original, and NOTHING else:
  1. Dead scratchpad paths -> parameterized: --model-dir defaults to
     the durable handoff mirror $DATA/_handoff/sara-tts-weights;
     sys.path uses $SARASHINA_DIR (the sarashina2.2-tts code); --teachers/--out default to the clean-room
     rebuild locations.
  2. Checkpoint log line says "SAVED adapter <dir>" (the CV2 chain-script convention that
     j5_after_retrain.sh / v3_onmethod_chain.sh grep for) instead of "saved <dir>".

Method (verbatim original): student = <|pron_start|>{accented-kana}<|pron_end|> carrier;
label = base LLM's own semantic-token sequence for the plain carrier (teachers jsonl).
Train LoRA on q/k/v/o + ONLY the 2 pron-token input-embedding rows (grad-masked, mean-init).
Freeze everything else; flow.pt/hift.pt never touched (zero timbre drift by construction).
CE on semantic-token positions only. Disclosed uniform teacher-length filter 20<=n_sem<=600.
Saves ckpt_{step}/ with the peft adapter + pron_emb.pt {"pron_start","pron_end","emb"[2,H]}.
Round-3 recipe: --steps 20000 --alpha 64 --save-every 4000 (rank 16, lr 1e-4,
bs 8 x accum 2 defaults).
"""
import sys, os, json, time, math, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/, for paths.py
import paths
sys.path.insert(0, paths.SARASHINA_DIR)
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from sarashina_tts.additional_tokens import SPEECH_START_TOKEN, PRON_START_TOKEN, PRON_END_TOKEN, PAD_TOKEN

ap = argparse.ArgumentParser()
ap.add_argument("--teachers", required=True, help="teacher jsonl (e.g. teachers/sarashina/ of the released dataset)")
ap.add_argument("--out", required=True)
ap.add_argument("--model-dir", required=True, help="local copy of sbintuitions/sarashina2.2-tts")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--save-every", type=int, default=150)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=0)  # 0 -> 2*rank
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

mdir = args.model_dir
tok = AutoTokenizer.from_pretrained(mdir)
if tok.pad_token is None:
    tok.pad_token = PAD_TOKEN
SEM_BASE = tok.convert_tokens_to_ids("<|semantic_0|>")
PRON_S = tok.convert_tokens_to_ids(PRON_START_TOKEN)
PRON_E = tok.convert_tokens_to_ids(PRON_END_TOKEN)
EOS = tok.eos_token_id or 2
PAD = tok.pad_token_id

rows = [json.loads(l) for l in open(args.teachers)]
rows = [r for r in rows if 20 <= r.get("n_sem", 0) <= 600]
print("train rows:", len(rows), flush=True)

def build(r):
    prompt = tok(r["phon"] + SPEECH_START_TOKEN, add_special_tokens=True)["input_ids"]
    sem = [SEM_BASE + s for s in r["sem"]]
    ids = prompt + sem + [EOS]
    labels = [-100] * len(prompt) + sem + [EOS]
    return torch.tensor(ids), torch.tensor(labels)
data = [build(r) for r in rows]

model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16).cuda()
model.config.use_cache = False
# mean-init the 2 untrained pron rows from text-token embeddings (avoid dead-token start)
emb0 = model.get_input_embeddings().weight
with torch.no_grad():
    mv = emb0[:100000].float().mean(0).to(emb0.dtype)
    emb0[PRON_S].copy_(mv); emb0[PRON_E].copy_(mv)

lcfg = LoraConfig(r=args.rank, lora_alpha=(args.alpha or 2*args.rank), lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lcfg)

# train ONLY the 2 pron input-embedding rows: enable grad + mask all other rows to zero
in_emb = model.get_input_embeddings()
in_emb.weight.requires_grad_(True)
freeze_mask = torch.ones(in_emb.weight.shape[0], dtype=torch.bool, device=in_emb.weight.device)
freeze_mask[PRON_S] = False; freeze_mask[PRON_E] = False
in_emb.weight.register_hook(lambda g: g.masked_fill(freeze_mask.unsqueeze(1), 0.0))

trainable = [p for p in model.parameters() if p.requires_grad]
n_tr = sum(p.numel() for p in trainable)
print(f"trainable params: {n_tr:,} (LoRA + 2 pron emb rows)", flush=True)
opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
def lr_at(step):
    if step < args.warmup:
        return args.lr * step / max(1, args.warmup)
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

def batches():
    order = list(range(len(data)))
    rng = random.Random(0)
    while True:
        rng.shuffle(order)
        for i in range(0, len(order), args.bs):
            yield [data[j] for j in order[i:i + args.bs]]

def collate(b):
    ids = pad_sequence([x[0] for x in b], batch_first=True, padding_value=PAD)
    lab = pad_sequence([x[1] for x in b], batch_first=True, padding_value=-100)
    att = (ids != PAD).long()
    return ids.cuda(), lab.cuda(), att.cuda()

def save(tag):
    d = os.path.join(args.out, f"ckpt_{tag}")
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)  # LoRA adapter
    torch.save({"pron_start": PRON_S, "pron_end": PRON_E,
                "emb": in_emb.weight.data[[PRON_S, PRON_E]].cpu().clone()},
               os.path.join(d, "pron_emb.pt"))
    print("SAVED adapter", d, flush=True)

model.train()
gen = batches()
t0 = time.time(); run_loss = 0.0; nl = 0
for step in range(1, args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    opt.zero_grad(set_to_none=True)
    for _ in range(args.accum):
        ids, lab, att = collate(next(gen))
        out = model(input_ids=ids, attention_mask=att, labels=lab)
        (out.loss / args.accum).backward()
        run_loss += out.loss.item(); nl += 1
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    if step % 20 == 0:
        print(f"step {step}/{args.steps} loss {run_loss/nl:.4f} lr {lr_at(step):.2e} {time.time()-t0:.0f}s", flush=True)
        run_loss = 0.0; nl = 0
    if step % args.save_every == 0 or step == args.steps:
        save(str(step))
print("TRAIN_DONE", flush=True)
