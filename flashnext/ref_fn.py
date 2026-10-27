"""Flash-Next (qwen4exp) oracle: the official transformers Qwen4Exp modules in float64, weights dequantized from the GGUF.

No FAIE2 code: weights come from llama.cpp's gguf-py dequantizer (plus Q2_0, written from ggml-quants.c), the math is
transformers' modeling_qwen4_exp.py unchanged. Routed experts and the 51.2B-parameter n-gram table are fetched lazily,
only the rows a forward touches. GGUF conventions undone here: RMSNorm weights are stored as 1 + w, ssm_a = -exp(A_log),
and (llama.cpp converter) DeltaNet value heads are stored tiled [v][k] instead of grouped [k][v].

usage: ref_fn.py nll                      -> teacher-forced NLL of a known sentence under both head orders
       ref_fn.py logits <ids.json> <out.npy> [reorder]  -> float64 log-probs at every position (teacher forced)
"""

import json
import sys
import time

import numpy as np
import torch
from torch import nn

# transformers first: faie2.gguf_reader installs a stub `yaml` module that breaks huggingface_hub's import
import transformers.models.qwen4_exp.modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

sys.path.insert(0, r"E:\ABU\FAIE2\tests")
sys.path.insert(0, r"E:\ABU\FAIE2\serve")
from faie2 import gguf_reader  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SH1 = r"E:\ABU\MODELS\llm\qwen3.8-flash-next-abliterated\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf"
SH2 = SH1.replace("00001-of", "00002-of")
CFG = r"E:\ABU\FAIE2\flashnext\config_text.json"

r1 = gguf_reader(SH1)
r2 = gguf_reader(SH2)
from gguf import quants  # noqa: E402  (gguf-py is on sys.path after gguf_reader)
from gguf.constants import GGMLQuantizationType as QT  # noqa: E402

T = {t.name: t for t in r1.tensors}
NG = r2.tensors[0]
f64 = torch.float64
torch.set_num_threads(16)


def dequant(data, qtype):
    data = np.asarray(data)
    if qtype == QT.Q2_0:  # ggml-quants.c dequantize_row_q2_0: 64 values, fp16 d, 2-bit codes, value = (q - 1) * d
        lead = data.shape[:-1]
        blk = data.reshape(-1, 18)
        d = blk[:, :2].copy().view(np.float16).astype(np.float64)
        qs = blk[:, 2:]
        q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(-1, 64).astype(np.float64)
        return ((q - 1.0) * d).reshape(*lead, -1)
    return quants.dequantize(data, qtype).astype(np.float64)


def W(name, data=None):
    t = T[name]
    return torch.from_numpy(np.ascontiguousarray(dequant(t.data if data is None else data, t.tensor_type)))


def untile(x, dim, hd, nk=16, nv=3):
    """GGUF tiled V-head order [v][k][hd] -> HF grouped order [k][v][hd] along `dim`."""
    shape = list(x.shape)
    new = shape[:dim] + [nv, nk, hd] + shape[dim + 1:]
    x = x.reshape(new)
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return x.permute(perm).contiguous().reshape(shape)


class LazyExperts(nn.Module):
    """Qwen3NextExperts with expert matrices dequantized on demand from the GGUF (same forward arithmetic)."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.act_fn = nn.SiLU()
        self.layer = None
        self.calls = 0

    def forward(self, hidden_states, top_k_index, top_k_weights):
        p = f"blk.{self.layer}."
        tg, tu, td = T[p + "ffn_gate_exps.weight"], T[p + "ffn_up_exps.weight"], T[p + "ffn_down_exps.weight"]
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            e = int(expert_idx)
            self.calls += 1
            g = W(tg.name, np.asarray(tg.data)[e])
            u = W(tu.name, np.asarray(tu.data)[e])
            d = W(td.name, np.asarray(td.data)[e])
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, g), nn.functional.linear(current_state, u)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, d)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class LazyRows(nn.Module):
    """The n-gram embedding table, read row by row from shard 2 (IQ4_NL, 160 values per row)."""

    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.register_buffer("weight", torch.zeros(1, dtype=f64), persistent=False)

    def forward(self, ids):
        flat = ids.reshape(-1).cpu().numpy()
        rows = np.asarray(NG.data)[flat]
        out = torch.from_numpy(np.ascontiguousarray(dequant(rows, NG.tensor_type)))
        return out.reshape(*ids.shape, self.embedding_dim)


def build(reorder):
    cfg = Qwen4ExpTextConfig(**json.load(open(CFG)))
    cfg._attn_implementation = "eager"
    torch.set_default_dtype(f64)
    M.Qwen4ExpTextExperts = LazyExperts
    M.Qwen4ExpPreTrainedModel._init_weights = lambda self, module: None
    real_embedding = torch.nn.Embedding
    torch.nn.Embedding = lambda n, d, *a, **k: LazyRows(n, d) if n > 50_000_000 else real_embedding(n, d, *a, **k)
    try:
        model = M.Qwen4ExpForCausalLM(cfg)
    finally:
        torch.nn.Embedding = real_embedding
    model.eval()
    t0 = time.time()
    sd = {}

    def put(name, value):
        sd[name] = value

    put("model.embed_tokens.weight", W("token_embd.weight"))
    put("lm_head.weight", W("output.weight"))
    mix = "model.hyper_connection_mixer."
    put(mix + "hc_norm.weight", W("output_hc_norm.weight") - 1)
    put(mix + "input_mix_weight_down.weight", W("output_hc_down.weight"))
    put(mix + "input_mix_weight_up.weight", W("output_hc_up.weight"))
    for i in range(cfg.num_hidden_layers):
        g, h = f"blk.{i}.", f"model.layers.{i}."
        model.model.layers[i].mlp.experts.layer = i
        for hc, src in (("attn_hyper_connection.", "hc_attn_"), ("mlp_hyper_connection.", "hc_ffn_")):
            put(h + hc + "hc_norm.weight", W(g + src + "norm.weight") - 1)
            put(h + hc + "input_mix_weight_down.weight", W(g + src + "down.weight"))
            put(h + hc + "input_mix_weight_up.weight", W(g + src + "up.weight"))
            put(h + hc + "block_inject_weight.weight", W(g + src + "inject.weight"))
        if cfg.layer_types[i] == "linear_attention":
            la = h + "linear_attn."
            qkv, z = W(g + "attn_qkv.weight"), W(g + "attn_gate.weight")
            a, b = W(g + "ssm_alpha.weight"), W(g + "ssm_beta.weight")
            conv = W(g + "ssm_conv1d.weight")  # (10240, 4)
            dt, ssm_a = W(g + "ssm_dt.bias"), W(g + "ssm_a")
            out = W(g + "ssm_out.weight")  # (2560, 6144)
            if reorder:
                qkv = torch.cat([qkv[:4096], untile(qkv[4096:], 0, 128)])
                z = untile(z, 0, 128)
                a, b = untile(a, 0, 1), untile(b, 0, 1)
                conv = torch.cat([conv[:4096], untile(conv[4096:], 0, 128)])
                dt = untile(dt.unsqueeze(-1), 0, 1).squeeze(-1)
                ssm_a = untile(ssm_a.unsqueeze(-1), 0, 1).squeeze(-1)
                out = untile(out, 1, 128)
            put(la + "in_proj_qkv.weight", qkv)
            put(la + "in_proj_z.weight", z)
            put(la + "in_proj_a.weight", a)
            put(la + "in_proj_b.weight", b)
            put(la + "conv1d.weight", conv.unsqueeze(1))
            put(la + "dt_bias", dt)
            put(la + "A_log", torch.log(-ssm_a))
            put(la + "norm.weight", W(g + "ssm_norm.weight"))
            put(la + "out_proj.weight", out)
        else:
            sa = h + "self_attn."
            put(sa + "q_proj.weight", W(g + "attn_q.weight"))
            put(sa + "k_proj.weight", W(g + "attn_k.weight"))
            put(sa + "v_proj.weight", W(g + "attn_v.weight"))
            put(sa + "o_proj.weight", W(g + "attn_output.weight"))
            put(sa + "q_norm.weight", W(g + "attn_q_norm.weight") - 1)
            put(sa + "k_norm.weight", W(g + "attn_k_norm.weight") - 1)
            put(sa + "indexer.index_qk_proj.weight", torch.cat([W(g + "indexer.q_proj.weight"), W(g + "indexer.k_proj.weight")]))
            put(sa + "indexer.q_layernorm.weight", W(g + "indexer.q_norm.weight") - 1)
            put(sa + "indexer.k_layernorm.weight", W(g + "indexer.k_norm.weight") - 1)
        put(h + "mlp.gate.weight", W(g + "ffn_gate_inp.weight"))
        put(h + "mlp.shared_expert.gate_proj.weight", W(g + "ffn_gate_shexp.weight"))
        put(h + "mlp.shared_expert.up_proj.weight", W(g + "ffn_up_shexp.weight"))
        put(h + "mlp.shared_expert.down_proj.weight", W(g + "ffn_down_shexp.weight"))
        put(h + "mlp.shared_expert_gate.weight", W(g + "ffn_gate_inp_shexp.weight").reshape(1, -1))
        if model.model.layers[i].ple is not None:
            pl = h + "ple."
            put(pl + "key_proj.weight", W(g + "ple_key.weight"))
            put(pl + "value_proj.weight", W(g + "ple_value.weight"))
            put(pl + "norm_key.weight", W(g + "ple_norm_key.weight") - 1)
            put(pl + "norm_query.weight", W(g + "ple_norm_query.weight") - 1)
            put(pl + "norm_conv.weight", W(g + "ple_norm_conv.weight") - 1)
            put(pl + "conv1d.weight", W(g + "ple_conv1d.weight").unsqueeze(1))
    params = dict(model.named_parameters())
    missing = sorted(set(params) - set(sd))
    extra = sorted(set(sd) - set(params))
    if missing or extra:
        raise SystemExit(f"weight map mismatch: missing {missing[:8]} extra {extra[:8]}")
    with torch.no_grad():
        for name, value in sd.items():
            if tuple(params[name].shape) != tuple(value.shape):
                raise SystemExit(f"{name}: model {tuple(params[name].shape)} vs gguf {tuple(value.shape)}")
            params[name].copy_(value)
    print(f"loaded {len(sd)} tensors in {time.time() - t0:.0f} s (reorder={reorder})", flush=True)
    return model


@torch.no_grad()
def logprobs(model, ids):
    out = model(input_ids=torch.tensor([ids]), use_cache=False)
    return torch.log_softmax(out.logits[0].to(f64), dim=-1)


def main():
    mode = sys.argv[1]
    if mode == "nll":
        from faie2_tok import QwenTokenizer
        tok = QwenTokenizer(SH1)
        text = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome."
        ids = tok.encode(text)
        print("ids", ids, flush=True)
        for reorder in (1, 0):
            model = build(reorder)
            t0 = time.time()
            lp = logprobs(model, ids)
            nll = -lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]
            top = lp.argmax(-1).tolist()
            print(f"reorder={reorder}: mean NLL {nll.mean():.3f} nats, forward {time.time() - t0:.0f} s, "
                  f"expert loads {sum(l.mlp.experts.calls for l in model.model.layers)}", flush=True)
            print("  predicted next tokens:", repr(tok.decode(top)), flush=True)
            del model
    elif mode == "logits":
        ids = json.load(open(sys.argv[2]))
        reorder = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        model = build(reorder)
        hidden = []
        for layer in model.model.layers:
            layer.register_forward_hook(lambda m, a, out: hidden.append(out[0].detach().numpy().astype(np.float32)))
        t0 = time.time()
        lp = logprobs(model, ids)
        np.save(sys.argv[3], lp.numpy())
        np.save(sys.argv[3].replace(".npy", "_hidden.npy"), np.stack(hidden))  # [NL][n][HCD]
        print(f"forward {len(ids)} tokens in {time.time() - t0:.0f} s -> {sys.argv[3]}", flush=True)


if __name__ == "__main__":
    main()
