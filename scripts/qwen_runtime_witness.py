"""Verify real Qwen3 generation and a seeded LoRA CUDA optimizer step."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def run(report):
    import accelerate
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    started = time.time()
    spec = json.loads((ROOT / 'experiments/llm_extraction/environment.json').read_text())
    spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    versions = {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'accelerate': accelerate.__version__}
    if any(versions[key] != spec[key] for key in versions):
        raise ValueError('Installed versions differ from the declared runtime.')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError('The declared V100 CUDA device is unavailable.')
    model_manifest = json.loads((ROOT / 'experiments/llm_extraction/model.json').read_text())
    if digest(ROOT / 'experiments/llm_extraction/model.json') != spec['model_manifest_sha256']:
        raise ValueError('The model manifest changed.')
    for filename, entry in model_manifest['files'].items():
        path = ROOT / spec['model_path'] / filename
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Model file verification failed: {filename}')
    print(json.dumps({'event': 'model_files_verified', 'seconds': time.time() - started}), flush=True)
    torch.manual_seed(20260929)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                        attn_implementation='eager')
    tiny = get_peft_model(Qwen3ForCausalLM(config).half(), LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).cuda()
    optimizer = torch.optim.AdamW([parameter for parameter in tiny.parameters() if parameter.requires_grad], lr=1e-3)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device='cuda')
    tracked = next(parameter for name, parameter in tiny.named_parameters() if 'lora_B' in name)
    previous = tracked.detach().clone()
    with torch.autocast('cuda', dtype=torch.float16):
        loss = tiny(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(tiny.parameters(), 1.0)
    if not torch.isfinite(loss) or not torch.isfinite(grad):
        raise FloatingPointError('Nonfinite Qwen3 LoRA kernel witness.')
    optimizer.step()
    if torch.equal(previous, tracked):
        raise RuntimeError('LoRA optimizer did not update its trainable weights.')
    kernel = {'loss': float(loss.detach()), 'gradient_norm': float(grad), 'optimizer_updated': True,
              'trainable_parameters': sum(parameter.numel() for parameter in tiny.parameters() if parameter.requires_grad)}
    del tiny, optimizer, tracked, previous, ids, loss
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'event': 'lora_kernel_passed', **kernel}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / spec['model_path'], local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(ROOT / spec['model_path'], local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation=spec['attention'])
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != model_manifest['parameter_count'] or parameter_count > 10_000_000_000:
        raise ValueError('Actual model parameter count differs from the declared competition limit.')
    source = 'The child had short stature and recurrent seizures.'
    messages = [{'role': 'system', 'content': 'Return only a JSON array of exact phenotype phrases quoted from the supplied sentence. Copy the original spelling exactly. Do not explain.'},
                {'role': 'user', 'content': source}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[:, -1]
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Real Qwen3 weights produced nonfinite logits in FP16.')
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    phrases = json.loads(response)
    if not isinstance(phrases, list) or not phrases or any(not isinstance(phrase, str) or not phrase or phrase not in source for phrase in phrases):
        raise ValueError('Real Qwen3 failed exact-source JSON generation.')
    result = {'status': 'passed', 'spec_sha256': spec_sha, 'versions': versions, 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
              'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
              'lora_kernel': kernel, 'real_model_parameter_count': parameter_count,
              'synthetic_source': source, 'response': phrases, 'exact_source_json_passed': True,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3, 'elapsed_seconds': time.time() - started,
              'scope': 'Runtime and format acceptance only; no competition accuracy claim.'}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('QWEN_CUDA_WITNESS ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args().report)
