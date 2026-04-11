"""Temporary verification script for parameter alignment."""
import sys
sys.path.insert(0, r'e:\chenshiyang\IMDD_transformer\scripts')
from train_kan_ideas import MODEL_TABLE, BASE, _fcnn_param_count, MATCH_PARAMS

target = _fcnn_param_count(BASE['window_size'], BASE['fcnn_hidden_dims'])
print(f"FCNN target params: {target:,}")
print(f"MATCH_PARAMS: {MATCH_PARAMS}")
print()

for k, v in MODEL_TABLE.items():
    model = v['build_fn'](v['config'], 'cpu')
    p = sum(pp.numel() for pp in model.parameters())
    print(f"  {v['display']:17s}  params={p:>6,}  (delta={p - target:+d})")
    del model

print("\nAll 7 models built successfully!")
