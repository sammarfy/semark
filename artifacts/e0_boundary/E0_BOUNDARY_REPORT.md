# E0-e Boundary/Context Report (Q10)

- criterion: median cosine >= 0.9990 AND median norm_l2 <= 0.0100 from b onward
- **b_context = None** frames (threshold not met at tested distances — report full curve)
- compares suffix re-encodes (offset k*hop, hop=1920) to the full encode; no DTW; alignment lag verified ~0 (frame-synchronous).
- the ~29-frame chunk seam is a *chunk-split* artifact; b_context is the NORMAL-encode startup length. Use b_context (not 29) for Stage-1 interior masks / prompt length (§7.1).
