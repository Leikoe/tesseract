# Shared dense decoder extraction benchmark

Revision `9f3f2917704a0ba68c829fd5fd6dfd2565ebd8ce` on an
NVIDIA A100-SXM4-80GB with CUDA 13.3 and Llama-3.2-1B-Instruct.

The first run reached 1,516.85 completion tokens/s for the first-shape workload
and 1,556.21 tokens/s for the identical warm-shape workload. An immediate repeat
reached 1,598.17 and 1,590.03 tokens/s. The repeat is retained because this
structural extraction does not change kernel selection, launches, graph shapes,
or batching, and the node has previously produced runs between roughly 1,586
and 1,720 tokens/s for this exact workload. The lower result is recorded as
unresolved node/run variance rather than presented as a performance win.
