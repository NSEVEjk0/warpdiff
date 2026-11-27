# WarpDiff · Vast Showcase

[Softmax report](https://warpdiff.vercel.app/?benchmark=softmax) · [Softmax check](https://github.com/NSEVEjk0/warpdiff/pull/1/checks)

[RMSNorm report](https://warpdiff.vercel.app/?benchmark=rmsnorm) · [RMSNorm check](https://github.com/NSEVEjk0/warpdiff/pull/2/checks)

[Demo screenshot](https://warpdiff.vercel.app/demo.png)

The softmax benchmark tested commit `8a2b4c98ac54df066330360c46a2ba392ff2641c`. The RMSNorm benchmark tested `3b2835e7df8f73e9511dfe4d3f6e0cac59919769`.

## What the runs show

**Softmax.** Both versions matched PyTorch on both GPUs. On the `8,192 × 1,024` input, the patch was **7.63% slower on the RTX 4090** and **2.95% slower on the A100 SXM4**. The A100 result stayed within the 5% cutoff. The `257 × 1,000` input slowed down beyond the cutoff on both GPUs. The two showcase rentals logged **62 minutes of active benchmark time** and cost **$0.572**. [A100 run](https://github.com/NSEVEjk0/warpdiff/actions/runs/45895403901) · [RTX 4090 run](https://github.com/NSEVEjk0/warpdiff/actions/runs/45899888445)

**RMSNorm.** The configuration selected three cases, an RTX 3090, and an RTX 4090. Both versions matched PyTorch for all three inputs on both GPUs. Fusing the reduction and scaling lowered kernel time by **21–33%**, and the check passed. The first RTX 3090 offer expired before rental. WarpDiff reused the 4090 timings on the retry and verified the source and benchmark definition before combining the results. [Config](https://github.com/NSEVEjk0/warpdiff/blob/main/warpdiff.toml) · [Combined RMSNorm run](https://github.com/NSEVEjk0/warpdiff/actions/runs/46041898731)

Both reports include the raw timings and rental charges. Vast confirmed the benchmark rentals were deleted.

## Entry requirements

Complete the Vast entry form using the same account email that rented the GPUs. Publish the demo screenshot or a short recording on X, Discord #general, or r/vastai. Include @vast_ai and #BuiltOnVast and keep the entry public through December 4. Use the same platform on the form and in the official post.

The deadline for both is Sunday, November 29, 2026 at 11:59 p.m. PST, which is Monday, November 30 at 07:59 UTC or 8:59 a.m. WAT. Include the public report URL and identify the tested commit. Vast requires at least one active GPU-hour on the account, excluding idle rental time.

[Official entry form and rules](https://vast.ai/showcase#rules)

## What to show

1. Open Softmax on the `8,192 × 1,024` input. Show where the patch crossed the cutoff on the RTX 4090. Open the pull request check.
2. Switch to RMSNorm. Show the config’s GPU choices and cases, then open the passing pull request check for the fused kernel.
3. Show where another repo can add its own benchmark and call the same action. Both reports have a JSON download with the individual timings.

Credit the tools you used. WarpDiff rented GPUs from Vast.ai, ran the Triton kernels in the PyTorch CUDA image, and checked the outputs against PyTorch.

Add your own handle to the entry form and disclose any material contributors. Keep your Vast account email and API credentials private.
