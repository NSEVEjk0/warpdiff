# WarpDiff · Vast Showcase

[Public benchmark report](https://warpdiff.vercel.app/)

[Pull request and results in Checks (private)](https://github.com/NSEVEjk0/warpdiff/pull/1)

[Demo screenshot](https://warpdiff.vercel.app/demo.png)

The benchmark tested commit `8a2b4c98ac54df066330360c46a2ba392ff2641c`.

## What the run shows

Both versions matched PyTorch on both GPUs. With an `8,192 × 1,024` input, the patch was **7.63% slower on the RTX 4090** and **2.95% slower on the A100 SXM4**. The A100 result was within the 5% cutoff. The `257 × 1,000` input slowed down beyond the cutoff on both GPUs, so the pull request check failed.

The two GPUs logged **62 minutes of active benchmark time**. Vast billed **$0.572** for those runs and **$0.683** for all four rentals, including the first test. All four rentals have been deleted.

[A100 run](https://github.com/NSEVEjk0/warpdiff/actions/runs/45895403901) · [RTX 4090 run and combined report](https://github.com/NSEVEjk0/warpdiff/actions/runs/45899888445)

## Entry requirements

Complete the Vast entry form using the same account email that rented the GPUs. Publish the demo screenshot or a short recording on X, Discord #general, or r/vastai. Include @vast_ai and #BuiltOnVast and keep the entry public through December 4. Use the same platform on the form and in the official post.

The deadline for both is Sunday, November 29, 2026 at 11:59 p.m. PST, which is Monday, November 30 at 07:59 UTC or 8:59 a.m. WAT. Include the public report URL and identify the tested commit. Vast requires at least one active GPU-hour on the account, excluding idle rental time.

[Official entry form and rules](https://vast.ai/showcase#rules)

## What to show

1. Open the report on the `8,192 × 1,024` input. Show the original and patched timings and where the 4090 slowed down beyond the cutoff.
2. Switch to `257 × 1,000` to show a slowdown on both cards. Point out the GPU time and the charges at the bottom of the report.
3. If your audience has access to the repository, open the pull request’s **Checks** tab. Anyone can download the timings from the public report.

Credit the tools you used. WarpDiff rented the GPUs from Vast.ai, ran a Triton softmax kernel in the PyTorch CUDA image, and checked the outputs against PyTorch. The JSON download has the individual timings, software versions, and Vast charges.

Add your own handle to the entry form and disclose any material contributors. Keep your Vast account email and API credentials private.
