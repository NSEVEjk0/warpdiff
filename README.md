# WarpDiff

WarpDiff checks whether a Triton change that passes correctness tests also holds up on the GPUs people use. Run it against a pull request to rent one Vast.ai GPU per model, measure both revisions on each rental, and attach the result to the candidate commit.

## Run on GitHub

1. Add a scoped Vast.ai API key as the Actions secret `VASTAI_API_KEY`. Grant `misc`, `instance_read`, `instance_write`, and `billing_read` for offer search, worker lifecycle, and receipts.
2. Open a same-repository PR to `main` that changes `kernels/softmax.py`. The demo branch is `demo/softmax-tiling`.
3. In **Actions → WarpDiff → Run workflow**, use `main`, enter the PR number, and select `review`. `showcase` runs additional iterations for at least 31 CUDA-timed minutes on each GPU. Both profiles rent real GPUs.
4. Open **Checks → WarpDiff / GPU performance** on the PR. The linked workflow run contains the raw timing samples and the provider receipt when available.

If one GPU finishes a showcase run and the other is unavailable, select the profile for that missing GPU and enter the earlier workflow's run ID under **comparison_run**. The check combines measurements only when the baseline and candidate kernel source match. The report links back to the original runs.

The workflow reads the benchmark harness from `main` and sends the two kernel files to the disposable workers. It accepts open PRs with trusted contributors who can push branches to the repository. Only dispatch a run for code you trust. The supported workload is the included FP32 softmax interface `softmax(source, output)`.

## Run locally

Requires Python 3.11 or later, Git, OpenSSH, and a Vast.ai API key. The controller can run on a CPU. Try the first command to list offers without renting.

```sh
python3 -m warpdiff compare --base main --head demo/softmax-tiling --dry-run
python3 -m warpdiff compare --base main --head demo/softmax-tiling --max-hourly 0.85
```

The CLI writes `artifacts/report.json` and prints a Markdown summary. Exit `0` means no regression beyond the threshold was detected, `2` means a regression or correctness failure, and `3` means an inconclusive or unavailable result. The default regression threshold is 5%.

Rental state is stored under `.warpdiff/`. If the controller stops before confirming cleanup, run `python3 -m warpdiff cleanup --state-dir .warpdiff`. For an interrupted Actions run, use its run ID and attempt to find the `warpdiff-<run-id>-<attempt>-card-*` labels in Vast. To refresh a provider receipt, run `python3 -m warpdiff billing --report artifacts/report.json`.

## How the check works

Each worker checks the baseline and candidate against PyTorch, then alternates their CUDA graph timings in randomized order on the same device. WarpDiff computes the median of the paired latency changes and a 95% bootstrap interval per input shape. It flags a regression when the entire interval is above +5%, an improvement when it is below −5%, and reports inconclusive when it crosses the tolerance boundary. Findings apply to the measured driver, devices, and input shapes. The quoted rental rate is an estimate for comparing hardware. The receipt comes from Vast's per-instance charges endpoint and can lag behind teardown.
