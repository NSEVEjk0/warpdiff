# WarpDiff

WarpDiff benchmarks a Triton kernel before and after a change. It rents one Vast.ai GPU per model, tests both versions on each GPU, and posts the timings under **Checks** on the pull request.

## Run on GitHub

1. Add a scoped Vast.ai API key as the Actions secret `VASTAI_API_KEY`. Grant `misc`, `instance_read`, `instance_write`, and `billing_read` so WarpDiff can find and rent GPUs, delete the rentals, and look up charges.
2. Open a pull request to `main` that changes `kernels/softmax.py`. Use a branch in this repository, such as `demo/softmax-tiling`.
3. In **Actions → WarpDiff → Run workflow**, choose `main` and enter the pull request number. Select `review` for a short run or `showcase` for at least 31 minutes of active time on each GPU.
4. On the pull request, open **Checks → WarpDiff / GPU performance**. The workflow run has the raw timings and any charges Vast has posted.

If only one GPU finishes a showcase run, retry the missing GPU with `showcase-4090` or `showcase-a100`. Enter the first run’s ID in `comparison_run`. WarpDiff combines the timings if both runs used the same original and patched kernel code.

The workflow runs the benchmark code from `main` and sends the two kernel files to the rented GPUs. Run it only for pull requests you trust. The kernel must implement the FP32 softmax function `softmax(source, output)`.

## Run locally

You need Python 3.11 or later, Git, OpenSSH, and a Vast.ai API key. You can run these commands on a machine without a GPU. The first command lists offers without renting a GPU.

```sh
python3 -m warpdiff compare --base main --head demo/softmax-tiling --dry-run
python3 -m warpdiff compare --base main --head demo/softmax-tiling --max-hourly 0.85
```

The command writes `artifacts/report.json` and prints a Markdown summary. Exit code `0` means the check passed. Code `2` means a regression or incorrect output. Code `3` means WarpDiff could not get a clear result. The default cutoff is 5%.

WarpDiff saves rental IDs under `.warpdiff/`. If a local run stops before it deletes the rentals, run `python3 -m warpdiff cleanup --state-dir .warpdiff`. If an Actions run stops early, look for the `warpdiff-<run-id>-<attempt>-card-*` rentals in the Vast console. Run `python3 -m warpdiff billing --report artifacts/report.json` to fetch updated charges.

## How the check works

On each GPU, WarpDiff checks both versions against PyTorch. It uses CUDA events to time pairs of CUDA graph replays and randomizes which version goes first. For each input size, it reports the median percentage change and a 95% bootstrap confidence interval. With the default 5% cutoff, the whole interval must be above +5% to report a slowdown or below −5% to report an improvement. WarpDiff calls the result inconclusive when the interval crosses a cutoff. Results depend on the GPU, driver, and input size.

The hourly price comes from the Vast offer. Vast posts the actual charges separately, and the final amount can take a few minutes to appear.
