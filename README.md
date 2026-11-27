# WarpDiff

WarpDiff runs a repo’s GPU benchmark on the original and patched code. It rents the GPU models you choose from Vast.ai, checks both outputs, and posts the timings under **Checks** on the pull request.

[Softmax report](https://warpdiff.vercel.app/?benchmark=softmax) · [RMSNorm report](https://warpdiff.vercel.app/?benchmark=rmsnorm) · [SwiGLU report](https://warpdiff.vercel.app/?benchmark=swiglu)

The softmax check caught a regression on the RTX 4090. The RMSNorm check measured an improvement on the RTX 3090 and RTX 4090. Both tests ran on Vast.ai. [Softmax pull request](https://github.com/NSEVEjk0/warpdiff/pull/1/checks) · [RMSNorm pull request](https://github.com/NSEVEjk0/warpdiff/pull/2/checks)

## Run a pull request in this repo

1. Add a scoped Vast.ai API key as the Actions secret `VASTAI_API_KEY`. Grant `misc`, `instance_read`, `instance_write`, and `billing_read` so the action can find GPUs, rent them, delete the rentals, and read charges.
2. Open a pull request to the default branch from a branch in this repo. Change `kernels/softmax.py` or `kernels/rmsnorm.py`.
3. In **Actions → WarpDiff → Run workflow**, enter the PR number and pick `softmax` or `rmsnorm`. Use `review` for a short run. The `showcase` profile measures at least 31 active minutes on each configured GPU.
4. Open **Checks → WarpDiff / GPU performance** on the PR. The workflow run has the raw timings and the charges Vast has posted.

Use the optional `gpu` and `comparison_run` inputs to retry a missing GPU and combine it with an earlier report. The runs must use the same source files, test definition, and cutoff. Use `max_hourly` to cap the quoted hourly rate for each rental.

## Use WarpDiff in another repo

The example repo uses the released action to test a SwiGLU change on Vast.ai. It has a small benchmark definition, a PyTorch reference, and a real pull request with a passing check. [Example repo](https://github.com/NSEVEjk0/warpdiff-triton-example) · [Pull request](https://github.com/NSEVEjk0/warpdiff-triton-example/pull/1/checks)

Add `warpdiff.toml` and a benchmark adapter to your default branch. Add the scoped secret described above, then copy the following workflow into `.github/workflows/warpdiff.yml`. The action reads the config and adapter from the checked-out default branch. Use it for pull requests from branches in your repo whose code you trust.

```yaml
name: GPU benchmark
on:
  workflow_dispatch:
    inputs:
      pull_request:
        description: Pull request number
        type: number
        required: true
      benchmark:
        description: Benchmark in warpdiff.toml
        type: string

permissions:
  actions: read
  contents: read
  pull-requests: read
  checks: write

jobs:
  benchmark:
    runs-on: ubuntu-24.04
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
        with:
          fetch-depth: 0
          persist-credentials: false
      - uses: NSEVEjk0/warpdiff@v0.2.1
        with:
          pull_request: ${{ inputs.pull_request }}
          benchmark: ${{ inputs.benchmark }}
          vast_api_key: ${{ secrets.VASTAI_API_KEY }}
```

## Define a benchmark

This example selects two GPU models and a case for a kernel and adapter in the repo. Add more cases to test different sizes or pass extra settings to the adapter.

```toml
default = "rmsnorm"

[benchmarks.rmsnorm]
name = "Triton RMSNorm"
kernel = "kernels/rmsnorm.py"
adapter = "benchmarks/rmsnorm.py"
gpus = ["RTX 3090", "RTX 4090"]
margin_pct = 5
cases = [
  { id = "decode", label = "Decode · 64 × 4,096", shape = [64, 4096], epsilon = 1e-6 },
]
```

The adapter exports `prepare(case)`. It creates the input and returns the expected CUDA tensor plus a function called `bind`. WarpDiff calls `bind` once for each Git revision and passes it that revision’s kernel module. Each call returns a zero-argument benchmark function and its own preallocated output tensor. See the [RMSNorm adapter](benchmarks/rmsnorm.py) and the [two benchmark definitions](warpdiff.toml).

List any Python helpers under `files` and import them with relative imports so each revision loads its own copy. Set `rtol` and `atol` for the reference comparison when necessary. The workload must run in the pinned PyTorch and Triton image and support CUDA graph capture. The action sends the listed files and trusted adapter to the GPU worker. The Vast API key and GitHub token stay on the Actions runner.

## Run locally

You need Python 3.11 or later, Git, and OpenSSH. The first command checks the config and both revisions on a machine without a GPU or API key.

```sh
python3 -m warpdiff compare --base main --head demo/rmsnorm-fusion --benchmark rmsnorm --validate-only
python3 -m warpdiff compare --base main --head demo/rmsnorm-fusion --benchmark rmsnorm --max-hourly 0.85
```

The command writes `artifacts/report.json` and prints a Markdown summary. Exit code `0` means the check passed. Code `2` means a regression or incorrect output. Code `3` means the run was inconclusive or unavailable.

WarpDiff saves rental IDs under `.warpdiff/`. If a local run stops early, use `python3 -m warpdiff cleanup --state-dir .warpdiff`. For an interrupted Actions run, look up its `warpdiff-<run-id>-<attempt>-card-*` labels in the Vast console. Run `python3 -m warpdiff billing --report artifacts/report.json` for updated charges.

## How the check works

WarpDiff checks both outputs against the adapter’s reference. It uses CUDA events to time CUDA graph replays, pairs measurements on the same GPU, and randomizes which version goes first. It reports the median percentage change and a 95% bootstrap confidence interval. With a 5% cutoff, the whole interval must be above +5% to report a slowdown or below −5% to report an improvement. An interval that crosses a cutoff yields an inconclusive result.
