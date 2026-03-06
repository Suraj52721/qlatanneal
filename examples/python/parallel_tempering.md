# parallel_tempering.py

**Goal:** Run parallel tempering (replica exchange) on a simple chain Ising model.

**What it demonstrates**
- Multiple beta ladder
- Swap acceptance tracking
- Classical PT API (`ParallelTemperingAnnealer`)

For quantum PT, see:
- `examples/python/tuned_sqapt_demo.py`
- `solve(..., method=\"sqapt\")`

**Run**
```bash
python examples/python/parallel_tempering.py
```

**Expected output (example)**
```
Best energy: <value>
Swap acceptance (last): <value>
```
