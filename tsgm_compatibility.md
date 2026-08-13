# TSGM compatibility patch

## Installed version

`tsgm==0.1.0` (only PyPI release with a working PyTorch-backend `ConditionalGAN`
via Keras 3's `train_step_torch`; `0.0.1`-`0.0.7` predate this).

## Original error

```
ValueError: Invalid dtype: object
```
raised from `tsgm/models/cgan.py`, inside `ConditionalGAN.train_step_torch`,
at the call `ops.repeat(rep_labels, repeats=[self._seq_len])`, the first time
`cond_gan.fit(...)` is called under `KERAS_BACKEND=torch`.

## Why the patch is required

`keras.ops.repeat(x, repeats=[N])` (a Python list containing one int) is
accepted under the TensorFlow backend but not the PyTorch backend — under
`KERAS_BACKEND=torch` it raises `ValueError: Invalid dtype: object` while
trying to interpret the list as an array-like `repeats` argument. Verified in
isolation:

```python
import os; os.environ["KERAS_BACKEND"] = "torch"
import keras; from keras import ops
import torch
x = torch.randn(4, 2, 1)
ops.repeat(x, repeats=[10])            # ValueError: Invalid dtype: object
ops.repeat(x, repeats=10, axis=-1)     # OK, shape (4, 2, 10)
```

`repeats=N, axis=-1` (scalar int + explicit axis) is accepted by both
backends and produces the same result the code already relies on downstream
(`rep_labels` reshaped to `(-1, seq_len, output_dim)` on the next line). This
is a bug in TSGM's own code — it happens to work only because TSGM's test
suite and CI exercise the TensorFlow backend, not the torch one — not a bug
in Keras or PyTorch, so neither of those is touched.

The identical line also exists in the neighboring `train_step_tf` (line 431)
but is not patched: we run exclusively under `KERAS_BACKEND=torch`, never
exercise `train_step_tf` in this environment, and have no TensorFlow
installation to verify a change against.

## File and line changed

`tsgm/models/cgan.py`, inside `ConditionalGAN.train_step_torch` (originally
lines 507-509):

```diff
--- a/tsgm/models/cgan.py
+++ b/tsgm/models/cgan.py
@@ -505,8 +505,12 @@
         batch_size = ops.shape(real_ts)[0]
         if not self._temporal:
             rep_labels = labels[:, :, None]
+            # PATCHED (tsgm_compatibility.patch): keras.ops.repeat(x, repeats=[N])
+            # raises "ValueError: Invalid dtype: object" under the torch backend
+            # (works under the tensorflow backend, which is why this shipped
+            # unnoticed). repeats=N, axis=-1 is equivalent and works on both.
             rep_labels = ops.repeat(
-                rep_labels, repeats=[self._seq_len]
+                rep_labels, repeats=self._seq_len, axis=-1
             )
         else:
             rep_labels = labels
```

Full diff also saved at `tsgm_compatibility.patch` in this repo root.

## How to reapply after reinstalling TSGM

```bash
source venv/bin/activate
pip install --no-deps --force-reinstall "tsgm==0.1.0"
patch -p1 -d venv/lib/python3.12/site-packages < tsgm_compatibility.patch
```

(`-p1` strips the leading `a/`/`b/` prefix; run from the repo root so the
relative `venv/lib/...` target resolves. Adjust the `python3.12` path
component if the venv's Python version differs.)

## Environment installation notes

Installed with `--no-deps` throughout to avoid pip resolving/upgrading any
existing dependency (see `tsgm_compatibility.patch`'s companion investigation
in this conversation for the full per-package dependency audit). Packages
added, none of which required changing an existing package's version:

```
tsgm==0.1.0
keras==3.15.1
antropy==0.1.6
dtaidistance==2.3.13
prettytable==3.16.0
absl-py==2.5.0
h5py==3.16.0
markdown-it-py==4.2.0
mdurl==0.1.2
ml-dtypes==0.5.4
namex==0.1.0
optree==0.19.1
pygments==2.20.0
rich==15.0.0
wcwidth==0.8.2
```

`KERAS_BACKEND=torch` must be set (as an environment variable, or
`os.environ["KERAS_BACKEND"] = "torch"` before `import keras`) for every
script that uses TSGM — Keras 3 reads this once at import time.

Installed footprint: ~53 MB (`du -sh` across the above packages'
`site-packages` entries), well under the 500 MB budget.
