# Frozen reference

`FrozenDFMReference(model)` copies the trained score and fixes `T` and `t0`; input gradients remain available.
`score(x, s)` takes forward OU time `s` in `[t0, T]`; reverse generation uses `s = T - t` and stops at `t = T - t0`.
This slice adds score freezing only. Time-dependent coordinates, full-portfolio conditioning and conditional initialization follow in later PRs.
