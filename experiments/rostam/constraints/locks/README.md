# Reviewed Rostam locks

This directory intentionally contains no installable lock yet. The two files
below must be generated on the target interpreter/platform, reviewed, and then
committed before `setup.sh` can run:

- `nccl-2.19.3.lock.txt`
- `nccl-2.20.5.lock.txt`

Every non-comment requirement line must be exact and carry one or more
`--hash=sha256:...` values. The reviewed `environment-contract.json` records
the SHA-256 of each complete lock, the resolver invocation, the interpreter and
platform tags, the downloaded wheel inventory, and `pip freeze --all` output.

NCCL 2.19.3 is a deliberate override of torch 2.4.1's native NCCL dependency.
Both complete locks are therefore installed with `--no-deps --require-hashes`:
every transitive wheel must be enumerated and reviewed, and pip must not try to
replace the reviewed 2.19.3 wheel with torch's declared 2.20.5 dependency. The
resolver report must retain the native solution and explicitly document this
single reviewed substitution; it must not pretend the combined set has a
normal resolver solution.
