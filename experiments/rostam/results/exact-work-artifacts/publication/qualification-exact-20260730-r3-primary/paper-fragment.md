<!-- generated: do not edit -->
# Validated Experiment Fragment

> **COMPLETENESS: COMPLETE** — 1/1 expected cells have selected successful attempts.

## Provenance

- Trusted join SHA-256: `fe7167045d918b08ba90746afa79c52014bdad6173898843150948f38e304ec8`
- Campaigns: 1
  - Run `qualification-exact-20260730-r3` / campaign `rostam-qualification-exact`
    - Manifest: `c0bc2e6a9ecb1da691a3caa083bd5842f054c770300a855a9ac69500a2834552`
    - Selection: `primary` (`bf23480f39dd49794aa3d81b23a13d5576388563aeef12e8e77681877dd4ce9b`)
    - Completeness verdict: `593772c0902024443dd3e457f0e59003eea41a59014821417f629f5a34ff97b8`
    - Repository commit: `9b1e58aef91eebc14a65a11cc46e2eb500d814b5`
- Verified raw archive: `urn:commcanary:sha256:92ddc56f7ecbe5a7b162a7df749eeb01f4fd343eeab224be3c4f9d9996453bbe` / `92ddc56f7ecbe5a7b162a7df749eeb01f4fd343eeab224be3c4f9d9996453bbe` (532480 bytes)

## Validated aggregates

| workload | configuration | selected reps | median us | IQR us | cell IDs |
|---|---|---:|---:|---:|---|
| qualification-exact | nccl-2.20.5-default | 1 | 1541.001500 | 8.972500 | c-qualification-exac-nccl-2.20.5-defaul-r000000-65c2ef2ea97d8adc |

## Selected-cell trace

| cell ID | attempt | attempt record SHA-256 | environment SHA-256 | measurement SHA-256 |
|---|---|---|---|---|
| c-qualification-exac-nccl-2.20.5-defaul-r000000-65c2ef2ea97d8adc | a-000001 | `43c41dac3423bfe5dfb7339842cb8cfd1b858559541565453e6302557ea55c79` | `bd59edf260e34008b166f2aedcb4f4bfdcbc864ef0c95f4f4c971835e90b2d82` | `96215e760dcf0efaf487ae6800292545071793f5ef028670ff180d6d7275a83d` |

## Failure and retry accounting

- Terminal attempts: 1
- Retries preserved: 0
- Unselected terminal attempts: 0
- Status counts: `{"cancelled":0,"excluded":0,"failed":0,"parse-failed":0,"success":1}`

## Claims

No Rostam ranking claim is applicable because this complete evidence set does not contain W-full.

## Exact regeneration command

```sh
/home/aagrawal/venvs/commcanary-dev/bin/python -m experiments.rostam.analyze verify --run-directory experiments/rostam/results/qualification-exact-20260730-r3 --selection-id primary --verdict-sha256 593772c0902024443dd3e457f0e59003eea41a59014821417f629f5a34ff97b8 --output-directory /home/aagrawal/commcanary-artifacts/exact-work-52bfebe-20260730/publication/qualification-exact-20260730-r3-primary --archive-descriptor /home/aagrawal/commcanary-artifacts/exact-work-52bfebe-20260730/raw/qualification-exact-20260730-r3-archive-descriptor.json --raw-archive /home/aagrawal/commcanary-artifacts/exact-work-52bfebe-20260730/raw/qualification-exact-20260730-r3-primary-593772c0.tar
```
