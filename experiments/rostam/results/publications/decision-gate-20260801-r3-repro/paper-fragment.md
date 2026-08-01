<!-- generated: do not edit -->
# Validated Experiment Fragment

> **COMPLETENESS: COMPLETE** — 8/8 expected cells have selected successful attempts.

## Provenance

- Trusted join SHA-256: `7328f458e4e8bcc44a94c2b78bf9fa96c1eafdca093899825335be93331d3a3a`
- Campaigns: 1
  - Run `decision-gate-20260801-r3` / campaign `rostam-decision-gate`
    - Manifest: `1b1909b3e8561b035c0ee4f9e9c39e497c82d737f8a61e348b1cf6792d06e842`
    - Selection: `primary` (`c675e42e34e44794a5dd8f12d3d4f4214e440c8b27643b9b4101edb1625391a0`)
    - Completeness verdict: `6377a65bb05761427ae0f384f0ad07a14430285fc50e344155c44e4e3f11ce8d`
    - Repository commit: `4585318ac244f218e438de06c0ccd38c0c88cbbf`
- Verified raw archive: `urn:commcanary:sha256:285d1924e34b732467ed5918e407fe56dbde8e2e4226e1fb1279651d1fafb5f6` / `285d1924e34b732467ed5918e407fe56dbde8e2e4226e1fb1279651d1fafb5f6` (3747840 bytes)

## Validated aggregates

| workload | configuration | selected reps | median us | IQR us | cell IDs |
|---|---|---:|---:|---:|---|
| decision-gate | nccl-2.19.3-default | 1 | 1414.144000 | 53.760000 | c-decision-gate-nccl-2.19.3-defaul-r000000-04b66d1ac8fa34af |
| decision-gate | nccl-2.20.5-default | 1 | 1436.672000 | 14.336000 | c-decision-gate-nccl-2.20.5-defaul-r000000-425550b712a971b2 |
| decision-gate | nccl-2.20.5-ring-ll | 1 | 1341.440000 | 60.416000 | c-decision-gate-nccl-2.20.5-ring-l-r000000-df1a35c03a7aa345 |
| decision-gate | nccl-2.20.5-ring-ll128 | 1 | 1380.352000 | 42.496000 | c-decision-gate-nccl-2.20.5-ring-l-r000000-e4f1bba0e8b48bd7 |
| decision-gate | nccl-2.20.5-ring-simple | 1 | 1430.528000 | 38.400000 | c-decision-gate-nccl-2.20.5-ring-s-r000000-98aaed50efe13070 |
| decision-gate | nccl-2.20.5-tree-ll | 1 | 1570.304000 | 745.472000 | c-decision-gate-nccl-2.20.5-tree-l-r000000-b28a4f80bc168cca |
| decision-gate | nccl-2.20.5-tree-ll128 | 1 | 1462.272000 | 167.936000 | c-decision-gate-nccl-2.20.5-tree-l-r000000-2930799903e9a040 |
| decision-gate | nccl-2.20.5-tree-simple | 1 | 1618.432000 | 220.672000 | c-decision-gate-nccl-2.20.5-tree-s-r000000-ff520eefdb46b5c0 |

## Selected-cell trace

| cell ID | attempt | attempt record SHA-256 | environment SHA-256 | measurement SHA-256 |
|---|---|---|---|---|
| c-decision-gate-nccl-2.19.3-defaul-r000000-04b66d1ac8fa34af | a-000001 | `6cfdd138f304dec530c327f7d683ae3e05b1bb072264227daabb351d5441c91c` | `441284e9eabadecceeaf6f54393d668f65bf50c946e8a58a755bf3704a3617c5` | `7c6d2a4b479691110c894b3bf86586c9bf924a032bf0ba671a6df46693a42d63` |
| c-decision-gate-nccl-2.20.5-defaul-r000000-425550b712a971b2 | a-000001 | `c826d90ce4b9d97d56c705d850f829ee8799403b51f2ba8859a1a206d3163667` | `27502be4e6e429f4be2ec67b7e227daacd0ed3ecfc5a8e8592219f27b8bb81a1` | `c4d4ff87a7ee6e6694963e5577e096771a032bfb6a3ba91434bf3980075c11f4` |
| c-decision-gate-nccl-2.20.5-ring-l-r000000-df1a35c03a7aa345 | a-000001 | `9afa1c1f2c781b6827b40559037ee03f916a7d518f33c575284a8e600c3fe19b` | `b6f9ab673207094232297f5e2e5595fe18727c75ce471094c57bc5c599726e94` | `4a2273d55feb9aa85026f08caa3e124abaf5da9d45c389a41eed3c5be00fdb9b` |
| c-decision-gate-nccl-2.20.5-ring-l-r000000-e4f1bba0e8b48bd7 | a-000001 | `aa47bf4ab8abc1d2b131e159a8f9e0731547c935c65bb81c6bf827866964b63c` | `dd8d12ad4efd6e25425f5413cb241534ac1c5b2ce73801b0d0ce31b14434996d` | `516fdec9829b6600f71e512f64c00bff17c03b9174d2a1b78c0d7af7c1c39ab2` |
| c-decision-gate-nccl-2.20.5-ring-s-r000000-98aaed50efe13070 | a-000001 | `0bcef8c4e3b6c3e013c00dae021ce4ce9ad48d5574be676fe418f15187c0ba27` | `7639ff57558d7b4539167fe54460c26e87d6ce505ff66c0a86599dd5003ed656` | `ed26973d24fd769751eb311bf04eb07fd6063b7d59fc2ce8d8651cbf10b01d92` |
| c-decision-gate-nccl-2.20.5-tree-l-r000000-2930799903e9a040 | a-000001 | `1d0e9bd498b03f3e0108288e943934cd64c6c5ea327cd6cd947f013684d0f393` | `1f29f2b2f48478068ccc006d10f51b3c8e6c28d8790d262bdea339ca3fce8b3a` | `6e107693d6cb094050398353d4b916dcaf5c9c3d9515a7433fc957eb99180cee` |
| c-decision-gate-nccl-2.20.5-tree-l-r000000-b28a4f80bc168cca | a-000001 | `4b779d4030e00c35903de5ef25cf5f33a6e589c390b96b83e0b80c2758fe21c8` | `e133ce1d60d87c7b4c94aa21ddf218ab4dfef86a62cf7e0949288228b8d3d8e4` | `bd69c3bc14b5673eefdda717e17adf3b8c9c6e28c7398aa9b8e43c98c0a598f7` |
| c-decision-gate-nccl-2.20.5-tree-s-r000000-ff520eefdb46b5c0 | a-000001 | `80f675c32836e7abb0a3d76f91c21a70532a851e6e4f30bd39b143bca2735083` | `8d7dc98ba640648d450a8f74b7bbe61e9754c8ff1984b94583151b2fc38f7023` | `77d856985751709638ee461b35fc712668f4ee86bead262618892b0ccad23549` |

## Failure and retry accounting

- Terminal attempts: 8
- Retries preserved: 0
- Unselected terminal attempts: 0
- Status counts: `{"cancelled":0,"excluded":0,"failed":0,"parse-failed":0,"success":8}`

## Claims

No Rostam ranking claim is applicable because this complete evidence set does not contain W-full.

## Exact regeneration command

```sh
python -m experiments.rostam.analyze verify --run-directory /home/aagrawal/CommCanary/experiments/rostam/results/decision-gate-20260801-r3 --selection-id primary --verdict-sha256 6377a65bb05761427ae0f384f0ad07a14430285fc50e344155c44e4e3f11ce8d --output-directory /home/aagrawal/CommCanary/experiments/rostam/results/publications/decision-gate-20260801-r3-primary --archive-descriptor /home/aagrawal/commcanary-transfer/decision-gate-20260801/decision-gate-20260801-r3-archive-descriptor.json --raw-archive /home/aagrawal/commcanary-transfer/decision-gate-20260801/decision-gate-20260801-r3-primary-6377a65b.tar --median-threshold-pct 8.0 --median-absolute-threshold-us 1.0
```
