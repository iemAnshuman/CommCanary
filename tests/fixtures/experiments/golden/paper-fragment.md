<!-- generated: do not edit -->
# Validated Experiment Fragment

> **COMPLETENESS: COMPLETE** — 4/4 expected cells have selected successful attempts.

## Provenance

- Trusted join SHA-256: `094487bbc9fcc91dde5d80f41eb74a787dd7a310c344b1132e0771306aa55a07`
- Campaigns: 1
  - Run `local-analysis-golden` / campaign `local-analysis`
    - Manifest: `75a62666035d757f86d6ae3d095972b55745f92d9192a404697b8d77028abad8`
    - Selection: `primary` (`73dfc9c3315d0c268545155a8895c4b2058ad9929ae62e894573f637c3944b6c`)
    - Completeness verdict: `bcfe1e272233e7e56bff575b040c8cb741709af6e62ce1dca17564e30057056e`
    - Repository commit: `1111111111111111111111111111111111111111`

## Validated aggregates

| workload | configuration | selected reps | median us | IQR us | cell IDs |
|---|---|---:|---:|---:|---|
| prepare | config-a | 2 | 11.000000 | 2.000000 | c-prepare-config-a-r000000-cfd2412a858f21a1, c-prepare-config-a-r000001-c25f3fedf89b8e25 |
| prepare | config-b | 2 | 21.000000 | 2.000000 | c-prepare-config-b-r000000-9f74b8ece2ce00db, c-prepare-config-b-r000001-90c1a0af23e3bc4d |

## Selected-cell trace

| cell ID | attempt | attempt record SHA-256 | environment SHA-256 | measurement SHA-256 |
|---|---|---|---|---|
| c-prepare-config-a-r000000-cfd2412a858f21a1 | a-000002 | `f8f2cf3ebe30d959e506fa7114a96a6ab2003eb9ca800c3d82ea9e91b55426cd` | `e67d9fea7b03e1c46b4c75418f410021ac91cd9ee429344957246442b65bf898` | `e9a89c68167e2406319acdf45d0d111421eb0efa5384161b451330525c5f7309` |
| c-prepare-config-a-r000001-c25f3fedf89b8e25 | a-000001 | `2005f92393c89cff3099ab3c4316baa6c9648d11bccfe85e2e5a79153d6f1af6` | `e67d9fea7b03e1c46b4c75418f410021ac91cd9ee429344957246442b65bf898` | `343d34a3a1c24ffb72042a44b30022d8796b9a40ea1d670fa005eeaa1e7842f0` |
| c-prepare-config-b-r000000-9f74b8ece2ce00db | a-000001 | `69e5c0d43f028f2af478f9998c2afc4491337a084827aa0cda31fe32c8a0fe24` | `34818b26b37c335e974852f6b5c509a1b3aba72bb4c692705d3ba298abd02a5c` | `72009a4d9ada1db0b5472f1c5afcc291b2b0a2d7711e8e0fca6991e181d1cc25` |
| c-prepare-config-b-r000001-90c1a0af23e3bc4d | a-000001 | `3baa14443898235a7200cb83bf7f1752b494ed99587204e97ae0c81b2afd2023` | `34818b26b37c335e974852f6b5c509a1b3aba72bb4c692705d3ba298abd02a5c` | `b716682c338593c5bb4a779f587bd3d477c0ee109ae4edb6523b28b3c03b6d6f` |

## Failure and retry accounting

- Terminal attempts: 5
- Retries preserved: 1
- Unselected terminal attempts: 1
- Status counts: `{"cancelled":0,"excluded":0,"failed":1,"parse-failed":0,"success":4}`

## Claims

No Rostam ranking claim is applicable because this complete evidence set does not contain W-full.

## Exact regeneration command

```sh
python -m experiments.rostam.analyze verify --fixture local-analysis-golden
```
