<!-- generated: do not edit -->
# Validated Experiment Fragment

> **COMPLETENESS: COMPLETE** — 40/40 expected cells have selected successful attempts.

## Provenance

- Trusted join SHA-256: `4014b06b279f9e1366b0c06a300feeee014c008c525bdff0576121ed0580a123`
- Campaigns: 1
  - Run `shared-replay-20260720-r2` / campaign `rostam-shared-replay`
    - Manifest: `a402a4ec73ea3a182ab6bd5ec92e896600b16510dc4c1621c6defe9382ee149c`
    - Selection: `primary` (`a1e876861f7f9a315cb00a11a67414e6125078a68306c0b07a4a5d9f14b98d64`)
    - Completeness verdict: `b6cd1aae4cfb2de020a840f941031d4a910d0d47d4c83aee1c09e0f5f6bc98db`
    - Repository commit: `2855275288e67a1a2d0bbefff0740841fdf0ecf0`
- Verified raw archive: `urn:commcanary:raw-archive:sha256:3451ee540b634e1daad0aa49b6b95173fd601f5b6baea04deb6395d8d2c7b273` / `3451ee540b634e1daad0aa49b6b95173fd601f5b6baea04deb6395d8d2c7b273` (1057573 bytes)

## Validated aggregates

| workload | configuration | selected reps | median us | IQR us | cell IDs |
|---|---|---:|---:|---:|---|
| shared-overlap | nccl-2.19.3-default | 5 | 145.408000 | 1.024000 | c-shared-overlap-nccl-2.19.3-defaul-r000000-42c4fd5dc8399155, c-shared-overlap-nccl-2.19.3-defaul-r000001-ec0a9d0317449c37, c-shared-overlap-nccl-2.19.3-defaul-r000002-532e099a04649fdb, c-shared-overlap-nccl-2.19.3-defaul-r000003-80d86b1ffee4520c, c-shared-overlap-nccl-2.19.3-defaul-r000004-e0f7cb5304795a1c |
| shared-overlap | nccl-2.20.5-default | 5 | 146.432000 | 1.024000 | c-shared-overlap-nccl-2.20.5-defaul-r000000-5d5e01ebc348c344, c-shared-overlap-nccl-2.20.5-defaul-r000001-ccb4f09f1afdf6fd, c-shared-overlap-nccl-2.20.5-defaul-r000002-9b5b645247c1e801, c-shared-overlap-nccl-2.20.5-defaul-r000003-259c49e7dec26faa, c-shared-overlap-nccl-2.20.5-defaul-r000004-6eb5c721f2b4f791 |
| shared-overlap | nccl-2.20.5-ring-ll | 5 | 148.480000 | 2.560000 | c-shared-overlap-nccl-2.20.5-ring-l-r000000-d9bdab399d443def, c-shared-overlap-nccl-2.20.5-ring-l-r000001-dc31f4bf29b06b9f, c-shared-overlap-nccl-2.20.5-ring-l-r000002-ec14b2d92ba89caa, c-shared-overlap-nccl-2.20.5-ring-l-r000003-e1c764ca7331da9c, c-shared-overlap-nccl-2.20.5-ring-l-r000004-7f827b3a7fb84072 |
| shared-overlap | nccl-2.20.5-ring-ll128 | 5 | 147.456000 | 0.000000 | c-shared-overlap-nccl-2.20.5-ring-l-r000000-8b533675e3bed4dd, c-shared-overlap-nccl-2.20.5-ring-l-r000001-aa1bab7f5c353738, c-shared-overlap-nccl-2.20.5-ring-l-r000002-5c684e97077f6cd4, c-shared-overlap-nccl-2.20.5-ring-l-r000003-64978dd8aeb511b4, c-shared-overlap-nccl-2.20.5-ring-l-r000004-8ecdd84d4ec98d77 |
| shared-overlap | nccl-2.20.5-ring-simple | 5 | 146.432000 | 0.000000 | c-shared-overlap-nccl-2.20.5-ring-s-r000000-734baa18e69dd1e5, c-shared-overlap-nccl-2.20.5-ring-s-r000001-38fe374f0c08b167, c-shared-overlap-nccl-2.20.5-ring-s-r000002-8da3d12aa2dfed3d, c-shared-overlap-nccl-2.20.5-ring-s-r000003-cfaa86517b56ab2c, c-shared-overlap-nccl-2.20.5-ring-s-r000004-cba63dba98c72727 |
| shared-overlap | nccl-2.20.5-tree-ll | 5 | 176.640000 | 4.608000 | c-shared-overlap-nccl-2.20.5-tree-l-r000000-819055e6d6f02186, c-shared-overlap-nccl-2.20.5-tree-l-r000001-447ecbfdcea47636, c-shared-overlap-nccl-2.20.5-tree-l-r000002-bc27bf255f3750df, c-shared-overlap-nccl-2.20.5-tree-l-r000003-3116d2f8f7e4c16e, c-shared-overlap-nccl-2.20.5-tree-l-r000004-9c416bf50a13f1fb |
| shared-overlap | nccl-2.20.5-tree-ll128 | 5 | 156.672000 | 0.512000 | c-shared-overlap-nccl-2.20.5-tree-l-r000000-3b163d78cc344d60, c-shared-overlap-nccl-2.20.5-tree-l-r000001-b459b21bd3d56f23, c-shared-overlap-nccl-2.20.5-tree-l-r000002-66d998ba7dc14478, c-shared-overlap-nccl-2.20.5-tree-l-r000003-2705619f9e19a116, c-shared-overlap-nccl-2.20.5-tree-l-r000004-d56ff9108fc1ef6a |
| shared-overlap | nccl-2.20.5-tree-simple | 5 | 152.576000 | 0.000000 | c-shared-overlap-nccl-2.20.5-tree-s-r000000-c7551302dbed84dd, c-shared-overlap-nccl-2.20.5-tree-s-r000001-6af1f4bf7e007b20, c-shared-overlap-nccl-2.20.5-tree-s-r000002-bbcbe2a1c71451e1, c-shared-overlap-nccl-2.20.5-tree-s-r000003-38fead51278b729f, c-shared-overlap-nccl-2.20.5-tree-s-r000004-c0e6fd51ac42fca0 |

## Selected-cell trace

| cell ID | attempt | attempt record SHA-256 | environment SHA-256 | measurement SHA-256 |
|---|---|---|---|---|
| c-shared-overlap-nccl-2.19.3-defaul-r000000-42c4fd5dc8399155 | a-000001 | `9fa3c8c8be0d586dabe9531e036bcb8632b1de032fe46d3746dd7bada878c0fa` | `b8b98e8152cec37aeb7d8e5b8ed7aeb02809027a86327166b2df410c6ff04e25` | `8b5dd85d917e38b9d296579c7a98754b564e87e2b14536b29bc2adba875dd9dc` |
| c-shared-overlap-nccl-2.19.3-defaul-r000001-ec0a9d0317449c37 | a-000001 | `b07e51a54f49e803ad8de40d476bcc5d5fac96f5268b6167c910ebb4deac37e0` | `ca01e5bae54b7074529754a658e1f5fa591e900b894d2b942b0ba679beae7e39` | `dc30fc260c6441aee32a4a831f7aee900d6dae13a971b462faafdca694128c37` |
| c-shared-overlap-nccl-2.19.3-defaul-r000002-532e099a04649fdb | a-000001 | `d9eae2a449a791e466a9b6d2d7d9c2007aecf5a56f466e91b60eaf3901eb9cef` | `a8ed6d8a98d17c0a2658afaa4275d09e9a8905024ac0386bbd86fe46c6bb0691` | `09e3a67494584f37c6a7f9c02e46ba9f08165eb10fe7f9b9741f9b4a6abce9d6` |
| c-shared-overlap-nccl-2.19.3-defaul-r000003-80d86b1ffee4520c | a-000001 | `1d2ff34277abd115ada3a690758f4ea9b9005cfede87f3e55e1dbe6464540173` | `299917fe8ae550aa88d4921c3b3b09cad63bfe6dfa8ab2dec124f149071221b1` | `7311f86440ec6ab106f600b42a96a565b4aeba803d4e531697c567e3698dda31` |
| c-shared-overlap-nccl-2.19.3-defaul-r000004-e0f7cb5304795a1c | a-000001 | `d2da300309b0670ef9239402448a0cefbc810fa25bbc0c5f2db1442c7b75c001` | `27e75550d84c21211187bdaecd66a7ad9022273e98a6d6f37b6ae377727aaffe` | `62492c07b94ad5dac3cde9a6de53ef11d0c88dcefa4720436a0f509a08e83304` |
| c-shared-overlap-nccl-2.20.5-defaul-r000000-5d5e01ebc348c344 | a-000001 | `d87b2e1fa8be3bf5604ef0d9640c6cf372cec88054edf56913a914ce29d68c1f` | `dea81e34739af5ee01492ddbd97fa72be3d5ac518d5c8854fd00d5aaf66c6934` | `182def6229190200113263ad5070e618c8bd1a52a25fc570a85aec36267dee17` |
| c-shared-overlap-nccl-2.20.5-defaul-r000001-ccb4f09f1afdf6fd | a-000001 | `509c1d422670d50bde4a490fee78e63f98099d791dfb394bd34fd598419dde93` | `ebb3e653e698c7b43212008756575f20672955eb3c948acfd4230f2513d5be43` | `8dd4f55eac2784c4b2906129071e3f8f437737c81df02e8c64e2251156f56b04` |
| c-shared-overlap-nccl-2.20.5-defaul-r000002-9b5b645247c1e801 | a-000001 | `2d06872f52443f91796e4ba6d7dee6bb39ded54ca527e2d2e4c534fdf92333bc` | `b144110d98c19830284612192ecb30828f016302077c55aa2df75dce1b68a3a5` | `b225b31ebe22f64a8407cb1fea6f31f07326297d40de7747676a32149698182f` |
| c-shared-overlap-nccl-2.20.5-defaul-r000003-259c49e7dec26faa | a-000001 | `80624743b06a7159951213ef32b58fe3a8c845a97a06964a600b188e37b68414` | `ad1d8d505fcb412dbdf584a498b012f1f9365908c94e1a5af6653d6bf8061c49` | `f0a13e49ed78a595001e7b61bf5ca18a4dc1440de47df47c04cc303491cb695d` |
| c-shared-overlap-nccl-2.20.5-defaul-r000004-6eb5c721f2b4f791 | a-000001 | `f37b304b7db18734454133e7db0529c7affe1dde7dc5adbcfc059106be058c09` | `f4770435adf856b00c2331830eb4efe74efd320884e30c7425d9dde37a8ce9ec` | `fb7299aa152dbbf1935115e9a1ba2ee25cf641b31f28b4a61d75d915a0ad4247` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000000-8b533675e3bed4dd | a-000001 | `0ef5fa36af9d66c743319e37b27af0cbddfee55981003f0d345939f76ab690a3` | `dd2a0a74166d7acbedece71fea237ec37e38f5199a0db3a021f2ad106a7aab88` | `94bfbf02ae78610ecb06d541e9130ebf5c7420cc0b63be37003fa16298298be2` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000000-d9bdab399d443def | a-000001 | `b7d1991bd7df6efe199fafc4e4bddd3ebdf026d886a38ee29767cb50b64d821f` | `74f0b285bcd3ba000d62b65fc124a1944b5fd4a59b7170021c510e08bd7caf9c` | `8390a6d2aa0a29a8c481a8d8b744e6c2821d9cdbb06e384ac064ee7ad68978be` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000001-aa1bab7f5c353738 | a-000001 | `7aedddfad5b2701c51b8452de00cbf3eae4f66acdac842eb92e097f3b45bab7f` | `39b4cf65c1e94a7c9fe7599d4025e718555a5b35270b46a38f43c519c319daa4` | `be28a46f1bd10d08271d41e3ba4cbb303db7ad2f448213a61035c67e60df86d7` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000001-dc31f4bf29b06b9f | a-000001 | `cef673ef64e11d248bbe5c9d975e44f6e5694af427dc02dc20a8413ac13270f9` | `18c8226ec35ed4860c0234f380da387d59a962f86658db0e05e742139f52d645` | `16f5f129032c757e7a22c7c8c2d271b99e38092ea87eaceec7e6edf1deacd391` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000002-5c684e97077f6cd4 | a-000001 | `f809a0bdf956e656cb8713a8e2b87105cb26223ba7cdbb7bf6cd49684144098c` | `49116fd5021b50125937597470acde3bd649ae16535274280de97b1ce312af1f` | `a98138c64e0257742e770f44d2c076ab9d1fa9409e719123ec463872ed437c35` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000002-ec14b2d92ba89caa | a-000001 | `fd53b058bb2f6d73360cf0059354339ac1456dd9c0fbe879bc890a0ca6c376f9` | `f029de337875928836ba7a1cc34ca24334a55e0f733adf617791a0246cd3a0ec` | `f070d6a166a93e7a2cf04900cb3b714878c7d589094ed4a362161bee5b95e393` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000003-64978dd8aeb511b4 | a-000001 | `eb362702141178f6c0aa5ac4bdcf241e2fce75aa3ed2891f7f5cf90642a06c6b` | `a4bf4dba6c95ed6da34c3cf6c86d1fe715c2ae21801a6c7e946ebb8c79dd57e5` | `1f095223858868499833dac91caeee1a8d2fd3a677d34eac3f7609fcda04d173` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000003-e1c764ca7331da9c | a-000001 | `5827b5a042d01119e855f1592f14ab9684db6e9c260bcd622502c1c8a4701b49` | `2491efebc3a586ccbd57c414f27a219be0b944c3cb8f3806423a2ae5f9dd4de0` | `03946d9e75c94020af8649e9f58dc2b7ef3dc717871e73a0c1aa83b5745a5187` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000004-7f827b3a7fb84072 | a-000001 | `3fe07400f8807bd68f2ff5d554cda631e96cd5365e44ca713f11ade0ae38d589` | `80781b287734dcdd46be39a5a415ecb38bacc3c1dc4f10a2ab1284eafb7269db` | `72fd7dc00f13cd915f5b9c429c712f78911ff0d428ed77b326ff571bdbab6f15` |
| c-shared-overlap-nccl-2.20.5-ring-l-r000004-8ecdd84d4ec98d77 | a-000001 | `77ffaa26f71a5978b3970468f2ef0d22d69b040528d29de28e1f5cfcb02b3bc2` | `10361457846d4cfebd270cd501ac2681224521a6de23a959883eeb3680b55e01` | `bda54ad8d5a4dac8ba252171ad765474f55093c99587f1063d01f5d426d0e055` |
| c-shared-overlap-nccl-2.20.5-ring-s-r000000-734baa18e69dd1e5 | a-000001 | `74d5beed20a958399171b559025cebb1870d9d1134b10a46ac16376d96e8805a` | `ca7704eab04ef798742b98be8ab9faec42e651f546585e329c74258afc581c7f` | `28baf9e9c199e91c535067bf04f0a6bcda225be7aa8cbd3893b7cb5e64c4c140` |
| c-shared-overlap-nccl-2.20.5-ring-s-r000001-38fe374f0c08b167 | a-000001 | `9e041787628329ec4e4a917a4a902c5316d2546ece59fc7749529ed9d0fab330` | `fbc6e9f633e11cbf6769eeb35a3ac123d9a2332a1a2b01ba943db4f33fba1abf` | `350ad5abb7fc4b16eee1e626ea614d2361618e28337f0b774bfefa4346342c6f` |
| c-shared-overlap-nccl-2.20.5-ring-s-r000002-8da3d12aa2dfed3d | a-000001 | `18aa07dfa33b1a6527b0edb3067baedeafc23f3a44bc263310cbe0011db0604a` | `dda3fe10b27f34305a02a80c48991b28aff75a577895ad99b98baf885550be0e` | `788a91821fb48569431645a156c55e269958d611fca89e79535aaf71bad2ce9b` |
| c-shared-overlap-nccl-2.20.5-ring-s-r000003-cfaa86517b56ab2c | a-000001 | `ef04aeca1863f4dbd4f1212679c1cdd78498a136096fe2a61f0814486ccaebdb` | `f61cfe30750d850fcbb704daa7096ba1708af10300d69010516d1f479bce51b9` | `c0a28b119ff9ccf38747e5cbc4648d7d42b42a78771bb93a24c06fba8f69c3db` |
| c-shared-overlap-nccl-2.20.5-ring-s-r000004-cba63dba98c72727 | a-000001 | `fd9ec759ecd95b42a65735e7dcf492d5a456a42999b7361b3f929d13ce0054e2` | `4e4d40d909eb075a739bdcf0950e1409472514f7d00406345abf8cf734146252` | `ae6171519fd187ed794aa3002a7b1feb8f7e7414fac5564011374af21a256dfb` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000000-3b163d78cc344d60 | a-000001 | `ce0c3cc654a294715d74dba5063b8dfb383daf0c3360a2c32a044e3cb1603c48` | `209f516ea106d782c16ed41c2e3deefc559a268c68e96736700462155b0a2347` | `9caa76cd4eafe914e65b56b09545d76067619136e0e52c13b9cbced7feb88b96` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000000-819055e6d6f02186 | a-000001 | `f4fab49bd0489f31962116bbe4a5581a3ffc9e1b56b101c88529d9566f1826dd` | `b4f447572ca87d7b50267df9db22371b4a407e61452e46396fbbdbbddcdcbc77` | `f589ffc52bb71f6f76139457a58a198554f2c540038a221585e1154932de849c` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000001-447ecbfdcea47636 | a-000001 | `73f934225987fd91d7292e36a295bebb1da81d6458c7313f0b1509b7e8c57005` | `5153215888739dec9d25468691d4fc897bdcd188af68568c04f7340af9d453f0` | `10ff7cf19cfd6e7dcbf46b146c51156e7c203fbed41a70dc50e1cf3eb8e551ca` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000001-b459b21bd3d56f23 | a-000001 | `0dcb5f6af2aeea28aa8bf6099e627fa6482d21dbe3710670a9b1a2c908d52b75` | `f99c4bccd685155c01fae55f4564fc59c609a8ede0a78e35c509c7b1ba10ddc4` | `5d223be7e13f3bcef2dff6d02eabcc50d1f91f02591ed478eb78fa079f7831c8` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000002-66d998ba7dc14478 | a-000001 | `4f8a2cd86688be95746f09922e85844ed94159b16961cf6ee567ed84542e6ca9` | `b780c534df3e9ed83d1eca97321f46d205e202835df7b1427eb11851b1adfabe` | `879614867340532529543a2cf5935a5063b8b5bd5a3a2bf2685a1e2530d27e9e` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000002-bc27bf255f3750df | a-000001 | `55f3068d629fbee932a1d78410837e94717fb87823dfd7d2e641e6bdd38d39d7` | `42b5d719a5c611d2ba9b977d384abf817dbe04022d1e3d4f5eb46c29fe2fe843` | `5a7f627cf16c239b55c58a8d1bbca00cf2809a965e67c7351b5e4f9d94815998` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000003-2705619f9e19a116 | a-000001 | `7265e57feca80d723d6dded308508a5e79bd8c6b347060d6a64f0e1fa02ef476` | `524d9840e9be011807ffb836c908e90fb56120f656030c34b6b44726dd4c46b2` | `0f83dec10ef450909e419058ebdc9bde7fc64b86b5e483ec86250bf534485d54` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000003-3116d2f8f7e4c16e | a-000001 | `a903576a0cc7af09433fb00dc3236be53dccb3caeb747f3944424ae1a31c0c5c` | `9d8e94cd51e51b762841315b7efd9e0caea9fa79f237d1189e44e8711f54ac52` | `772b0dd37880acf496975c959cf20e5e5d9340cf208cde6e2be2b57f6c306b84` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000004-9c416bf50a13f1fb | a-000001 | `739bc232e72ccc461a2fecb21ea437f15f3b1c184c4b8b13f0433a451a809a4f` | `632bf015b828fb639b824c5a6426e3fa2b1986b7d0d31d3a25fd273c9dc95f64` | `90b4a7a072e47a5d96b62c720f6e9964be009ba6fcb68399e3628924fbeea41b` |
| c-shared-overlap-nccl-2.20.5-tree-l-r000004-d56ff9108fc1ef6a | a-000001 | `4bf1281c6edfb7633856b4957541d0855c98416a7b07e624359e39758f8c56b3` | `1341743bbebdd2a8b2019af0315b1846c4c2ca257c08d2b2f6aa543801881973` | `05ea240a5453da097b438184f768f00d5c52ec995946747797ed05bf9323fdb2` |
| c-shared-overlap-nccl-2.20.5-tree-s-r000000-c7551302dbed84dd | a-000001 | `91e3575abd0f289026fb920e3170de02e02f06b16c2c7ea06b18a3c071a6237d` | `52e954edf46e7fe4e89bcd296f665cf05c0b980b4092be97c1693d50e0ea7eee` | `1fcf62e074b2ce35044616b67a1b1aed6e50c5b25e798ab210db44676f8e1042` |
| c-shared-overlap-nccl-2.20.5-tree-s-r000001-6af1f4bf7e007b20 | a-000001 | `fb485d54e1c04ebccbcbbd6fff7ba3fb6b5250bf51ceea34ed93b84d60120176` | `85fcc79ff74686ac4b94e28255e09db786406b43e1d3292ef504d9869fd84503` | `76e7d57087429177b1ccba96cd086b5f196c1bad407d5389d0386c55995a7c7d` |
| c-shared-overlap-nccl-2.20.5-tree-s-r000002-bbcbe2a1c71451e1 | a-000001 | `607d003a54477ab8147caeb9573d70f909a12753c9a9a43cd35aacfd52a68598` | `d716b703db91c8331bfe2d667ed0b71f30099ff05ba8ddad7e3a178d1452ce87` | `b97122e44238738e00879bf3f0e6744699c684d13ce994c86f13456f8d3221c7` |
| c-shared-overlap-nccl-2.20.5-tree-s-r000003-38fead51278b729f | a-000001 | `6099baadabdfc93189d89f819d42bcb6cb02e6bd5e1755c76e00d6fdc03f7567` | `3d6a399f10c2a507285ba6d933887bbefdd64be44a77027094e2f50fdb10bc8c` | `d1970a8f5dd45d9bd14180d303a831fd1b4635b77c0a69601bc3cdf86525c71c` |
| c-shared-overlap-nccl-2.20.5-tree-s-r000004-c0e6fd51ac42fca0 | a-000001 | `2b4eb8905994176740a5f457ee00cd3095fa8150d1a8fef53f367b227579cf69` | `6db17b8a19d1ff2dc28b5520ce6f8d311137220ddb73c266877f4df4192e8827` | `de725ac6112632361bf324f87a13a7e652d5434f6e44b1a42b7309daf4b3b1d9` |

## Failure and retry accounting

- Terminal attempts: 40
- Retries preserved: 0
- Unselected terminal attempts: 0
- Status counts: `{"cancelled":0,"excluded":0,"failed":0,"parse-failed":0,"success":40}`

## Claims

No Rostam ranking claim is applicable because this complete evidence set does not contain W-full.

## Exact regeneration command

```sh
python -m experiments.rostam.analyze verify --run-directory experiments/rostam/results/shared-replay-20260720-r2 --selection-id primary --verdict-sha256 b6cd1aae4cfb2de020a840f941031d4a910d0d47d4c83aee1c09e0f5f6bc98db --output-directory experiments/rostam/results/publications/shared-replay-20260720-r2-primary --archive-descriptor experiments/rostam/results/archives/shared-replay-20260720-r2-primary-b6cd1aae4cfb2de020a840f941031d4a910d0d47d4c83aee1c09e0f5f6bc98db.raw-archive-descriptor.json --raw-archive experiments/rostam/results/archives/shared-replay-20260720-r2-primary-b6cd1aae4cfb2de020a840f941031d4a910d0d47d4c83aee1c09e0f5f6bc98db.raw.tar.gz --baseline-config nccl-2.19.3-default --candidate-config nccl-2.20.5-default --median-threshold-pct 8.0 --median-absolute-threshold-us 1.0
```
