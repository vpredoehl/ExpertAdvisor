# Census economic-release fixture provenance

These deterministic fixtures are minimal excerpts of first-party Census
occurrence PDFs. They retain only the contemporaneous release header, Census
release number, publication heading, reference period, and a source sentence.
No numerical values are parsed or imported. The complete PDFs were retrieved
from the official historical-release indexes and verified on 2026-08-24.

| Fixture | Authoritative URL | Complete PDF SHA-256 | Excerpt SHA-256 |
| --- | --- | --- | --- |
| `2010-retail-sales-advance.txt` | https://www2.census.gov/retail/releases/historical/marts/adv1005.pdf | `560569853c598f8a4659d458795e0509e643dd0cb2824e42ad38223df68776a7` | `8ca65243cb6e62640733342265c9fb6edce3a86e7046540093b293e3d61c386d` |
| `2024-retail-sales-advance.txt` | https://www2.census.gov/retail/releases/historical/marts/adv2402.pdf | `78009b610a23ca760b91a98c9c9856584a9a076ad0f7a1f7986bb5ca1babb3fb` | `fd997aff4112b62f82fff8d3540a1e8387fa0d87f7616f39e4060e2bb4f02e91` |
| `2010-new-residential-construction.txt` | https://www.census.gov/construction/nrc/pdf/newresconst_201005.pdf | `b8d28b8f0ec174b88fb8929322ce90fd6e27ce9bbd92fed5b42e60a802cf7ab1` | `5f4b0f7ab1becd7b4785de8c6872dac46603b7eeb06c4f6312aa0abb83afaf26` |
| `2024-new-residential-construction.txt` | https://www.census.gov/construction/nrc/pdf/newresconst_202404.pdf | `7a3ce2d5218f56ffcffc3c294b9b551f257b0d8c78296d3f333074d2e2ea0373` | `f06be8dde931fc41bb91ce5647d9a7cb67ef45fc11b13c37fea481668736d8f9` |
| `2010-new-residential-sales.txt` | https://www.census.gov/construction/nrs/pdf/newressales_201005.pdf | `835b625ee13501665b9679ab66dee9b5d8d4a53f73b846549d542ac5352955f5` | `6d32a676b32716f6527702dd607aeaf0110b53c9ce21b35c42dbd53bfbae2c3c` |
| `2024-new-residential-sales.txt` | https://www.census.gov/construction/nrs/pdf/newressales_202412.pdf | `40559d1886a26c2356a96484332e4422a7d317b0b9da47ef3fb52fce9ff7a78e` | `f5d56f8381ac752bb1025737d97f8dd37aa30fc68ce6b288dec5cda09fea7441` |
| `2010-manufacturers-orders.txt` | https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2010/may10prel.pdf | `b71634b22be824db064a2c6b78331130a7bf857225e19f91651dc6759cf10c0e` | `f11057399380bdfe6fbbf286f923d8fcaeb92a528befe4dba9a48d69ae0d4420` |
| `2024-manufacturers-orders.txt` | https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2024/feb24prel.pdf | `c2ed37debf7fcfb6f2fe243bd228c6bb9403d6d5079dbafb927a6cf9b191f19b` | `92d1dd9465ac60b0afd9100a265c58e92665c8c08ab2504f033a56340255abda` |
| `2010-durable-goods-advance.txt` | https://www.census.gov/manufacturing/m3/historical_data/pressreleases/adv/2010/may10adv.pdf | `270c5b02c414f3332cfbab05f3a02d7dc748aaf3a8f241b109930deedab676da` | `93f84b87e0838f0c02cc5e1c1b8fff7c004fbd207c1c2cb4e9751c3caab91f32` |
| `2024-durable-goods-advance.txt` | https://www.census.gov/manufacturing/m3/historical_data/pressreleases/adv/2024/feb24adv.pdf | `43e2b4481530c6e6409595027b9dd65a6dda368ee1e4f9922870e5901cf9244f` | `f648eb4b8cfdb0053847bea3bd84ad219f6800b59df43c912b1c73a154abd478` |
| `2010-construction-spending.txt` | https://www.census.gov/construction/c30/pdf/pr201005.pdf | `0fd3bf223c59c897f77ea6a41bf8f34e9c65667b2199a58cb371688114efea9c` | `17cd24d4e4ddd97fb4cbe6d42d6c6d24540e59087a4c984410bb069467896229` |
| `2024-construction-spending.txt` | https://www.census.gov/construction/c30/pdf/pr202402.pdf | `f2e994c22a3f6925e39e20645a36e93c7609cbddff21b1c42b641a175e7baec4` | `70cf68c32152d740762d85a6f01456caddd1f6b7282cd3db35c62a409ff2254b` |

The excerpt hashes are repeated in `manifest.tsv`, the offline parser handoff.
The acquisition script retains complete PDFs plus deterministic `pdftotext`
outputs and records both hashes; reduced fixtures are explicitly labeled
`first_party_excerpt_v1`.

New Residential Construction and New Residential Sales are joint Census/HUD
publications hosted in the Census occurrence archive. They remain one raw
publication event each; permits, starts, completions, prices, and tables are
not exploded into synthetic events.
