# BEA economic-release fixture provenance

These deterministic fixtures are minimal excerpts of first-party BEA occurrence
pages. They retain only the embargo header, BEA release number, publication
heading, and opening sentence needed to establish product, GDP vintage,
reference period, and publication-level semantics. No numerical values are
parsed or imported. Retrieved and verified 2026-08-24.

- `2010-gdp-advance.txt`
  - Authoritative URL: https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-advance-estimate
  - Artifact type: reduced text from BEA HTML occurrence page.
  - SHA-256: `8bd92683795928f9cabb95e7424ad8816ee88bb43dd8e17dbcbc4d993f4bbd37`
  - Establishes BEA 10-37, advance vintage, Q2 2010, and July 30 at 8:30 a.m. EDT.
- `2010-gdp-second.txt`
  - Authoritative URL: https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-second-estimate-corporate-profits-2nd-quarter
  - Artifact type: reduced text from BEA HTML occurrence page.
  - SHA-256: `4550edb649a36cde1e69bf075f53e21160489516a58e54e09543ce050fd63c15`
  - Establishes BEA 10-41, second vintage, Q2 2010, and August 27 at 8:30 a.m. EDT.
- `2010-gdp-third.txt`
  - Authoritative URL: https://www.bea.gov/news/2010/gross-domestic-product-2nd-quarter-2010-third-estimate-corporate-profits-2nd-quarter-2010
  - Artifact type: reduced text from BEA HTML occurrence page.
  - SHA-256: `d4214f38479868bd76658ca31c1e3202f239a532a65d4a117d0b9f2834c29725`
  - Establishes BEA 10-47, third vintage, Q2 2010, and September 30 at 8:30 a.m. EDT.
- `2010-personal-income-outlays.txt`
  - Authoritative URL: https://www.bea.gov/news/2010/personal-income-and-outlays-june-2010
  - Artifact type: reduced text from BEA HTML occurrence page.
  - SHA-256: `05cc4856cb75db1e6505a1d095f760e1651eca10886ddfe1ccefbfbadac696c4`
  - Establishes BEA 10-38, the June 2010 publication, and August 3 at 8:30 a.m. EDT.
- `2024-gdp-third.txt`
  - Authoritative URL: https://www.bea.gov/news/2024/gross-domestic-product-third-estimate-corporate-profits-revised-estimate-and-gdp-1
  - Artifact type: reduced text from BEA HTML occurrence page.
  - SHA-256: `b86a992f27213bb43168026de14627eff2f73c549341581a9198774b0162ae84`
  - Establishes the modern word-ordinal title era, BEA 24-61, third vintage,
    Q3 2024, and December 19 at 8:30 a.m. EST.
- `2025-personal-income-outlays.txt`
  - Authoritative URL: https://www.bea.gov/news/2025/personal-income-and-outlays-july-2025
  - Artifact type: reduced text from BEA HTML occurrence page.
  - SHA-256: `66d82b173c0ebc833a902c6425e274d5f7ddf7afbf4491b72af187c6338cba2d`
  - Establishes the modern page format, BEA 25-36, the July 2025 publication,
    and August 29 at 8:30 a.m. EDT.

The hashes are repeated in `manifest.tsv`, which is the offline parser handoff.
The acquisition script retains and hashes complete occurrence HTML; reduced
fixtures are explicitly identified by `first_party_excerpt_v1`.
