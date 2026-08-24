# Federal Reserve economic-publication fixture provenance

These are minimal text excerpts of first-party Board of Governors occurrence
artifacts. They retain only the publication heading, contemporaneous
publication-time evidence, meeting association, and a source sentence needed
by the raw-publication adapter. No policy-rate values, votes, District
sub-events, or model features are imported. Retrieved and verified 2026-08-24.

| Fixture | Artifact type | Canonical first-party URL | Complete source SHA-256 | Excerpt SHA-256 | Historical-time evidence |
| --- | --- | --- | --- | --- | --- |
| `2010-fomc-statement.txt` | reduced HTML | https://www.federalreserve.gov/newsevents/pressreleases/monetary20100810a.htm | `67e12147768c94c66c63b63d842408c8c71ad6f2ef3ae0055f2dd7244af7c07e` | `3b2015756f6a149b83556bf746f1d92a5988d9d6e8d90cf246623b27ccbec09b` | The occurrence page establishes August 10, 2010 but says only “For immediate release.” It does not establish a clock time, so this fixture is deliberately `date_only`. |
| `2010-fomc-minutes.txt` | reduced HTML | https://www.federalreserve.gov/newsevents/pressreleases/monetary20100831a.htm | `363bc9a35e2089a9e3e81fb68a56ca58880d254481bed7e37260e034e4add842` | `341f70cf2d2a2deeb6f21920bf09c05f8b60ed08dd945d38bf39591046771101` | The occurrence page explicitly establishes August 31, 2010 at 2:00 p.m. EDT and associates the publication with the August 10 meeting. |
| `2010-beige-book.txt` | reduced PDF text | https://www.federalreserve.gov/fomc/beigebook/2010/20100113/fullreport20100113.pdf | `44eeacb3a92dedf7cfaf3a1b5275017bf2df6a407c8fcee702e54004121acca2` | `71b740abe631ad7e76c8fc6f5fdcc73cbb62abd592e9df2fd4945fdc7e6de1b3` | The PDF cover explicitly establishes January 13, 2010 at 2:00 p.m. EST. |
| `2024-fomc-statement.txt` | reduced HTML | https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm | `77319a344e9a69c92c09029f8e57dcef447cefa91604dc22d09d16a7482e8cf8` | `84cc2cdc68c79cb12a577f76a4fb8e7bde01f76ed15150cee2bb39f7a401ed61` | The occurrence page explicitly establishes June 12, 2024 at 2:00 p.m. EDT. |
| `2024-fomc-minutes.txt` | reduced HTML | https://www.federalreserve.gov/newsevents/pressreleases/monetary20240703a.htm | `0fdab59f53b2b9fe89fa16537458188d0e12ed599b8691ad3fcc38d1bbc99472` | `819ef04e688dc43132bfdde665f89e3b8b48d84619f1064aa31bf45a7e86f3c8` | The occurrence page explicitly establishes July 3, 2024 at 2:00 p.m. EDT and associates the publication with the June 11–12 meeting. |
| `2024-beige-book.txt` | reduced PDF text | https://www.federalreserve.gov/monetarypolicy/files/BeigeBook_20240417.pdf | `753ef721d294293cf4b1525f08e58598447a68e4d5fc9b5f0a3b1d183b4189c3` | `676a70a231f6dca06d42e85f97abb99f3eff1f9b6b48518c7a1c32cb48b37b58` | The PDF cover explicitly establishes April 17, 2024 at 2:00 p.m. EDT. |

The excerpt hashes are pinned in `manifest.tsv`. Normal acquisition retains
complete HTML, or both the complete Beige Book PDF and deterministic
`pdftotext` output, and records every SHA-256 in the same manifest contract.
