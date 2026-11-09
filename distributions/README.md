# Complete high-throughput distribution

[Download the actual ZIP](https://media.githubusercontent.com/media/xqh19970407/PhononBench/main/distributions/PhononBench_high_throughput_20261109.zip) (85,605,096 bytes, 81.64 MiB).

SHA-256: `d312fe064d320788f8ccd19815be3c81cfe0dd0eaf6a7a54e4ae7520b7d8ce46`.

The archive contains code, the actual MatterSim 5M checkpoint, inference source, environment TXT exports, supporting wheels and documentation. It does not contain the entire installed Conda environment or private user submissions. Git tracks the ZIP with LFS; the media link resolves to its actual contents.

All 59 packaged file hashes and ZIP CRC were verified. A 2D job was run from the extracted package with working directory `/tmp`, loading the checkpoint from the package itself. See [verification.json](verification.json).

The packaged source-code commit is `2b4a02faf69c178e38db0e613566ad283d2d737c`; the subsequent artifact commit only adds this distribution and its verification records.
