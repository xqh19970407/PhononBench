# Third-party notices

The packaged MatterSim checkpoint and inference Python/Cython sources are from
Microsoft MatterSim and retain its MIT license. The source snapshot corresponds
to the running website; `vendor/MatterSim/SOURCE_MANIFEST.json` records exact
source hashes. Only inference dependencies were collected. The upstream build
configuration is retained as `pyproject.upstream.toml`; the new packaging metadata
describes this inference snapshot and preserves its local version `0.0`.

Phonopy is distributed under the BSD 3-Clause license; its unmodified wheel
includes the upstream license. ASE, PyTorch and the remaining dependencies
retain their respective licenses. Environment inventories do not redistribute
all their binaries. This update does not change the licensing status of the
repository's pre-existing legacy scripts.
