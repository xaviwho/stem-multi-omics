"""One-shot on-chain provenance registration for Paper 1 inputs.

Submits an `Ingestion` record to the deployed MultiOmicsProvenance contract on
PureChain mainnet for each input dataset (Jerber, Novak/Dittmar, optional
PSP). Each record commits SHA-256(file) + git commit hash + access metadata.

Cost on PureChain is zero PCG (gas-free), but each transaction is permanent
and public. Idempotent: the contract keys records by
keccak256(sourceName, rawDataHash, msg.sender), so re-running with the same
file overwrites the existing record (timestamp refreshes; key stays stable).

Usage:
    python scripts/register_on_chain.py            # prompts for confirmation
    python scripts/register_on_chain.py --yes      # skip prompt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env then force ENABLE_CHAIN=true regardless of file setting.
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())
os.environ["ENABLE_CHAIN"] = "true"

from stem_dt import provenance  # noqa: E402

ICUFN = os.environ.get("ICUFN_REPO_PATH") or str(ROOT.parent / "stem-cell-digital-twin")
NOVAK = os.environ["NOVAK2022_PROTEINGROUPS_PATH"]
PSP   = os.environ.get("PSP_KINASE_SUBSTRATE_PATH") or None
PROV  = ROOT / "artifacts" / "provenance"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive confirmation (use for headless).")
    args = parser.parse_args()

    print("[chain] computing SHA-256 of pipeline inputs (this is the slow step)...")
    records = provenance.build_paper1_ingestion_records(
        icufn_repo_path=ICUFN,
        novak_proteingroups_path=NOVAK,
        psp_kinase_substrate_path=PSP,
    )

    print(f"\n[chain] {len(records)} records ready for on-chain registration:")
    for r in records:
        print(f"  - {r.source_name:<22} hash={r.raw_data_hash[:24]}...")
        print(f"      version: {r.source_version}")
        print(f"      url:     {r.source_url}")
        print(f"      markers: {r.markers_used}")

    contract = os.environ.get("PROVENANCE_CONTRACT_ADDRESS", "")
    chain_id = os.environ.get("CHAIN_ID", "")
    print(f"\n[chain] target: contract {contract} on chain {chain_id}")
    print(f"[chain] signer: {os.environ.get('WALLET_ADDRESS', '?')}")
    print(f"[chain] cost: 0 PCG per transaction (PureChain is gas-free)")
    print(f"[chain] permanence: records are immutable once mined")

    if not args.yes:
        print("\nPress Enter to broadcast, Ctrl-C to abort.")
        try:
            input()
        except KeyboardInterrupt:
            print("\n[chain] aborted by user; no transactions sent.")
            return 1

    print(f"\n[chain] registering {len(records)} records on PureChain ...")
    manifest = provenance.register_paper1_provenance(records, PROV)

    print(f"\n[chain] manifest written: artifacts/provenance/manifest_{manifest.run_id}.json")
    print(f"[chain] verify on-chain: contract.ingestionCount() should now reflect "
          f"the {len(records)} new (or refreshed) records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
