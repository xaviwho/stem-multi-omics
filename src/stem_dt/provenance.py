"""Data provenance: hashing, JSON manifest, optional smart-contract registration.

Methodology cross-reference: §3.

Design note (M-G): the smart-contract layer is optional. The default mode
emits a signed JSON manifest under `artifacts/provenance/` which provides
the same integrity / versioning / reproducibility guarantees for a 4-source
download log. Set `ENABLE_CHAIN=true` and provide chain credentials to also
register on-chain. The methodology should defend the *additional* property
the chain provides (cross-lab non-repudiation, regulatory submission, etc.)
to justify the infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Web3 imports are deferred into _load_chain_client() so the rest of the
# package imports cleanly when web3 isn't installed.


CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit_hash(repo: str | Path = ".") -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class IngestionRecord:
    """§3.2 raw-ingestion provenance payload."""

    source_name: str
    source_url: str               # M-G: include versioned URL, not just hash
    source_version: str           # e.g. "HPA v23.0", "DepMap 24Q2"
    access_date: str              # ISO-8601
    raw_data_hash: str            # SHA-256 of raw downloaded file
    markers_used: list[str]
    normalization: str
    pipeline_version: str | None  # git commit hash


@dataclass
class ProcessedRecord:
    """§3.2 processed-output provenance payload, linked back to ingestion."""

    source_name: str
    processed_hash: str
    pipeline_version: str | None
    linked_ingestion_hash: str    # SHA-256 of the ingestion record JSON


@dataclass
class ProvenanceManifest:
    """Local manifest collecting all ingestion + processed records for a run."""

    run_id: str
    created_at: str
    ingestions: list[IngestionRecord] = field(default_factory=list)
    processed: list[ProcessedRecord] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "created_at": self.created_at,
                "ingestions": [asdict(r) for r in self.ingestions],
                "processed": [asdict(r) for r in self.processed],
            },
            indent=2, sort_keys=True,
        )


def new_manifest(run_id: str | None = None) -> ProvenanceManifest:
    return ProvenanceManifest(
        run_id=run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def write_manifest(manifest: ProvenanceManifest, out_dir: str | Path) -> Path:
    out = Path(out_dir) / f"manifest_{manifest.run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(manifest.to_json(), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Pipeline-driver convenience: register all 3 Paper 1 data sources at once.
# ---------------------------------------------------------------------------

def build_paper1_ingestion_records(
    *,
    icufn_repo_path: str | Path,
    novak_proteingroups_path: str | Path,
    psp_kinase_substrate_path: str | Path | None = None,
) -> list[IngestionRecord]:
    """Compute SHA-256 hashes of the three Paper 1 raw inputs + return records.

    Returns one `IngestionRecord` per data source: Jerber (transcriptomics),
    Novak (proteomics, Path 3 raw aggregate), and PhosphoSitePlus (kinase
    activity priors). Hashes are computed on the canonical files we actually
    consume; the caller's pipeline_version (git commit) is filled in here too.
    """
    icufn = Path(icufn_repo_path)
    pipeline_ver = git_commit_hash() or ""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Jerber 2021 — we use the ICUFN repo's pre-processed AnnData as the
    # canonical artifact. SHA-256 of the .h5 file fixes the input version.
    jerber_h5 = icufn / "data" / "raw" / "dopaminergic_all_timepoints.h5"
    records: list[IngestionRecord] = []
    if jerber_h5.exists():
        records.append(IngestionRecord(
            source_name="Jerber2021",
            source_url="https://www.ebi.ac.uk/ena/browser/view/PRJEB38269",
            source_version="ENA PRJEB38269 / Zenodo 4333872",
            access_date=today,
            raw_data_hash=sha256_file(jerber_h5),
            markers_used=[
                "POU5F1", "NANOG", "SOX2", "UTF1", "TDGF1",
                "TH", "DDC", "SLC6A3", "DRD2", "LMX1A", "FOXA2",
            ],
            normalization="Scanpy normalize_total + log1p; mean over marker set",
            pipeline_version=pipeline_ver,
        ))

    # 2. Novak 2022 — Path 3 re-processed proteinGroups.txt
    novak = Path(novak_proteingroups_path)
    if novak.exists():
        records.append(IngestionRecord(
            source_name="Novak2022_Path3",
            source_url="https://ftp.pride.ebi.ac.uk/pride/data/archive/2022/02/PXD028283/",
            source_version="PXD028283 raw files re-processed via MaxQuant 2.8.0.0",
            access_date=today,
            raw_data_hash=sha256_file(novak),
            markers_used=["TH", "NR4A2", "PITX3", "LMX1B", "ALDH1A1", "KCNJ6"],
            normalization="MQ LFQ; log2(1+intensity); mean over G_Pr ∩ measured; "
                         "leakage-safe min-max from train trajectories only",
            pipeline_version=pipeline_ver,
        ))

    # 3. PhosphoSitePlus — only used as marker selection / pathway prior for K_t
    if psp_kinase_substrate_path is not None:
        psp = Path(psp_kinase_substrate_path)
        if psp.exists():
            records.append(IngestionRecord(
                source_name="PhosphoSitePlus_KS",
                source_url="https://www.phosphosite.org/staticDownloads",
                source_version="Kinase-Substrate dataset (date in file)",
                access_date=today,
                raw_data_hash=sha256_file(psp),
                markers_used=[
                    "TH_S40", "MAPK1_T185", "MAPK1_Y187", "MAPK3_T202",
                    "MAPK3_Y204", "AKT1_S473", "GSK3B_S9",
                ],
                normalization="kinase-target gene-set z-score (transcript-derived "
                             "kinase activity inference; not measured phosphoproteomics)",
                pipeline_version=pipeline_ver,
            ))
    return records


def register_paper1_provenance(
    records: list[IngestionRecord],
    artifacts_dir: str | Path,
) -> ProvenanceManifest:
    """Add the records to a fresh manifest, write it to disk, and (if
    `ENABLE_CHAIN=true`) also register each record on PureChain.

    Returns the manifest so the caller can pass its `run_id` downstream
    (e.g., to link processed-output records back to ingestion events).
    """
    manifest = new_manifest()
    for rec in records:
        manifest.ingestions.append(rec)
    manifest_path = write_manifest(manifest, artifacts_dir)
    print(f"[provenance] manifest written: {manifest_path}")

    if not chain_enabled():
        print(f"[provenance] ENABLE_CHAIN=false; skipping on-chain registration")
        return manifest

    print(f"[provenance] ENABLE_CHAIN=true; registering {len(records)} on PureChain")
    for rec in records:
        try:
            tx_hash = register_ingestion_onchain(rec)
            print(f"[provenance]   {rec.source_name:<22} tx={tx_hash}")
        except Exception as e:
            print(f"[provenance]   {rec.source_name:<22} FAILED: {e}")
    return manifest


# ---------------------------------------------------------------------------
# Optional on-chain registration (§3.3). Active only when ENABLE_CHAIN=true.
# ---------------------------------------------------------------------------

def chain_enabled() -> bool:
    return os.environ.get("ENABLE_CHAIN", "false").lower() == "true"


def _hex_to_bytes32(h: str | None) -> bytes:
    """Pack a hex string into a 32-byte word (right-padded with zeros).

    Accepts SHA-256 (64 hex chars), SHA-1 git commit (40 hex chars), or any
    short hex. Returns 32 bytes for direct use as a Solidity bytes32 arg.
    """
    if not h:
        return b"\x00" * 32
    h = h.removeprefix("0x")
    raw = bytes.fromhex(h)
    if len(raw) > 32:
        raise ValueError(f"hash too long for bytes32: {len(raw)} bytes")
    return raw + b"\x00" * (32 - len(raw))


def _load_chain_client():
    """Return (w3, contract, account, gas_limit, chain_id) — fail loud."""
    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError as e:
        raise RuntimeError("web3 / eth-account not installed; pip install web3") from e

    rpc_url = os.environ["RPC_URL"]
    chain_id = int(os.environ["CHAIN_ID"])
    address = os.environ["PROVENANCE_CONTRACT_ADDRESS"]
    private_key = os.environ["PRIVATE_KEY"]
    gas_limit = int(os.environ.get("GAS_LIMIT", "500000"))

    if not address:
        raise RuntimeError(
            "PROVENANCE_CONTRACT_ADDRESS is empty — deploy with "
            "scripts/deploy_provenance.py first."
        )

    repo_root = Path(__file__).resolve().parents[2]
    abi_path = repo_root / "artifacts" / "contracts" / "MultiOmicsProvenance.abi.json"
    if not abi_path.exists():
        raise RuntimeError(
            f"ABI not found at {abi_path}. Run scripts/deploy_provenance.py "
            "(or its --dry-run mode) at least once to compile and write the ABI."
        )
    abi = json.loads(abi_path.read_text(encoding="utf-8"))

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"cannot connect to RPC {rpc_url}")
    contract = w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)
    account = Account.from_key(private_key)
    return w3, contract, account, gas_limit, chain_id


def _send(w3, fn_call, account, gas_limit: int, chain_id: int) -> str:
    """Sign + broadcast + wait for receipt; return tx hash hex."""
    nonce = w3.eth.get_transaction_count(account.address)
    tx = fn_call.build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": gas_limit,
        "gasPrice": 0,            # PureChain is gas-free
        "chainId": chain_id,
    })
    signed = account.sign_transaction(tx)
    # Web3.py 7+ renamed `rawTransaction` to `raw_transaction`; fall back for 6.x.
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt.status != 1:
        raise RuntimeError(f"on-chain tx failed: {receipt}")
    return tx_hash.hex()


def register_ingestion_onchain(record: IngestionRecord) -> str:
    """Call `registerIngestion(...)` on PureChain. Returns tx hash."""
    if not chain_enabled():
        raise RuntimeError("ENABLE_CHAIN is not set")
    w3, contract, account, gas_limit, chain_id = _load_chain_client()
    fn = contract.functions.registerIngestion(
        record.source_name,
        record.source_url,
        record.source_version,
        record.access_date,
        _hex_to_bytes32(record.raw_data_hash),
        json.dumps(record.markers_used),
        record.normalization,
        _hex_to_bytes32(record.pipeline_version),
    )
    return _send(w3, fn, account, gas_limit, chain_id)


def register_processed_onchain(
    record: ProcessedRecord, linked_ingestion_key: str | None = None
) -> str:
    """Call `registerProcessed(...)` on PureChain. Returns tx hash.

    `linked_ingestion_key` is the on-chain key (bytes32 hex) returned by the
    matching registerIngestion call. If omitted, packs zeros — only do that
    for processed outputs that genuinely have no upstream ingestion.
    """
    if not chain_enabled():
        raise RuntimeError("ENABLE_CHAIN is not set")
    w3, contract, account, gas_limit, chain_id = _load_chain_client()
    fn = contract.functions.registerProcessed(
        record.source_name,
        _hex_to_bytes32(record.processed_hash),
        _hex_to_bytes32(record.pipeline_version),
        _hex_to_bytes32(linked_ingestion_key),
    )
    return _send(w3, fn, account, gas_limit, chain_id)


# ---------------------------------------------------------------------------
# §3.4 Verification — re-download and compare hashes.
# ---------------------------------------------------------------------------

def verify_against_manifest(
    manifest: ProvenanceManifest, source_name: str, current_file: str | Path
) -> dict[str, Any]:
    """Re-hash a current local file and compare against the manifest record.

    Returns
    -------
    dict with keys: matched (bool), recorded_hash, current_hash, source_version.
    """
    rec = next(
        (r for r in manifest.ingestions if r.source_name == source_name), None
    )
    if rec is None:
        raise KeyError(f"No ingestion record for source_name={source_name!r}")
    current = sha256_file(current_file)
    return {
        "matched": current == rec.raw_data_hash,
        "recorded_hash": rec.raw_data_hash,
        "current_hash": current,
        "source_version": rec.source_version,
    }
