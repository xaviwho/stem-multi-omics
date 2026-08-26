"""Compile + deploy MultiOmicsProvenance.sol to PureChain.

Usage
-----
    python scripts/deploy_provenance.py [--dry-run]

Flow
----
1. Loads RPC_URL / CHAIN_ID / PRIVATE_KEY from .env
2. Compiles contracts/MultiOmicsProvenance.sol with py-solc-x (auto-installs
   solc 0.8.20 the first time)
3. With --dry-run: prints bytecode size + estimated gas; no transaction sent.
4. Without --dry-run: signs + broadcasts the deployment tx, waits for receipt,
   appends PROVENANCE_CONTRACT_ADDRESS=<addr> to .env, and writes the ABI to
   artifacts/contracts/MultiOmicsProvenance.abi.json for runtime use.

Safety
------
- Will refuse to send the tx if PRIVATE_KEY is empty.
- Will print the contract address it's about to deploy from and require
  enter-key confirmation before broadcasting (unless --yes is passed).
- PureChain is gas-free, but we still set a sane gas limit for safety.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Compile only; do not broadcast a deployment tx.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the interactive 'are you sure' prompt.")
    return p.parse_args()


def load_env() -> dict[str, str]:
    """Minimal .env loader (no python-dotenv dep)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise SystemExit(f"Missing .env at {env_path} — copy .env.example first.")
    out: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def compile_contract() -> tuple[bytes, list]:
    """Returns (bytecode, abi). Auto-installs solc 0.8.20 the first time."""
    try:
        from solcx import compile_standard, install_solc, set_solc_version
    except ImportError as e:
        raise SystemExit(
            "py-solc-x not installed. Run: pip install py-solc-x"
        ) from e

    install_solc("0.8.20")
    set_solc_version("0.8.20")

    src_path = ROOT / "contracts" / "MultiOmicsProvenance.sol"
    src = src_path.read_text(encoding="utf-8")

    out = compile_standard(
        {
            "language": "Solidity",
            "sources": {"MultiOmicsProvenance.sol": {"content": src}},
            "settings": {
                # viaIR enables Yul-based codegen, required for the deep
                # parameter list of registerIngestion (8 args).
                "viaIR": True,
                # PureChain is on a pre-Shanghai EVM fork that does not
                # support the PUSH0 opcode (0x5F, EIP-3855). Targeting
                # `paris` keeps the codegen on opcodes available since
                # London (2021). Without this, calls to deployed contracts
                # revert with "invalid opcode: PUSH0".
                "evmVersion": "paris",
                "optimizer": {"enabled": True, "runs": 200},
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode.object"],
                    }
                },
            },
        },
        solc_version="0.8.20",
    )
    contract = out["contracts"]["MultiOmicsProvenance.sol"]["MultiOmicsProvenance"]
    bytecode = bytes.fromhex(contract["evm"]["bytecode"]["object"])
    abi = contract["abi"]
    return bytecode, abi


def main() -> int:
    args = parse_args()
    env = load_env()

    rpc_url = env.get("RPC_URL", "")
    chain_id = int(env.get("CHAIN_ID", "0") or "0")
    private_key = env.get("PRIVATE_KEY", "")
    wallet = env.get("WALLET_ADDRESS", "")
    gas_limit = int(env.get("GAS_LIMIT", "500000"))

    print(f"[deploy] RPC: {rpc_url}")
    print(f"[deploy] chain id: {chain_id}")
    print(f"[deploy] wallet: {wallet or '<empty>'}")

    print("[deploy] compiling MultiOmicsProvenance.sol ...")
    bytecode, abi = compile_contract()
    print(f"[deploy] bytecode size: {len(bytecode)} bytes")

    # Persist ABI immediately — useful for tests + runtime even if deploy fails.
    abi_dir = ROOT / "artifacts" / "contracts"
    abi_dir.mkdir(parents=True, exist_ok=True)
    abi_path = abi_dir / "MultiOmicsProvenance.abi.json"
    abi_path.write_text(json.dumps(abi, indent=2), encoding="utf-8")
    print(f"[deploy] ABI saved: {abi_path}")

    if args.dry_run:
        print("[deploy] --dry-run: skipping broadcast.")
        return 0

    if not private_key:
        raise SystemExit(
            "[deploy] PRIVATE_KEY is empty in .env. Set it before deploying. "
            "Reminder: the private key in .env was previously shared in chat — "
            "consider rotating to a fresh wallet first."
        )

    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError as e:
        raise SystemExit("web3 / eth-account not installed; pip install web3") from e

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise SystemExit(f"[deploy] cannot connect to RPC at {rpc_url}")

    acct = Account.from_key(private_key)
    if wallet and acct.address.lower() != wallet.lower():
        raise SystemExit(
            f"[deploy] WALLET_ADDRESS ({wallet}) does not match PRIVATE_KEY-derived "
            f"address ({acct.address}). Aborting to prevent signing under the wrong identity."
        )
    print(f"[deploy] signer address: {acct.address}")

    nonce = w3.eth.get_transaction_count(acct.address)
    contract = w3.eth.contract(abi=abi, bytecode=bytecode.hex())

    # Contract deployment needs far more gas units than the .env GAS_LIMIT
    # (which is sized for the small per-call register* transactions). Storing
    # ~4.6 KB of bytecode at 200 gas/byte alone costs ~930k, before the
    # constructor runs. Estimate dynamically + 1.5x buffer; fall back to 5M
    # if estimation fails. Cost on PureChain is zero either way (gas price = 0).
    deploy_gas = max(gas_limit, 5_000_000)
    try:
        estimated = w3.eth.estimate_gas(
            contract.constructor().build_transaction({
                "from": acct.address,
                "nonce": nonce,
                "gasPrice": 0,
                "chainId": chain_id,
            })
        )
        deploy_gas = max(int(estimated * 1.5), 5_000_000)
        print(f"[deploy] gas estimate: {estimated:,} -> using limit: {deploy_gas:,}")
    except Exception as e:
        print(f"[deploy] gas estimation failed ({e!r}); using fallback {deploy_gas:,}")

    tx = contract.constructor().build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "gas": deploy_gas,
        "gasPrice": 0,                       # PureChain is gas-free
        "chainId": chain_id,
    })

    if not args.yes:
        print(
            f"\nAbout to deploy MultiOmicsProvenance to chain {chain_id} "
            f"from {acct.address}.\n"
            "Press Enter to confirm, Ctrl-C to abort."
        )
        try:
            input()
        except KeyboardInterrupt:
            print("\n[deploy] aborted by user.")
            return 1

    signed = acct.sign_transaction(tx)
    # Web3.py 7+ renamed `rawTransaction` to `raw_transaction`; fall back for 6.x.
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    print(f"[deploy] tx hash: {tx_hash.hex()}")

    print("[deploy] waiting for receipt ...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt.status != 1:
        raise SystemExit(f"[deploy] tx failed: {receipt}")

    address = receipt.contractAddress
    print(f"[deploy] deployed at: {address}")

    # Append/update PROVENANCE_CONTRACT_ADDRESS in .env
    env_path = ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith("PROVENANCE_CONTRACT_ADDRESS"):
            lines[i] = f"PROVENANCE_CONTRACT_ADDRESS={address}"
            found = True
            break
    if not found:
        lines.append(f"PROVENANCE_CONTRACT_ADDRESS={address}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[deploy] updated PROVENANCE_CONTRACT_ADDRESS in {env_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
