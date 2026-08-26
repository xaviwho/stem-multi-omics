// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title  Multi-Omics Provenance Ledger
/// @notice Two-checkpoint audit trail for multi-omics dataset ingestion and
///         processed-output registration, per Paper 1 §3.
/// @dev    Designed for PureChain (gas-free) but works on any EVM chain.
///         Storage is intentionally minimal — large strings are kept off-chain
///         and committed-to via SHA-256 hashes encoded as bytes32.
contract MultiOmicsProvenance {

    struct Ingestion {
        string  sourceName;        // e.g. "Novak2022_PXD028283"
        string  sourceUrl;         // versioned download URL
        string  sourceVersion;     // e.g. "PXD028283 (v1, 2021-12-07)"
        string  accessDate;        // ISO-8601
        bytes32 rawDataHash;       // SHA-256 of the raw downloaded file
        string  markersUsed;       // JSON-encoded list of HGNC symbols
        string  normalization;     // free-form description
        bytes32 pipelineVersion;   // git commit hash (lo 32 bytes of full SHA-1 padded)
        uint256 timestamp;         // block.timestamp at registration
        address registrar;         // msg.sender
    }

    struct Processed {
        string  sourceName;
        bytes32 processedHash;     // SHA-256 of the processed score matrix
        bytes32 pipelineVersion;
        bytes32 linkedIngestionKey; // key into the `ingestions` mapping
        uint256 timestamp;
        address registrar;
    }

    mapping(bytes32 => Ingestion) public ingestions;
    mapping(bytes32 => Processed) public processedRecords;

    bytes32[] public ingestionKeys;
    bytes32[] public processedKeys;

    event IngestionRegistered(
        bytes32 indexed key,
        string  sourceName,
        bytes32 rawDataHash,
        address indexed registrar
    );

    event ProcessedRegistered(
        bytes32 indexed key,
        bytes32 indexed linkedIngestionKey,
        string  sourceName,
        bytes32 processedHash,
        address indexed registrar
    );

    /// @notice Register a raw-data ingestion event.
    /// @return key Deterministic key (keccak256(sourceName, rawDataHash, registrar))
    ///             so re-registering the same (source, hash, signer) triple
    ///             overwrites the earlier record — useful when a script re-runs.
    function registerIngestion(
        string calldata sourceName,
        string calldata sourceUrl,
        string calldata sourceVersion,
        string calldata accessDate,
        bytes32 rawDataHash,
        string calldata markersUsed,
        string calldata normalization,
        bytes32 pipelineVersion
    ) external returns (bytes32 key) {
        key = keccak256(abi.encode(sourceName, rawDataHash, msg.sender));
        bool isNew = ingestions[key].timestamp == 0;
        ingestions[key] = Ingestion({
            sourceName: sourceName,
            sourceUrl: sourceUrl,
            sourceVersion: sourceVersion,
            accessDate: accessDate,
            rawDataHash: rawDataHash,
            markersUsed: markersUsed,
            normalization: normalization,
            pipelineVersion: pipelineVersion,
            timestamp: block.timestamp,
            registrar: msg.sender
        });
        if (isNew) {
            ingestionKeys.push(key);
        }
        emit IngestionRegistered(key, sourceName, rawDataHash, msg.sender);
    }

    /// @notice Register a processed-output event linked to a prior ingestion.
    function registerProcessed(
        string calldata sourceName,
        bytes32 processedHash,
        bytes32 pipelineVersion,
        bytes32 linkedIngestionKey
    ) external returns (bytes32 key) {
        key = keccak256(abi.encode(sourceName, processedHash, msg.sender));
        bool isNew = processedRecords[key].timestamp == 0;
        processedRecords[key] = Processed({
            sourceName: sourceName,
            processedHash: processedHash,
            pipelineVersion: pipelineVersion,
            linkedIngestionKey: linkedIngestionKey,
            timestamp: block.timestamp,
            registrar: msg.sender
        });
        if (isNew) {
            processedKeys.push(key);
        }
        emit ProcessedRegistered(
            key, linkedIngestionKey, sourceName, processedHash, msg.sender
        );
    }

    function ingestionCount() external view returns (uint256) {
        return ingestionKeys.length;
    }

    function processedCount() external view returns (uint256) {
        return processedKeys.length;
    }

    /// @notice Verify a current file hash against a stored ingestion record.
    function verifyIngestion(bytes32 key, bytes32 currentHash)
        external
        view
        returns (bool matched, bytes32 recordedHash, string memory sourceVersion)
    {
        Ingestion storage rec = ingestions[key];
        require(rec.timestamp != 0, "no record for key");
        return (rec.rawDataHash == currentHash, rec.rawDataHash, rec.sourceVersion);
    }
}
