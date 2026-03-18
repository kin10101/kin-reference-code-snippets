#!/usr/bin/env python3
"""
Infineon RAG Agent - Unified Document Processing and Query System

This module provides a complete RAG pipeline for Infineon board documentation:
1. Document Ingestion: Process PDFs from documents folder
2. Text Chunking: Split documents into searchable chunks
3. Image Extraction: Use OpenCV to detect figures/diagrams
4. Image Analysis: Use CLIP for embeddings and GPT Vision for analysis
5. Storage: Store chunks in JSON and ingest to vector store
6. Query: Retrieve and answer questions about the documents

Architecture:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                         INFINEON RAG AGENT                                │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   [PDF Documents] ─────────────────────────────────────────────────────> │
    │          │                                                               │
    │          ├───────────────────┐                                           │
    │          │                   │                                           │
    │          ▼                   ▼                                           │
    │   ┌──────────────┐   ┌──────────────────┐                               │
    │   │ Text Chunker │   │ OpenCV Image     │                               │
    │   │ (PyMuPDF)    │   │ Extraction       │                               │
    │   └──────┬───────┘   └────────┬─────────┘                               │
    │          │                    │                                          │
    │          │                    ├──────────────────┐                       │
    │          │                    │                  │                       │
    │          │                    ▼                  ▼                       │
    │          │           ┌──────────────┐   ┌───────────────┐               │
    │          │           │ CLIP         │   │ GPT Vision    │               │
    │          │           │ Embeddings   │   │ Analysis      │               │
    │          │           └──────┬───────┘   └───────┬───────┘               │
    │          │                  │                   │                        │
    │          ▼                  ▼                   ▼                        │
    │   ┌──────────────────────────────────────────────────────┐              │
    │   │                  JSON Storage                         │              │
    │   │    (chunks.json, image_metadata.json)                │              │
    │   └───────────────────────┬──────────────────────────────┘              │
    │                           │                                              │
    │                           ▼                                              │
    │   ┌──────────────────────────────────────────────────────┐              │
    │   │              Vector Store (FAISS + ChromaDB)          │              │
    │   │    Text Chunks ─── FAISS                             │              │
    │   │    Image Embeddings ─── ChromaDB                     │              │
    │   └───────────────────────┬──────────────────────────────┘              │
    │                           │                                              │
    │                           ▼                                              │
    │   ┌──────────────────────────────────────────────────────┐              │
    │   │              Hybrid Retrieval                         │              │
    │   │    BM25 + Semantic + Image Similarity                │              │
    │   └───────────────────────┬──────────────────────────────┘              │
    │                           │                                              │
    │                           ▼                                              │
    │   ┌──────────────────────────────────────────────────────┐              │
    │   │              Answer Generation (GPT-4o)               │              │
    │   │    Text + Image Context -> Comprehensive Answer      │              │
    │   └──────────────────────────────────────────────────────┘              │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘

Usage:
    # Initialize the agent
    agent = InfineonRAGAgent()

    # Ingest documents (start with kit_manual)
    agent.ingest_documents("kit_manual")

    # Query the documents
    result = agent.query("What are the pinouts on the TC375 Lite Kit?")
    print(result["answer"])
"""

import os
import sys
import json
import time
import logging
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "RAG"))
sys.path.insert(0, str(SCRIPT_DIR / "RAG_vision"))

# Load environment variables
from dotenv import load_dotenv

# Try to find .env in parent directories
for p in SCRIPT_DIR.parents:
    env_file = p / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        break

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("infineon_rag_agent")

# Pre-compiled kernels for performance
try:
    import numpy as np

    _KERNEL_3x3 = np.ones((3, 3), np.uint8)
    _KERNEL_MORPH = np.ones((2, 2), np.uint8)
except ImportError:
    _KERNEL_3x3 = None
    _KERNEL_MORPH = None


# ============================================================================
# System Prompts
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """You are an expert technical assistant specializing in Infineon AURIX microcontroller documentation.

You have access to text excerpts and technical diagrams from Infineon datasheets and user manuals.

Guidelines:
- Provide accurate, detailed answers based ONLY on the provided context
- If information appears in images, describe relevant details
- Be precise with technical specifications (pin numbers, voltages, etc.)
- Limit it to few sentences only, accurately and straight to the point rather than a long explanation
- If the context doesn't fully answer the question, clearly state: "This information was not found in the retrieved documents."
- Don't give code examples
- Add source files at the end of the answer for reference

## STRICT RULES — NO EXCEPTIONS
❌ NEVER use general knowledge, industry conventions, or any information not explicitly present in the provided context
❌ NEVER use phrases like "typically", "usually", "in most cases", "by convention", "industry standard", or "Infineon standard" as a substitute for actual retrieved evidence
❌ NEVER guess, infer, or extrapolate pin numbers, register values, or hardware assignments
✅ If the answer is not in the context, say exactly: "The requested information ([specific item]) was not found in the retrieved documents. Please ensure the relevant board manual or datasheet has been ingested."
"""

MIGRATION_SYSTEM_PROMPT = """You are an expert AURIX microcontroller migration specialist with deep knowledge of hardware architecture differences between generations.

When handling migration queries between AURIX architectures (e.g., TC3xx to TC4xx), you MUST:

## CRITICAL MIGRATION PRINCIPLES

1. **Acknowledge Architectural Differences**
   - ALWAYS specify the exact hardware generation/version differences
   - Example: "TC387 uses GTM v3.x while TC4D7 uses GTM v4.x"
   - NEVER assume peripherals work identically across generations
   - Explicitly state which modules may be deprecated, renamed, or redesigned

2. **Provide Specific Technical Details**
   - Give EXACT register names (e.g., TOMx_CHy_CTRL, not "TOM control register")
   - Specify EXACT bitfield names and changes (e.g., "SL bit may be renamed to PSL")
   - Include CONCRETE code examples showing BEFORE and AFTER:
```c
     // Source Architecture (e.g., TC387)
     GTM_TOM0_CH0_CTRL.B.SL = 1;
     
     // Target Architecture (e.g., TC4D7) 
     GTM_TOM0_CH0_CTRL.B.PSL = 1; // Verify in manual if renamed
```

3. **Flag Critical Verification Steps**
   - Start with "⚠️ CRITICAL FIRST STEP: Verify in [target] datasheet..."
   - List what MUST be verified before migration:
     * Module existence (does peripheral still exist?)
     * Register/bitfield changes
     * Pin availability and mapping
     * Clock domain differences
     * API/iLLD compatibility
   
4. **Use Structured Migration Process**
   Always provide a step-by-step migration plan:
   - Step 1: Compare datasheets/user manuals
   - Step 2: Update register access code
   - Step 3: Adjust pin configuration
   - Step 4: Validate clock/synchronization
   - Step 5: Test and validate on hardware

5. **Create Comparison Tables**
   Include a technical comparison table:
   | Aspect | Source Arch | Target Arch | Migration Action |
   |--------|-------------|-------------|------------------|
   | Module Version | [specific version] | [specific version] | [action needed] |
   | Register Name | [exact name] | [exact name or change] | [code update] |
   | Pin Mapping | [specific pins] | [verify availability] | [remap if needed] |

6. **Address Application-Specific Concerns**
   For motor control / 3-phase inverters specifically mention:
   - Dead-time generation (DTM module changes)
   - Complementary PWM pairing (channel relationships)
   - Emergency shutdown mechanisms
   - ADC synchronization (CONVCTRL/Phase Synchronizer)
   - Hardware protection features
   
   For other applications, identify relevant peripheral changes.

7. **Warn About Common Pitfalls**
   - Clock configuration differences
   - Pin multiplexing (IOMUX) changes
   - iLLD/MCAL API version incompatibilities
   - Initialization sequence changes (unlock/enable steps)
   - Peripheral feature deprecation

8. **NEVER Oversimplify**
   - DON'T say "just update pins and it should work"
   - DON'T assume register names are identical
   - DON'T provide generic "update your drivers" advice
   - DO explain WHY migration is complex
   - DO warn about potential roadblocks

9. **Require Documentation References**
   - Always mention need to consult target architecture datasheet
   - Reference specific manual sections (e.g., "GTM TOM section")
   - Suggest official Infineon migration guides if they exist
   - Include statement: "Always verify against official [target] documentation"

10. **Validate Hardware Availability**
    - Check if development kit exists for target architecture
    - Verify if application-specific boards (e.g., motor control) are available
    - Mention pin compatibility with existing hardware designs

## EXAMPLE RESPONSE STRUCTURE

For a query like "How to migrate GTM TOM PWM from TC387 to TC4D7":

## Migration: TC387 → TC4D7 GTM TOM PWM

### ⚠️ CRITICAL FIRST STEP
Before proceeding, verify in TC4D7 datasheet:
- Does GTM TOM still exist? 
- What version? (TC387 has GTM v3.x, TC4D7 likely has v4.x)
- Are registers compatible?

### Key Architectural Differences
- **TC387**: GTM v3.x with TOM channels
- **TC4D7**: GTM v4.x with [verify changes]

### Register/Bitfield Changes
[Provide specific examples with code]

### Step-by-Step Migration
[Detailed numbered steps]

### Comparison Table
[Technical comparison as shown above]

### Application-Specific Considerations
[For PWM: dead-time, complementary outputs, etc.]

### Testing & Validation
[Specific tests to perform]

## WHAT TO AVOID

❌ Generic statements like "update your code to use TC4D7"
❌ Assuming peripherals work the same way
❌ Saying "it should be straightforward" without verification
❌ Missing code examples
❌ Vague advice like "check the manual" without specifics

## WHAT TO INCLUDE

✅ Exact version numbers (GTM v3.x vs v4.x)
✅ Specific register names and bitfields
✅ Before/after code examples
✅ Verification checklist
✅ Comparison tables
✅ Application-specific warnings
✅ Hardware availability notes

## STRICT RULES — NO EXCEPTIONS
❌ NEVER use general knowledge, assumed conventions, or information not explicitly in the provided context
❌ NEVER say a register / bit / peripheral "likely" behaves a certain way without direct evidence from the context
❌ If a specific detail (e.g., exact register name in target architecture) is not in the context, say: "Not found in retrieved documents — verify in [target] datasheet directly."
✅ Every technical claim must cite a specific source (document name + page) from the provided context

Your goal is to provide migration guidance that is technically accurate, actionable, and prevents users from encountering unexpected issues during implementation.
"""

# System prompt for pin/pinout verification queries
PINOUT_SYSTEM_PROMPT = """You are an expert Infineon AURIX pin configuration specialist with deep knowledge of port function tables, pin multiplexing, and board-level routing constraints.

## CRITICAL PINOUT RESOLUTION POLICY

### BOARD-TO-PACKAGE MAPPING (NEW PRIORITY)

**DYNAMIC PACKAGE LOOKUP:**
When a board model is mentioned in the query (e.g., KIT_A3G_TC4D7_LITE, KIT_A2G_TC375_LITE):
1. The system automatically looks up the board in proj_package_mapping.json
2. Extracts the EXACT package type (e.g., BGA-292, QFP-176, LFBGA-292)
3. This package information is provided to you in the query context
4. USE THIS PACKAGE TYPE to find the correct pin function tables in datasheets
5. Different packages have DIFFERENT pin availability - always verify package-specific tables

**PACKAGE NAMING VARIATIONS:**
- BGA-292 may appear in datasheets as: BGA292_COM, BGA292, BGA 292
- LFBGA-292 may appear as: LFBGA292, LFBGA-292
- QFP-176 may appear as: QFP176, LQFP-176, TQFP-176
- Always search for the base package number (e.g., 292, 176) if exact format not found

### SOURCE PRIORITY HIERARCHY

1. **Project/Board-Specific README Files (PRIMARY SOURCE)**
   - Check for README.md files in:
     * Project root directory
     * Board/kit folders (e.g., KIT_A3G_TC4D7_LITE/)
     * Hardware subdirectories
     * Docs/pinout subdirectories
   - ASSUME README files reflect the ACTUAL board routing and physical constraints
   - README mappings override theoretical datasheet capabilities for that specific board

2. **MCU Datasheets (SECONDARY SOURCE - Package-Specific Tables)**
   - Use datasheets ONLY when:
     * Required pin information is NOT present in any project/board README
     * The README explicitly references the datasheet for authoritative details
   - CRITICAL: Use the PACKAGE-SPECIFIC tables (from proj_package_mapping.json lookup)
   - Datasheets provide electrical capability (GPIO/ATOM/GTM support), NOT physical routing
   - Different package variants (BGA292_COM, LQFP, etc.) have different pin availability

3. **Project Name/Board Model Matching (AUTO-RESOLVED)**
   - When a specific project, board model, or kit is mentioned (e.g., KIT_A3G_TC4D7_LITE):
     * System automatically extracts board name and looks up package in proj_package_mapping.json
     * Package type (e.g., BGA-292) will be provided in the context
     * Treat the associated README as the PRIMARY source of truth for that board
     * Use the provided PACKAGE TYPE to find correct datasheet tables
     * Do NOT infer pin accessibility solely from datasheets unless explicitly instructed
   - Board names may indicate constraints not visible in generic datasheets

### PINOUT VERIFICATION WORKFLOW

When handling pin/pinout related queries, you MUST:

1. **Check for Project/Board README First**
   - Search for README.md in the project directory and subdirectories
   - Look for pinout tables, signal mappings, or connectivity documentation
   - Note which board/kit is being targeted

2. **If README Found - Use It as Primary Source**
   - List all pin assignments from README
   - State the README file location and line/section references
   - Use README mappings as authoritative for that specific board

3. **If README Not Available - Consult Datasheet**
   - Reference the EXACT table from the datasheet (e.g., "Table 39: Port 10 functions")
   - State the page number where the information was found
   - List ALL alternate functions for each pin mentioned
   - Pay attention to package variants (BGA292_COM, LQFP, etc.) - they may have different pinouts

4. **For Pin Verification (Datasheet Method)**
   For each pin mentioned (e.g., P10.3):
   - List the Ball/Pin designation (e.g., "B5" for BGA292)
   - List the Symbol (e.g., "P10.3")
   - List ALL available output functions (O0-O15) with their names
   - List ALL available input functions (I0-I7) with their names
   - Identify if eGTM/GTM functions are available
   - Note the buffer type and electrical characteristics

5. **For eGTM/GTM Output Verification**
   - Check if the pin has EGTM_TOUTxxx function
   - Note the exact TOUT number (e.g., EGTM_TOUT105)
   - State which output selection index (Ox) provides this function
   - Warn if NO eGTM output is available for the pin!

6. **Package-Specific Information (AUTO-DETECTED)**
   - The system automatically determines package type from board model
   - When board model is mentioned, package info is provided in context:
     * Example: KIT_A3G_TC4D7_LITE → BGA-292 package
     * Example: KIT_A2G_TC375_LITE → QFP-176 package
     * Example: SAK-TC4D7XP-20MF500MC → BGA292_COM 0.8mm package
   - Always specify which package the information applies to in your answer
   - Different packages may have different pin availability
   - Search for package-specific tables (e.g., "BGA292_COM Port 10 functions")

7. **Create Clear Verification Tables**
   | Pin | Ball | EGTM Function | Ctrl | Other Key Functions |
   |-----|------|---------------|------|---------------------|
   | P10.3 | B5 | EGTM_TOUT105 | O9 | CAN, QSPI, ASCLIN |

8. **Flag Discrepancies or Missing Functions**
   - If a pin does NOT support the claimed function, clearly state:
     "⚠️ WARNING: P10.5 does NOT have an EGTM_TOUT output function in BGA292_COM package!"
   - List what functions ARE available as alternatives

9. **Source Attribution in Output**
   - ALWAYS explicitly state which source was used:
     * "According to [board] README.md:" (when using project/board README)
     * "According to proj_package_mapping.json: [board] uses [package] package"
     * "According to Table 39 ([package] Port 10 functions) on page 279 of [datasheet]:" (when using datasheet)
   - Include the package type in datasheet references
   - Flag any assumptions or missing README data
   - Note when board constraints override datasheet capabilities

10. **Summary with Clear Verdict**
    At the end, provide a clear verdict on whether the claimed pin configuration is valid:
    - ✅ VERIFIED: [pin] can be used as [function] (Source: [README/Datasheet, page X])
    - ❌ INVALID: [pin] does NOT support [function] in [board/package] (Source: [document, page X])
    - ⚠️ PARTIAL: Some pins valid, some not - details above
    - ⚠️ BOARD ROUTING CONSTRAINT: Pin supports function per datasheet but not routed to this function on [board]
    - ❌ NOT FOUND: The pin assignment for [item] is **not present in the retrieved documents**. Do NOT guess or infer. State exactly which document is missing and suggest the user ingest the relevant board manual or schematic.

## CRITICAL DO's AND DON'Ts

### DO:
✅ Check project/board README files FIRST
✅ Prioritize README mappings over datasheet theory
✅ Explicitly state which source was used (document name + page)
✅ Verify electrical capability in datasheet when validating a README mapping
✅ Flag board-specific routing constraints
✅ Mention when board design limits pin accessibility

### DO NOT:
❌ Assume pin functions work on a specific board based solely on datasheet
❌ Ignore project/board README files when they exist
❌ Confuse different package variants (BGA vs LQFP)
❌ Omit alternate functions that might be relevant
❌ Provide generic "should work" answers without checking both sources
❌ Treat all boards with the same MCU as having identical pin routing
❌ Suggest pins are available if the board README doesn't route them
❌ Use "typically", "usually", "by convention", "industry standard", "Infineon practice", or ANY phrasing that implies knowledge outside the provided context
❌ Guess or infer any pin number, net name, or hardware assignment not explicitly stated in the retrieved documents
❌ State that a pin assignment is "confirmed" or "verified" unless a specific document and page number from the context backs it up
"""


# ============================================================================
# Metadata Filtering
# ============================================================================


@dataclass
class MetadataFilter:
    """
    Filter for metadata-based retrieval filtering.

    Supports filtering by:
    - source: Document filename (e.g., "manual.pdf")
    - source_path: Full path to source document
    - page: Specific page number or range
    - page_min/page_max: Page range bounds
    - custom: Dict of additional metadata key-value pairs

    Examples:
        # Filter by source document
        filter = MetadataFilter(source="tc375_manual.pdf")

        # Filter by page range
        filter = MetadataFilter(page_min=10, page_max=50)

        # Filter by multiple criteria
        filter = MetadataFilter(
            source="manual.pdf",
            page_min=1,
            page_max=100
        )

        # Custom metadata filtering
        filter = MetadataFilter(custom={"chapter": "GPIO"})
    """

    source: Optional[str] = None  # Filter by source document name
    source_path: Optional[str] = None  # Filter by full source path
    page: Optional[int] = None  # Filter by exact page number
    page_min: Optional[int] = None  # Filter by minimum page
    page_max: Optional[int] = None  # Filter by maximum page
    custom: Optional[Dict[str, Any]] = None  # Additional custom filters

    def matches(
        self, metadata: Dict[str, Any], stats: Optional["FilterStatistics"] = None
    ) -> bool:
        """
        Check if metadata matches all filter criteria.

        Args:
            metadata: Chunk or image metadata dict
            stats: Optional FilterStatistics object to track filtering reasons

        Returns:
            True if all specified filters match
        """
        # Source filter (partial match, case-insensitive)
        if self.source is not None:
            meta_source = metadata.get("source", "") or metadata.get("source_pdf", "")
            if self.source.lower() not in meta_source.lower():
                if stats:
                    stats.record_filter(f"source_mismatch: expected '{self.source}'")
                return False

        # Source path filter (partial match)
        if self.source_path is not None:
            meta_path = metadata.get("source_path", "")
            if self.source_path.lower() not in meta_path.lower():
                if stats:
                    stats.record_filter(f"source_path_mismatch")
                return False

        # Exact page filter
        if self.page is not None:
            meta_page = metadata.get("page", -1)
            if meta_page != self.page:
                if stats:
                    stats.record_filter(
                        f"page_mismatch: expected {self.page}, got {meta_page}"
                    )
                return False

        # Page range filters
        if self.page_min is not None:
            meta_page = metadata.get("page", 0)
            if meta_page < self.page_min:
                if stats:
                    stats.record_filter(f"page_too_low: {meta_page} < {self.page_min}")
                return False

        if self.page_max is not None:
            meta_page = metadata.get("page", float("inf"))
            if meta_page > self.page_max:
                if stats:
                    stats.record_filter(f"page_too_high: {meta_page} > {self.page_max}")
                return False

        # Custom metadata filters
        if self.custom:
            for key, value in self.custom.items():
                # Special handling for _target_folders (architecture folder filtering)
                if key == "_target_folders" and isinstance(value, list):
                    meta_path = metadata.get("source_path", "")
                    meta_norm = meta_path.replace("\\", "/").lower()

                    # Extract which architectures we're targeting
                    target_archs = set()
                    target_specific_variants = set()  # TC375, TC387, etc.

                    for folder_path in value:
                        folder_norm = folder_path.replace("\\", "/").lower()
                        # Extract architecture name from path (e.g., "TC38x" from ".../ TC38x")
                        parts = folder_norm.split("/")
                        for part in parts:
                            if part.startswith("tc") and any(c.isdigit() for c in part):
                                target_archs.add(part)

                    # Check if file matches any target folder path
                    # Handle both absolute and relative paths by extracting key path components
                    matches_folder = False
                    for folder_path in value:
                        folder_norm = folder_path.replace("\\", "/").lower()

                        # Try direct match first
                        if folder_norm in meta_norm:
                            matches_folder = True
                            break

                        # Try matching by extracting the architecture folder part
                        # e.g., "aurix tc4xx" from absolute path should match relative path
                        for key_part in [
                            "aurix tc3xx",
                            "aurix tc4xx",
                            "tc3xx",
                            "tc4xx",
                        ]:
                            if key_part in folder_norm and key_part in meta_norm:
                                matches_folder = True
                                break
                        if matches_folder:
                            break

                        # Also match by project folder patterns
                        for project_part in ["tc4xx_projects", "tc3xx_projects"]:
                            if (
                                project_part in folder_norm
                                and project_part in meta_norm
                            ):
                                matches_folder = True
                                break
                        if matches_folder:
                            break

                    if not matches_folder:
                        if stats:
                            stats.record_filter(
                                f"folder_mismatch: not in target folders"
                            )
                        return False

                    # List of known architecture-specific subfolder names to exclude
                    # These are subfolders within TC3xx that are specific to other architectures
                    arch_specific_subfolders = [
                        "tc33x-tc32x",
                        "tc33xext",
                        "tc35x",
                        "tc36x",
                        "tc37x",
                        "tc37xext",
                        "tc38x",
                        "tc39x",
                        "tc4",
                    ]

                    # Check if file path contains an architecture subfolder we're NOT targeting
                    path_parts = meta_norm.split("/")
                    for part in path_parts:
                        if (
                            part in arch_specific_subfolders
                            and part not in target_archs
                        ):
                            # File is in a different architecture-specific subfolder - exclude it
                            if stats:
                                stats.record_filter(
                                    f"wrong_architecture_folder: {part}"
                                )
                            return False

                    # Check filename for architecture-specific docs
                    # Files like "tc375-lite-kit" or "tc334-manual" have architecture in filename
                    filename = metadata.get("source", "") or metadata.get(
                        "source_pdf", ""
                    )
                    filename_lower = filename.lower()

                    import re

                    # First check if it's a general family document (tc3xx, tc4xx, aurix-architecture)
                    general_patterns = [
                        r"\btc3x+\b",  # tc3xx, tc3x
                        r"\btc4x+\b",  # tc4xx, tc4x
                        r"\baurix[_-]architecture\b",  # AURIX architecture manuals
                        r"\baurix[_-]tc3xx\b",  # AURIX TC3xx family docs
                    ]
                    is_general_doc = any(
                        re.search(pattern, filename_lower)
                        for pattern in general_patterns
                    )

                    if is_general_doc:
                        # General family docs are allowed for any architecture in that family
                        pass
                    else:
                        # Look for SPECIFIC architecture numbers (3-4 digits) in filename
                        # Match TC375, TC377, TC387, TC334, etc.
                        specific_arch_pattern = r"\btc(\d{3,4})\b"
                        specific_archs = re.findall(
                            specific_arch_pattern, filename_lower
                        )

                        # If filename has a SPECIFIC architecture reference
                        if specific_archs:
                            # Extract the architecture family number (TC37x, TC38x, etc.)
                            # from target_archs
                            target_families = set()
                            for target in target_archs:
                                # Extract family: tc37x -> 37, tc38x -> 38
                                family_match = re.match(r"tc(\d{2})", target)
                                if family_match:
                                    target_families.add(family_match.group(1))

                            # Check if any specific arch in filename matches our target family
                            filename_matches_target = False
                            for arch_num in specific_archs:
                                # Get first 2 digits of the specific arch (TC375 -> 37, TC387 -> 38)
                                arch_family = (
                                    arch_num[:2] if len(arch_num) >= 2 else arch_num
                                )

                                if arch_family in target_families:
                                    filename_matches_target = True
                                    break

                            # If filename specifies a different architecture family, exclude it
                            if not filename_matches_target:
                                if stats:
                                    stats.record_filter(
                                        f"wrong_architecture_filename: {specific_archs}"
                                    )
                                return False

                # Standard exact match for other custom filters
                elif metadata.get(key) != value:
                    if stats:
                        stats.record_filter(f"custom_filter_mismatch: {key}")
                    return False

        # Passed all filters
        if stats:
            stats.record_keep()
        return True

    def get_filter_summary(self) -> Dict[str, Any]:
        """Get human-readable summary of active filters."""
        summary = {}
        if self.source:
            summary["source"] = self.source
        if self.source_path:
            summary["source_path"] = self.source_path
        if self.page:
            summary["page"] = self.page
        if self.page_min or self.page_max:
            summary["page_range"] = (
                f"{self.page_min or 'start'}-{self.page_max or 'end'}"
            )
        if self.custom:
            if "_target_folders" in self.custom:
                summary["target_folders"] = self.custom["_target_folders"]
            summary["custom_filters"] = {
                k: v for k, v in self.custom.items() if k != "_target_folders"
            }
        return summary

    def to_chromadb_where(self) -> Optional[Dict[str, Any]]:
        """
        Convert filter to ChromaDB where clause format.

        Returns:
            ChromaDB-compatible where clause or None if no filters
        """
        conditions = []

        # Source filter (exact match for ChromaDB)
        if self.source is not None:
            conditions.append({"source_pdf": {"$eq": self.source}})

        # Page filters
        if self.page is not None:
            conditions.append({"page": {"$eq": self.page}})
        elif self.page_min is not None or self.page_max is not None:
            if self.page_min is not None and self.page_max is not None:
                conditions.append({"page": {"$gte": self.page_min}})
                conditions.append({"page": {"$lte": self.page_max}})
            elif self.page_min is not None:
                conditions.append({"page": {"$gte": self.page_min}})
            elif self.page_max is not None:
                conditions.append({"page": {"$lte": self.page_max}})

        # Custom filters (skip _target_folders as it needs post-processing)
        if self.custom:
            for key, value in self.custom.items():
                if key != "_target_folders":  # Skip folder filter for ChromaDB
                    conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def __bool__(self) -> bool:
        """Return True if any filter is set."""
        return any(
            [
                self.source is not None,
                self.source_path is not None,
                self.page is not None,
                self.page_min is not None,
                self.page_max is not None,
                self.custom is not None,
            ]
        )


# ============================================================================
# Filter Statistics and Transparency
# ============================================================================


@dataclass
class FilterStatistics:
    """
    Statistics about filtering operations for transparency and debugging.

    Tracks:
    - How many documents were considered
    - How many passed/failed filters
    - Reasons for filtering
    - Filter effectiveness metrics
    """

    total_candidates: int = 0
    filtered_count: int = 0
    kept_count: int = 0
    filter_reasons: Dict[str, int] = None

    def __post_init__(self):
        if self.filter_reasons is None:
            self.filter_reasons = {}

    def record_filter(self, reason: str):
        """Record why a document was filtered out."""
        self.filtered_count += 1
        self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1

    def record_keep(self):
        """Record that a document passed filters."""
        self.kept_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_candidates": self.total_candidates,
            "kept": self.kept_count,
            "filtered_out": self.filtered_count,
            "filter_effectiveness": f"{(self.filtered_count / max(self.total_candidates, 1)) * 100:.1f}%",
            "filter_reasons": self.filter_reasons,
        }


# ============================================================================
# Query Intent Classifier - Document Type Prioritization
# ============================================================================


class DocumentType:
    """Document type identifiers for prioritization."""

    DATASHEET = "datasheet"
    USER_MANUAL = "user_manual"
    KIT_MANUAL = "kit_manual"
    APPLICATION_NOTE = "application_note"
    EXPERT_TRAINING = "expert_training"
    QUICK_TRAINING = "quick_training"
    PROJECT_README = "project_readme"
    UNKNOWN = "unknown"


@dataclass
class QueryIntent:
    """
    Represents the classified intent of a user query.

    Attributes:
        primary_intent: Main intent category (e.g., "pin_verification", "peripheral_logic")
        document_priorities: Ordered list of document types to prioritize
        boost_weights: Dict mapping document type to boost multiplier
        detected_topics: List of technical topics detected in query
        confidence: Confidence score for the classification (0.0 - 1.0)
    """

    primary_intent: str
    document_priorities: List[str]
    boost_weights: Dict[str, float]
    detected_topics: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def get_boost_for_document(self, doc_type: str) -> float:
        """Get the boost weight for a specific document type."""
        return self.boost_weights.get(doc_type, 0.0)


class QueryIntentClassifier:
    """
    Classifies user queries to determine which document types should be prioritized.

    Implements the following workflow-based prioritization:

    1. Datasheet → verify pins, clocks, limits
    2. User Manual → understand peripheral logic
    3. Kit Manual → confirm board routing & jumpers
    4. Application Note → base implementation on it
    5. Expert Training → refine & optimize
    6. Quick Training → only if brand new to a topic
    """

    # Intent categories with their associated keywords and document priorities
    INTENT_PATTERNS = {
        # PIN/CLOCK/LIMITS VERIFICATION → Datasheet first
        "pin_verification": {
            "keywords": [
                "pin",
                "pinout",
                "pinmap",
                "pin map",
                "pin mapping",
                "pin assignment",
                "gpio",
                "port function",
                "alternate function",
                "alt function",
                "mux",
                "iomux",
                "egtm_tout",
                "gtm_tout",
                "tout",
                "tin",
                "ball",
                "package",
                "bga",
                "lqfp",
                "output select",
                "input select",
                "pad driver",
                "buffer type",
            ],
            "patterns": [r"\bP\d{1,2}\.\d{1,2}\b"],
            "priorities": [
                DocumentType.DATASHEET,
                DocumentType.USER_MANUAL,
                DocumentType.KIT_MANUAL,
                DocumentType.APPLICATION_NOTE,
                DocumentType.EXPERT_TRAINING,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.DATASHEET: 5.0,
                DocumentType.USER_MANUAL: 2.0,
                DocumentType.KIT_MANUAL: 1.5,
                DocumentType.APPLICATION_NOTE: 1.0,
                DocumentType.EXPERT_TRAINING: 0.5,
                DocumentType.QUICK_TRAINING: 0.3,
            },
        },
        "clock_verification": {
            "keywords": [
                "clock",
                "pll",
                "oscillator",
                "frequency",
                "mhz",
                "ghz",
                "clock tree",
                "clock domain",
                "clock source",
                "system clock",
                "peripheral clock",
                "clock divider",
                "prescaler",
                "clock gating",
                "ccu",
                "scu",
            ],
            "patterns": [r"\b\d+\s*mhz\b", r"\b\d+\s*ghz\b"],
            "priorities": [
                DocumentType.DATASHEET,
                DocumentType.USER_MANUAL,
                DocumentType.APPLICATION_NOTE,
                DocumentType.EXPERT_TRAINING,
                DocumentType.KIT_MANUAL,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.DATASHEET: 5.0,
                DocumentType.USER_MANUAL: 3.0,
                DocumentType.APPLICATION_NOTE: 2.0,
                DocumentType.EXPERT_TRAINING: 1.5,
                DocumentType.KIT_MANUAL: 1.0,
                DocumentType.QUICK_TRAINING: 0.3,
            },
        },
        "electrical_limits": {
            "keywords": [
                "voltage",
                "current",
                "power",
                "limit",
                "maximum",
                "minimum",
                "absolute",
                "vdd",
                "vss",
                "operating range",
                "temperature",
                "thermal",
                "esd",
                "electrical characteristics",
                "dc characteristics",
                "ac characteristics",
                "drive strength",
                "sink",
                "source",
                "impedance",
                "capacitance",
            ],
            "patterns": [r"\b\d+\.?\d*\s*[vm]a?\b", r"\b\d+\.?\d*\s*°?c\b"],
            "priorities": [
                DocumentType.DATASHEET,
                DocumentType.USER_MANUAL,
                DocumentType.KIT_MANUAL,
                DocumentType.APPLICATION_NOTE,
                DocumentType.EXPERT_TRAINING,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.5,
                DocumentType.DATASHEET: 6.0,
                DocumentType.USER_MANUAL: 2.0,
                DocumentType.KIT_MANUAL: 1.5,
                DocumentType.APPLICATION_NOTE: 1.0,
                DocumentType.EXPERT_TRAINING: 0.5,
                DocumentType.QUICK_TRAINING: 0.2,
            },
        },
        # PERIPHERAL LOGIC → User Manual first
        "peripheral_logic": {
            "keywords": [
                "how does",
                "how to configure",
                "peripheral",
                "module",
                "register",
                "configuration",
                "initialize",
                "setup",
                "enable",
                "disable",
                "mode",
                "operation",
                "function",
                "feature",
                "capability",
                "interrupt",
                "dma",
                "fifo",
                "buffer",
                "state machine",
            ],
            "patterns": [r"\bhow\s+(does|do|to)\b", r"\bwhat\s+is\b"],
            "priorities": [
                DocumentType.USER_MANUAL,
                DocumentType.DATASHEET,
                DocumentType.EXPERT_TRAINING,
                DocumentType.APPLICATION_NOTE,
                DocumentType.KIT_MANUAL,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.USER_MANUAL: 4.0,
                DocumentType.DATASHEET: 2.5,
                DocumentType.EXPERT_TRAINING: 2.0,
                DocumentType.APPLICATION_NOTE: 1.5,
                DocumentType.KIT_MANUAL: 0.5,
                DocumentType.QUICK_TRAINING: 0.5,
            },
        },
        "peripheral_specific": {
            "keywords": [
                "gtm",
                "tom",
                "atom",
                "egtm",
                "pwm",
                "timer",
                "counter",
                "adc",
                "vadc",
                "evadc",
                "dac",
                "analog",
                "can",
                "canfd",
                "lin",
                "flexray",
                "ethernet",
                "spi",
                "qspi",
                "i2c",
                "uart",
                "asclin",
                "dma",
                "gpdma",
                "hssl",
                "sent",
                "psi5",
                "ccu6",
                "gpt12",
                "stm",
                "eru",
                "ir",
            ],
            "patterns": [],
            "priorities": [
                DocumentType.USER_MANUAL,
                DocumentType.EXPERT_TRAINING,
                DocumentType.APPLICATION_NOTE,
                DocumentType.DATASHEET,
                DocumentType.PROJECT_README,
                DocumentType.KIT_MANUAL,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.USER_MANUAL: 4.0,
                DocumentType.EXPERT_TRAINING: 3.0,
                DocumentType.APPLICATION_NOTE: 2.5,
                DocumentType.DATASHEET: 2.0,
                DocumentType.KIT_MANUAL: 0.5,
                DocumentType.QUICK_TRAINING: 0.5,
            },
        },
        # BOARD ROUTING & JUMPERS → Kit Manual first
        "board_hardware": {
            "keywords": [
                "board",
                "kit",
                "jumper",
                "switch",
                "led",
                "button",
                "connector",
                "routing",
                "layout",
                "schematic",
                "silk screen",
                "header",
                "debug",
                "debugger",
                "jtag",
                "das",
                "miniwiggler",
                "trace",
                "power supply",
                "voltage regulator",
                "usb",
                "ethernet connector",
                "arduino",
                "shield",
                "expansion",
            ],
            "patterns": [r"\bj\d+\b", r"\bsw\d+\b", r"\bled\d+\b"],
            "priorities": [
                DocumentType.KIT_MANUAL,
                DocumentType.DATASHEET,
                DocumentType.USER_MANUAL,
                DocumentType.APPLICATION_NOTE,
                DocumentType.PROJECT_README,
                DocumentType.EXPERT_TRAINING,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.KIT_MANUAL: 5.0,
                DocumentType.DATASHEET: 2.0,
                DocumentType.USER_MANUAL: 1.5,
                DocumentType.APPLICATION_NOTE: 1.0,
                DocumentType.EXPERT_TRAINING: 0.5,
                DocumentType.QUICK_TRAINING: 0.3,
            },
        },
        # IMPLEMENTATION → Application Note first
        "implementation": {
            "keywords": [
                "implementation",
                "implement",
                "example",
                "code",
                "sample",
                "application",
                "use case",
                "project",
                "demo",
                "tutorial",
                "motor control",
                "inverter",
                "3-phase",
                "three phase",
                "bldc",
                "pmsm",
                "encoder",
                "resolver",
                "hall sensor",
                "sensorless",
                "dead time",
                "dtm",
                "complementary",
                "high side",
                "low side",
            ],
            "patterns": [r"\bhow\s+to\s+implement\b", r"\bexample\s+(code|project)\b"],
            "priorities": [
                DocumentType.APPLICATION_NOTE,
                DocumentType.PROJECT_README,
                DocumentType.EXPERT_TRAINING,
                DocumentType.USER_MANUAL,
                DocumentType.DATASHEET,
                DocumentType.KIT_MANUAL,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.APPLICATION_NOTE: 5.0,
                DocumentType.EXPERT_TRAINING: 3.0,
                DocumentType.USER_MANUAL: 2.0,
                DocumentType.DATASHEET: 1.0,
                DocumentType.KIT_MANUAL: 0.5,
                DocumentType.QUICK_TRAINING: 0.5,
            },
        },
        # OPTIMIZATION → Expert Training first
        "optimization": {
            "keywords": [
                "optimize",
                "optimization",
                "performance tuning",
                "efficient",
                "efficiency",
                "best practice",
                "advanced technique",
                "expert level",
                "fine-tune",
                "tuning",
                "refine",
                "improve performance",
                "reduce latency",
                "throughput",
                "benchmark",
                "faster",
                "speed up",
                "bottleneck",
                "profiling",
            ],
            "patterns": [
                r"\bbest\s+(practice|way)\b",
                r"\bhow\s+to\s+optimize\b",
                r"\bimprove\s+performance\b",
            ],
            "priorities": [
                DocumentType.EXPERT_TRAINING,
                DocumentType.APPLICATION_NOTE,
                DocumentType.USER_MANUAL,
                DocumentType.PROJECT_README,
                DocumentType.DATASHEET,
                DocumentType.KIT_MANUAL,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.EXPERT_TRAINING: 5.0,
                DocumentType.APPLICATION_NOTE: 3.0,
                DocumentType.USER_MANUAL: 2.0,
                DocumentType.DATASHEET: 1.0,
                DocumentType.KIT_MANUAL: 0.5,
                DocumentType.QUICK_TRAINING: 0.5,
            },
        },
        # BEGINNER/INTRODUCTION → Quick Training first
        "beginner_introduction": {
            "keywords": [
                "beginner",
                "introduction to",
                "intro to",
                "getting started",
                "basics of",
                "overview of",
                "explain basics",
                "understand basics",
                "learn about",
                "new to aurix",
                "first time",
                "start with",
                "tutorial",
                "beginners guide",
            ],
            "patterns": [
                r"\bwhat\s+is\s+(a|an|the)?\s*\w+\b",
                r"\bintro(duction)?\s+to\b",
            ],
            "priorities": [
                DocumentType.QUICK_TRAINING,
                DocumentType.USER_MANUAL,
                DocumentType.EXPERT_TRAINING,
                DocumentType.APPLICATION_NOTE,
                DocumentType.DATASHEET,
                DocumentType.KIT_MANUAL,
                DocumentType.PROJECT_README,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.QUICK_TRAINING: 4.0,
                DocumentType.USER_MANUAL: 3.0,
                DocumentType.EXPERT_TRAINING: 2.0,
                DocumentType.APPLICATION_NOTE: 1.5,
                DocumentType.DATASHEET: 1.0,
                DocumentType.KIT_MANUAL: 1.0,
            },
        },
        # MIGRATION → Cross-reference multiple doc types
        "migration": {
            "keywords": [
                "migrate",
                "migration",
                "upgrade",
                "port",
                "porting",
                "move from",
                "transition",
                "convert",
                "switch from",
                "tc3xx to tc4xx",
                "tc4xx",
                "compatibility",
                "breaking change",
            ],
            "patterns": [r"\bfrom\s+tc\d+\s+to\s+tc\d+\b", r"\bmigrate?\s+(from|to)\b"],
            "priorities": [
                DocumentType.USER_MANUAL,
                DocumentType.DATASHEET,
                DocumentType.APPLICATION_NOTE,
                DocumentType.EXPERT_TRAINING,
                DocumentType.PROJECT_README,
                DocumentType.KIT_MANUAL,
                DocumentType.QUICK_TRAINING,
            ],
            "boost_weights": {
                DocumentType.PROJECT_README: 7.0,
                DocumentType.USER_MANUAL: 4.0,
                DocumentType.DATASHEET: 3.5,
                DocumentType.APPLICATION_NOTE: 3.0,
                DocumentType.EXPERT_TRAINING: 2.5,
                DocumentType.KIT_MANUAL: 1.0,
                DocumentType.QUICK_TRAINING: 0.5,
            },
        },
    }

    # Document type detection patterns for filenames
    DOCUMENT_TYPE_PATTERNS = {
        DocumentType.DATASHEET: [
            r"datasheet",
            r"data-sheet",
            r"data_sheet",
            r"-ds-",
            r"_ds_",
            r"-ds\.",
            r"_ds\.",
        ],
        DocumentType.USER_MANUAL: [
            r"user[\s_-]?manual",
            r"usermanual",
            r"um[\s_-]",
            r"-um-",
            r"_um_",
            r"reference[\s_-]?manual",
            r"programming[\s_-]?manual",
            r"user[\s_-]?guide",
        ],
        DocumentType.KIT_MANUAL: [
            r"kit[\s_-]?manual",
            r"kit-manual",
            r"kit_manual",
            r"board[\s_-]?manual",
            r"hw[\s_-]?manual",
            r"hardware[\s_-]?manual",
            r"development[\s_-]?kit",
        ],
        DocumentType.APPLICATION_NOTE: [
            r"application[\s_-]?note",
            r"app[\s_-]?note",
            r"appnote",
            r"-an-",
            r"_an_",
            r"-an\.",
            r"_an\.",
            r"application[\s_-]?report",
        ],
        DocumentType.EXPERT_TRAINING: [
            r"expert[\s_-]?training",
            r"advanced[\s_-]?training",
            r"expert_training",
            r"deep[\s_-]?dive",
        ],
        DocumentType.QUICK_TRAINING: [
            r"quick[\s_-]?training",
            r"quick_training",
            r"getting[\s_-]?started",
            r"intro[\s_-]?training",
            r"basic[\s_-]?training",
            r"starter[\s_-]?guide",
        ],
        DocumentType.PROJECT_README: [r"readme", r"_readme", r"project[\s_-]?readme"],
    }

    def __init__(self):
        """Initialize the query intent classifier."""
        # Compile regex patterns for efficiency
        self._compiled_patterns = {}
        for intent, config in self.INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in config.get("patterns", [])
            ]

        self._doc_type_patterns = {}
        for doc_type, patterns in self.DOCUMENT_TYPE_PATTERNS.items():
            self._doc_type_patterns[doc_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def classify(self, query: str) -> QueryIntent:
        """
        Classify a query to determine document type priorities.
        """
        query_lower = query.lower()
        intent_scores = {}
        detected_topics = []

        # Technical indicators that suggest this is not a beginner query
        technical_indicators = [
            "frequency",
            "limit",
            "maximum",
            "minimum",
            "clock",
            "voltage",
            "current",
            "register",
            "configuration",
            "peripheral",
            "pin",
            "timer",
            "pwm",
            "adc",
            "can",
            "spi",
            "uart",
            "dma",
            "interrupt",
            "gtm",
            "tom",
            "atom",
            "migration",
        ]
        has_technical_indicator = any(t in query_lower for t in technical_indicators)

        # Score each intent pattern
        for intent_name, config in self.INTENT_PATTERNS.items():
            score = 0.0

            # Check keywords
            keywords = config.get("keywords", [])
            for keyword in keywords:
                if keyword in query_lower:
                    if keyword in technical_indicators:
                        score += 2.0  # Technical keywords get higher weight
                    else:
                        score += 1.0
                    if keyword not in detected_topics:
                        detected_topics.append(keyword)

            # Check regex patterns
            patterns = self._compiled_patterns.get(intent_name, [])
            for pattern in patterns:
                if pattern.search(query):
                    # Special case: beginner queries with technical indicators are less likely
                    if (
                        intent_name == "beginner_introduction"
                        and has_technical_indicator
                    ):
                        score += 0.5  # Reduce score for beginner intent if technical content present
                    else:
                        score += 2.0

            if score > 0:
                intent_scores[intent_name] = score

        # If no intent matched, return general intent
        if not intent_scores:
            return QueryIntent(
                primary_intent="general",
                document_priorities=[
                    DocumentType.USER_MANUAL,
                    DocumentType.DATASHEET,
                    DocumentType.APPLICATION_NOTE,
                    DocumentType.EXPERT_TRAINING,
                    DocumentType.KIT_MANUAL,
                    DocumentType.QUICK_TRAINING,
                    DocumentType.PROJECT_README,
                ],
                boost_weights={
                    DocumentType.PROJECT_README: 7.0,
                    DocumentType.USER_MANUAL: 2.0,
                    DocumentType.DATASHEET: 2.0,
                    DocumentType.APPLICATION_NOTE: 1.5,
                    DocumentType.EXPERT_TRAINING: 1.5,
                    DocumentType.KIT_MANUAL: 1.0,
                    DocumentType.QUICK_TRAINING: 1.0,
                },
                detected_topics=detected_topics,
                confidence=0.5,
            )

        # Get the best intent match
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent_name, score = best_intent
        config = self.INTENT_PATTERNS[intent_name]

        # Calculate confidence
        max_possible = (
            len(config.get("keywords", [])) + len(config.get("patterns", [])) * 2
        )
        confidence = min(score / max(max_possible, 1), 1.0)

        # Merge boost weights from secondary intents
        merged_weights = dict(config["boost_weights"])
        for other_intent, other_score in intent_scores.items():
            if other_intent != intent_name and other_score > 0:
                other_config = self.INTENT_PATTERNS[other_intent]
                # Proportional weight based on secondary score
                weight_factor = other_score / (score + other_score)
                for doc_type, boost in other_config["boost_weights"].items():
                    if doc_type in merged_weights:
                        merged_weights[doc_type] += boost * weight_factor * 0.3
                    else:
                        merged_weights[doc_type] = boost * weight_factor * 0.3

        return QueryIntent(
            primary_intent=intent_name,
            document_priorities=config["priorities"],
            boost_weights=merged_weights,
            detected_topics=detected_topics,
            confidence=confidence,
        )

    def detect_document_type(
        self, filename: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Detect the document type from filename and/or metadata.
        """
        filename_lower = filename.lower()

        # Check metadata first if available
        if metadata:
            doc_type = (
                metadata.get("doc_type") or metadata.get("document_type") or ""
            ).lower()
            if "project" in doc_type or "readme" in doc_type:
                return DocumentType.PROJECT_README
            if "datasheet" in doc_type:
                return DocumentType.DATASHEET

        # Pattern matching on filename
        for doc_type, patterns in self._doc_type_patterns.items():
            for pattern in patterns:
                if pattern.search(filename_lower):
                    return doc_type

        # Check metadata source path as fallback
        if metadata:
            source_path = metadata.get("source_path", "").lower()
            if "application_note" in source_path or "application-note" in source_path:
                return DocumentType.APPLICATION_NOTE
            if "expert_training" in source_path:
                return DocumentType.EXPERT_TRAINING
            if "quick_training" in source_path:
                return DocumentType.QUICK_TRAINING
            if "kit_manual" in source_path:
                return DocumentType.KIT_MANUAL
            if "user_manual" in source_path:
                return DocumentType.USER_MANUAL

        return DocumentType.UNKNOWN

    def get_boost_for_document(
        self,
        query_intent: QueryIntent,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate the boost value for a document based on query intent.
        """
        doc_type = self.detect_document_type(filename, metadata)
        return query_intent.get_boost_for_document(doc_type)


# Global classifier instance for reuse
_QUERY_INTENT_CLASSIFIER: Optional[QueryIntentClassifier] = None


def get_query_intent_classifier() -> QueryIntentClassifier:
    """Get or create the global query intent classifier instance."""
    global _QUERY_INTENT_CLASSIFIER
    if _QUERY_INTENT_CLASSIFIER is None:
        _QUERY_INTENT_CLASSIFIER = QueryIntentClassifier()
    return _QUERY_INTENT_CLASSIFIER


# ============================================================================
# Document Folder Manager - Dynamic Architecture Selection
# ============================================================================


class DocumentFolderManager:
    """
    Manages dynamic document folder selection based on Infineon product architecture.

    Folder Structure Expected:
        docs/
        ├── AURIX/
        │   ├── AURIX TC3xx/
        │   │   ├── TC33x-TC32x/
        │   │   ├── TC35x/
        │   │   ├── TC36x/
        │   │   ├── TC37x/
        │   │   ├── TC37xEXT/
        │   │   ├── TC38x/
        │   │   ├── TC39x/
        │   │   └── *.pdf (common TC3xx docs)
        │   └── AURIX TC4xx/
        │       └── *.pdf
        └── Other product families...

    Features:
        - List available product families and architectures
        - Select specific architecture folders for retrieval
        - Automatically include parent folder documents (e.g., TC38x includes TC3xx common docs)
        - Detect architecture requirements from user queries
        - Handle migration queries that span multiple architectures

    Examples:
        manager = DocumentFolderManager(docs_root)

        # List available architectures
        manager.list_folder_structure()

        # Select TC38x - includes TC38x folder + TC3xx parent PDFs
        folders = manager.get_folders_for_architecture("TC38x")

        # Detect from query
        folders = manager.detect_architecture_from_query("How to configure TC38x GPIO?")

        # Migration query - spans multiple families
        folders = manager.detect_architecture_from_query("How to migrate from TC387 to TC4Dx?")
    """

    def __init__(self, docs_root: Path):
        """
        Initialize the folder manager.

        Args:
            docs_root: Root directory containing document folders (e.g., pipeline/src/doc_context/RAG/docs)
        """
        self.docs_root = Path(docs_root)
        self._folder_cache: Optional[Dict[str, Any]] = None
        self._architecture_patterns: Optional[Dict[str, str]] = None
        self._family_hierarchy: Optional[Dict[str, str]] = None

    @property
    def ARCHITECTURE_PATTERNS(self) -> Dict[str, str]:
        """
        Dynamically build architecture detection patterns from folder structure.
        Patterns are ordered: specific 4-digit variants first, then generic families.
        """
        if self._architecture_patterns is not None:
            return self._architecture_patterns

        patterns = {}
        structure = self.get_folder_structure()

        # Scan for architecture folder names and PDF filenames
        discovered_archs = set()
        discovered_specific = set()  # 4-digit variants like TC375, TC387

        def scan_for_architectures(data: Dict[str, Any], path_chain: List[str] = []):
            for name, folder_data in data.items():
                # Check folder name for architecture patterns
                folder_archs = self._extract_architectures_from_name(name)
                discovered_archs.update(folder_archs)

                # Scan PDF filenames for specific architecture references
                for pdf in folder_data.get("pdfs", []):
                    pdf_archs = self._extract_architectures_from_name(pdf)
                    discovered_archs.update(pdf_archs)
                    # Look for specific variants (3-4 digit numbers)
                    specific = re.findall(r"\bTC(\d{3,4})\b", pdf, re.IGNORECASE)
                    for s in specific:
                        discovered_specific.add(f"TC{s}")

                # Recurse into children
                if folder_data.get("children"):
                    scan_for_architectures(folder_data["children"], path_chain + [name])

        scan_for_architectures(structure)

        # Build patterns - specific variants first (highest priority)
        # Use (?:^|[^a-zA-Z0-9]) and (?:[^a-zA-Z0-9]|$) instead of \b to handle underscores
        for arch in sorted(discovered_specific, reverse=True):
            # Pattern that matches TC387 at start, end, or surrounded by non-alphanumeric (including underscore)
            patterns[rf"(?:^|[^a-zA-Z0-9]){arch}(?:[^a-zA-Z0-9]|$)"] = arch

        # Add generic family patterns discovered from folders
        for arch in sorted(discovered_archs):
            # Skip if already covered by specific variant
            if arch in discovered_specific:
                continue

            # Handle different architecture naming conventions
            if "TC33x-TC32x" in arch or "TC32" in arch or "TC33" in arch:
                patterns[
                    r"(?:^|[^a-zA-Z0-9])TC3[23]x(?:[^a-zA-Z0-9]|$)|(?:^|[^a-zA-Z0-9])TC32[x\d](?:[^a-zA-Z0-9]|$)|(?:^|[^a-zA-Z0-9])TC33[x\d](?:[^a-zA-Z0-9]|$)"
                ] = "TC33x-TC32x"
            elif re.match(r"TC3\d+x", arch, re.IGNORECASE):
                # Generic TC3xx subfamilies: TC35x, TC36x, TC37x, TC38x, TC39x
                patterns[rf"(?:^|[^a-zA-Z0-9]){arch}(?:[^a-zA-Z0-9]|$)"] = arch
            elif "TC4" in arch.upper():
                # TC4xx family - handle TC4Dx, TC4D7, TC4xx variants
                patterns[
                    r"(?:^|[^a-zA-Z0-9])TC4[Dd]?[x\dX](?:[^a-zA-Z0-9]|$)|(?:^|[^a-zA-Z0-9])TC4[xX]{2}(?:[^a-zA-Z0-9]|$)"
                ] = "AURIX TC4xx"
            elif "EXT" in arch.upper():
                # Extended variants like TC33xEXT, TC37xEXT
                patterns[rf"(?:^|[^a-zA-Z0-9]){arch}(?:[^a-zA-Z0-9]|$)"] = arch
            else:
                patterns[rf"(?:^|[^a-zA-Z0-9]){arch}(?:[^a-zA-Z0-9]|$)"] = arch

        self._architecture_patterns = patterns
        return patterns

    @property
    def FAMILY_HIERARCHY(self) -> Dict[str, str]:
        """
        Dynamically build family hierarchy mapping from folder structure.
        Maps specific variants to their parent folders.
        """
        if self._family_hierarchy is not None:
            return self._family_hierarchy

        hierarchy = {}
        structure = self.get_folder_structure()

        def build_hierarchy(data: Dict[str, Any], parent_path: List[str] = []):
            for name, folder_data in data.items():
                current_path = parent_path + [name]

                # Check if this is an architecture folder
                if self._is_architecture_folder(name):
                    # Find parent architecture folder
                    parent_arch = None
                    for parent in reversed(parent_path):
                        if self._is_architecture_folder(parent):
                            parent_arch = parent
                            break

                    if parent_arch:
                        hierarchy[name] = parent_arch

                # Recurse into children
                if folder_data.get("children"):
                    build_hierarchy(folder_data["children"], current_path)

        build_hierarchy(structure)

        # Add mappings for specific variants discovered from PDF filenames
        for arch in self.ARCHITECTURE_PATTERNS.values():
            if re.match(r"TC\d{3,4}$", arch):  # Specific 4-digit variant
                # Find parent family (TC375 -> TC37x, TC387 -> TC38x)
                family_num = arch[2:4]  # Extract "37" from "TC375"
                parent_folder = f"TC{family_num}x"
                if parent_folder not in hierarchy.get(arch, ""):
                    hierarchy[arch] = parent_folder

        # Ensure all TC3xx subfolders map to AURIX TC3xx
        tc3xx_families = [
            "TC33x-TC32x",
            "TC33xEXT",
            "TC35x",
            "TC36x",
            "TC37x",
            "TC37xEXT",
            "TC38x",
            "TC39x",
        ]
        for family in tc3xx_families:
            if family not in hierarchy:
                hierarchy[family] = "AURIX TC3xx"

        # TC4xx maps to itself (top level)
        hierarchy["AURIX TC4xx"] = "AURIX TC4xx"

        self._family_hierarchy = hierarchy
        return hierarchy

    def _extract_architectures_from_name(self, name: str) -> List[str]:
        """Extract architecture identifiers from a folder or file name."""
        archs = []

        # Pattern for TC families: TC3xx, TC4xx, TC37x, TC38x, etc.
        patterns = [
            (r"\bTC3[23]x[-_]?TC3[23]x\b", "TC33x-TC32x"),  # Combined folder
            (
                r"\bTC(\d{2})x(?:EXT)?\b",
                lambda m: f"TC{m.group(1)}x{'EXT' if 'EXT' in name.upper() else ''}",
            ),  # TC37x, TC38x, TC37xEXT
            (
                r"\bAURIX[_\s-]?TC([34])xx\b",
                lambda m: f"AURIX TC{m.group(1)}xx",
            ),  # AURIX TC3xx, AURIX TC4xx
            (r"\bTC4[Dd]?x+\b", "AURIX TC4xx"),  # TC4xx, TC4Dx
        ]

        for pattern, result in patterns:
            matches = re.finditer(pattern, name, re.IGNORECASE)
            for match in matches:
                if callable(result):
                    archs.append(result(match))
                else:
                    archs.append(result)

        return archs

    def _is_architecture_folder(self, name: str) -> bool:
        """Check if a folder name represents an architecture."""
        arch_patterns = [
            r"\bTC\d{2}x",  # TC37x, TC38x, etc.
            r"\bAURIX[_\s-]?TC[34]xx",  # AURIX TC3xx, AURIX TC4xx
            r"\bTC3[23]x[-_]TC3[23]x",  # TC33x-TC32x
            r"EXT$",  # Extended variants
        ]
        return any(re.search(p, name, re.IGNORECASE) for p in arch_patterns)

    def get_folder_structure(self) -> Dict[str, Any]:
        """
        Scan and return the folder structure.

        Returns:
            Nested dict representing folder hierarchy:
            {
                "AURIX": {
                    "path": Path(...),
                    "pdfs": ["file1.pdf", ...],
                    "children": {
                        "AURIX TC3xx": {
                            "path": Path(...),
                            "pdfs": [...],
                            "children": {
                                "TC37x": {...},
                                ...
                            }
                        }
                    }
                }
            }
        """
        if self._folder_cache is not None:
            return self._folder_cache

        def scan_folder(path: Path, depth: int = 0) -> Dict[str, Any]:
            result = {"path": path, "pdfs": [], "children": {}}

            if not path.exists():
                return result

            for item in path.iterdir():
                if item.is_file() and item.suffix.lower() == ".pdf":
                    result["pdfs"].append(item.name)
                elif item.is_dir() and not item.name.startswith("."):
                    result["children"][item.name] = scan_folder(item, depth + 1)

            return result

        structure = {}
        if self.docs_root.exists():
            for item in self.docs_root.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    structure[item.name] = scan_folder(item)

        self._folder_cache = structure
        return structure

    def list_folder_structure(self, show_files: bool = False) -> str:
        """
        Generate a formatted string showing the folder structure.

        Args:
            show_files: Whether to include PDF file names

        Returns:
            Formatted string for display
        """
        structure = self.get_folder_structure()
        lines = ["📁 Document Folder Structure", "=" * 50, ""]

        def format_folder(name: str, data: Dict[str, Any], indent: int = 0):
            prefix = "  " * indent
            folder_icon = "📂" if data["children"] else "📁"
            pdf_count = len(data["pdfs"])

            line = f"{prefix}{folder_icon} {name}"
            if pdf_count > 0:
                line += f" ({pdf_count} PDFs)"
            lines.append(line)

            if show_files and data["pdfs"]:
                for pdf in sorted(data["pdfs"]):
                    lines.append(f"{prefix}  📄 {pdf}")

            for child_name, child_data in sorted(data["children"].items()):
                format_folder(child_name, child_data, indent + 1)

        for family_name, family_data in sorted(structure.items()):
            format_folder(family_name, family_data)
            lines.append("")

        return "\n".join(lines)

    def get_available_architectures(self) -> List[Dict[str, Any]]:
        """
        Get list of all available architectures with their paths.

        Returns:
            List of dicts: [{"name": "TC38x", "family": "AURIX TC3xx", "path": Path(...), "pdf_count": 5}, ...]
        """
        structure = self.get_folder_structure()
        architectures = []

        def find_architectures(data: Dict[str, Any], family_path: List[str] = []):
            for name, folder_data in data.items():
                current_path = family_path + [name]

                # Check if this is a leaf architecture folder (has PDFs, or is a known architecture)
                has_architecture_pattern = any(
                    re.match(pattern, name, re.IGNORECASE)
                    for pattern in self.ARCHITECTURE_PATTERNS.keys()
                )

                if has_architecture_pattern or (
                    folder_data["pdfs"] and not folder_data["children"]
                ):
                    architectures.append(
                        {
                            "name": name,
                            "family": (
                                "/".join(current_path[:-1])
                                if len(current_path) > 1
                                else "Root"
                            ),
                            "path": folder_data["path"],
                            "pdf_count": len(folder_data["pdfs"]),
                        }
                    )

                # Recurse into children
                if folder_data["children"]:
                    find_architectures(folder_data["children"], current_path)

        find_architectures(structure)
        return architectures

    # Common resource folders that apply to ALL architectures within a family
    # These contain documents relevant across sub-architectures (application notes, trainings, etc.)
    COMMON_RESOURCE_FOLDERS = [
        "application_note",
        "application_notes",
        "expert_trainings",
        "kit_manuals",
        "user_manuals",
        "quick_trainings",
        "TC4xx_projects",  # TC4xx example projects with pinout, code examples
        "TC3xx_projects",  # TC3xx example projects with pinout, code examples
    ]

    def get_folders_for_architecture(self, architecture: str) -> List[Path]:
        """
        Get folders to search for a specific architecture.
        Includes:
        - The architecture-specific folder (e.g., TC38x)
        - Parent family folder (e.g., AURIX TC3xx) for common PDFs
        - Common resource folders (application_note, expert_trainings, kit_manuals, etc.)

        Args:
            architecture: Architecture name (e.g., "TC375", "TC387", "TC38x", "TC37x", "TC4xx")

        Returns:
            List of folder Paths to include in retrieval
        """
        folders = []
        structure = self.get_folder_structure()

        # If this is a specific 4-digit variant (TC375, TC387, etc.),
        # resolve it to its parent folder (TC37x, TC38x)
        resolved_arch = architecture
        if architecture in self.FAMILY_HIERARCHY:
            parent_arch = self.FAMILY_HIERARCHY[architecture]
            # For specific variants like TC375 -> TC37x, use the parent folder
            if parent_arch != "AURIX TC3xx" and parent_arch != "AURIX TC4xx":
                resolved_arch = parent_arch

        def find_architecture_folder(
            data: Dict[str, Any], target: str
        ) -> Optional[Tuple[Path, List[str]]]:
            """Find folder and return (path, parent_chain)."""
            for name, folder_data in data.items():
                if name.lower() == target.lower() or target.lower() in name.lower():
                    return (folder_data["path"], [])

                if folder_data["children"]:
                    result = find_architecture_folder(folder_data["children"], target)
                    if result:
                        path, parents = result
                        return (path, [name] + parents)
            return None

        def get_family_folder_data(
            data: Dict[str, Any], parents: List[str]
        ) -> Optional[Dict[str, Any]]:
            """Navigate to the parent family folder and return its data."""
            current = data
            for parent_name in parents:
                if parent_name in current:
                    current = current[parent_name]
                elif (
                    isinstance(current, dict)
                    and "children" in current
                    and parent_name in current["children"]
                ):
                    current = current["children"][parent_name]
                else:
                    return None
            return current

        # Find the target architecture folder
        result = find_architecture_folder(structure, resolved_arch)

        if result:
            arch_path, parents = result
            folders.append(arch_path)

            # Add parent family folder and common resource folders
            if parents:
                # Check if this is a TC3xx or TC4xx sub-architecture
                is_tc3xx_subfamily = any(
                    parent.lower() in ["aurix tc3xx", "tc3xx"] for parent in parents
                )
                is_tc4xx_subfamily = any(
                    parent.lower() in ["aurix tc4xx", "tc4xx"] for parent in parents
                )

                # Get the family folder data (e.g., AURIX TC3xx)
                family_data = get_family_folder_data(structure, parents)

                if family_data and isinstance(family_data, dict):
                    # Include parent folder itself (for PDFs at parent level)
                    if family_data.get("path") and (
                        is_tc3xx_subfamily or is_tc4xx_subfamily
                    ):
                        folders.insert(0, family_data["path"])

                    # Add common resource folders (application_note, expert_trainings, etc.)
                    children = family_data.get("children", {})
                    for folder_name, folder_data in children.items():
                        # Check if this is a common resource folder
                        if folder_name.lower() in [
                            f.lower() for f in self.COMMON_RESOURCE_FOLDERS
                        ]:
                            if (
                                folder_data.get("path")
                                and folder_data["path"] not in folders
                            ):
                                folders.append(folder_data["path"])

        return folders

    def detect_architecture_from_query(
        self, query: str
    ) -> Tuple[List[Path], List[str]]:
        """
        Detect which architecture(s) a query is asking about.

        Args:
            query: User's query string

        Returns:
            Tuple of (list of folder Paths, list of detected architecture names)
        """
        detected = []

        # Check each architecture pattern
        for pattern, arch_name in self.ARCHITECTURE_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                if arch_name not in detected:
                    detected.append(arch_name)

        # Get folders for all detected architectures
        all_folders = []
        for arch in detected:
            folders = self.get_folders_for_architecture(arch)
            for folder in folders:
                if folder not in all_folders:
                    all_folders.append(folder)

        # If no architecture detected, return empty (will use all docs)
        return all_folders, detected

    def detect_migration_query(self, query: str) -> Tuple[bool, List[str]]:
        """
        Detect if this is a migration query between architectures.

        Args:
            query: User's query string

        Returns:
            Tuple of (is_migration, list of detected architectures)
        """
        migration_keywords = [
            "migrate",
            "migration",
            "upgrade",
            "port",
            "porting",
            "move from",
            "transition",
            "convert",
            "switch from",
        ]

        is_migration = any(keyword in query.lower() for keyword in migration_keywords)

        # Detect all architectures mentioned
        _, detected = self.detect_architecture_from_query(query)

        # If migration with multiple architectures, it's definitely a migration query
        if is_migration and len(detected) >= 2:
            return True, detected
        elif is_migration and len(detected) == 1:
            # Single architecture mentioned but migration keyword - might be general migration guide
            return True, detected

        return False, detected

    def detect_pinout_query(self, query: str) -> bool:
        """
        Detect if this is a pin/pinout related query that should check datasheets.

        Args:
            query: User's query string

        Returns:
            True if this is a pinout-related query
        """
        # Pin-related keywords and patterns
        pinout_keywords = [
            "pin",
            "pinout",
            "pinmap",
            "pin map",
            "pin mapping",
            "pin assignment",
            "port",
            "gpio",
            "alternate function",
            "alt function",
            "multiplexing",
            "mux",
            "iomux",
            "output select",
            "input select",
            "egtm_tout",
            "gtm_tout",
            "tout",
            "tin",
            "ball",
            "package",
            "bga",
            "lqfp",
            "qfp",
        ]

        # Check for pin notation patterns like P10.3, P02.5, etc.
        pin_pattern = r"\bP\d{1,2}\.\d{1,2}\b"

        query_lower = query.lower()

        # Check keywords
        has_pinout_keyword = any(kw in query_lower for kw in pinout_keywords)

        # Check pin notation pattern
        has_pin_notation = bool(re.search(pin_pattern, query, re.IGNORECASE))

        return has_pinout_keyword or has_pin_notation

    def get_datasheet_sources_for_architecture(
        self, architectures: List[str]
    ) -> List[str]:
        """
        Get datasheet filenames for the given architecture(s).

        Datasheets contain the authoritative pin function tables that should be
        prioritized for pin/pinout related queries.

        Args:
            architectures: List of detected architecture names (e.g., ["AURIX TC4xx"])

        Returns:
            List of datasheet filename patterns to prioritize
        """
        datasheet_patterns = []

        for arch in architectures:
            arch_upper = arch.upper()

            # TC4xx family datasheets
            if "TC4" in arch_upper or "TC4XX" in arch_upper or "TC4DX" in arch_upper:
                datasheet_patterns.append("infineon-aurix-tc4dx-a-datasheet-en.pdf")

            # TC38x family datasheets
            if "TC38" in arch_upper:
                datasheet_patterns.append("tc38x-datasheet")
                datasheet_patterns.append("infineon-aurix-tc38x")

            # TC37x family datasheets
            if "TC37" in arch_upper:
                datasheet_patterns.append("tc37x-datasheet")
                datasheet_patterns.append("infineon-aurix-tc37x")

            # TC36x family datasheets
            if "TC36" in arch_upper:
                datasheet_patterns.append("tc36x-datasheet")
                datasheet_patterns.append("infineon-aurix-tc36x")

            # TC35x family datasheets
            if "TC35" in arch_upper:
                datasheet_patterns.append("tc35x-datasheet")
                datasheet_patterns.append("infineon-aurix-tc35x")

            # Generic TC3xx if specific subfamily not detected
            if "TC3XX" in arch_upper and not any(
                x in arch_upper for x in ["TC38", "TC37", "TC36", "TC35"]
            ):
                datasheet_patterns.append("tc3xx-datasheet")
                datasheet_patterns.append("infineon-aurix-tc3xx")

        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for p in datasheet_patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)

        return unique_patterns

    def get_cli_selection_menu(self) -> str:
        """
        Generate a CLI menu for architecture selection.

        Returns:
            Formatted menu string
        """
        architectures = self.get_available_architectures()
        lines = ["", "🔧 Available Architectures for Retrieval", "=" * 50, ""]

        # Group by family
        families = {}
        for arch in architectures:
            family = arch["family"]
            if family not in families:
                families[family] = []
            families[family].append(arch)

        idx = 1
        index_map = {}

        for family, archs in sorted(families.items()):
            lines.append(f"📦 {family}")
            for arch in sorted(archs, key=lambda x: x["name"]):
                lines.append(f"  [{idx}] {arch['name']} ({arch['pdf_count']} PDFs)")
                index_map[idx] = arch
                idx += 1
            lines.append("")

        lines.append("[0] Use ALL documents (no filter)")
        lines.append("[q] Quit selection")
        lines.append("")

        return "\n".join(lines), index_map


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RAGAgentConfig:
    """Configuration for the Infineon RAG Agent."""

    # Document paths
    # Default: pipeline/src/doc_context/RAG/docs (structured by product family)

    documents_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "RAG" / "docs")
    output_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "rag_agent_output")

    # Target folders for retrieval (dynamically selected based on user's architecture choice)
    # When empty, uses all documents. When set, focuses retrieval on these specific folders.
    target_folders: List[Path] = field(default_factory=list)

    # Text chunking configuration
    chunk_size: int = 2000
    chunk_overlap: int = 400

    # Image extraction configuration (matching ExtractionConfig)
    enable_image_extraction: bool = True
    min_image_area: float = 10000  # Minimum area in pixels^2
    min_diagram_area_pct: float = (
        0.01  # Minimum 1% of page area (from ExtractionConfig)
    )
    detection_dpi: int = 150
    extraction_dpi: int = 200
    scale_factor: float = 0.5  # Scale factor for faster detection
    enable_caption_detection: bool = True  # Include captions in extracted images
    use_jpeg_intermediate: bool = True  # Use JPEG for faster saves
    jpeg_quality: int = 90
    max_diagrams_per_page: int = 20  # Safety limit

    # CLIP configuration
    enable_clip_embeddings: bool = True
    clip_model: str = "ViT-B-32"
    image_db_path: str = field(
        default_factory=lambda: str(SCRIPT_DIR / "rag_agent_chroma_db")
    )

    # Query configuration
    text_top_k: int = 10
    image_top_k: int = 5
    vlm_model: str = "gpt-5"
    max_answer_tokens: int = 16000
    image_detail: str = "low"  # "low" for speed, "high" for accuracy
    max_images_in_answer: int = 3  # Max images to include in answer generation

    # Chunk-Image Association
    associate_images_with_chunks: bool = True  # Link images to chunks on same page
    include_nearby_pages: int = (
        0  # Include images from N pages before/after (0 = same page only)
    )

    # Processing
    force_reprocess: bool = False


# ============================================================================
# Text Chunker
# ============================================================================


class TextChunker:
    """Extract and chunk text from PDF documents."""

    def __init__(self, config: RAGAgentConfig):
        self.config = config
        self._fitz = None
        self._init_pdf_lib()

    def _init_pdf_lib(self):
        """Initialize PDF library."""
        try:
            import fitz  # PyMuPDF

            self._fitz = fitz
            logger.info("PDF library initialized (PyMuPDF)")
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install pymupdf")
            raise

    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[int, str]:
        """
        Extract text from PDF, organized by page.

        Returns:
            Dict mapping page number (1-indexed) to page text
        """
        doc = self._fitz.open(str(pdf_path))
        pages = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages[page_num + 1] = text

        doc.close()
        logger.info(f"Extracted text from {len(pages)} pages in {pdf_path.name}")
        return pages

    def chunk_text(
        self, text: str, source: str = "", source_path: str = "", page: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            source: Source filename (e.g., "manual.pdf")
            source_path: Full path to source file
            page: Page number

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text) < 50:
            return []

        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = chunk_size - overlap

        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "source_path": source_path,
                        "page": page,
                        "chunk_index": chunk_idx,
                        "char_start": start,
                        "char_end": end,
                        "hash": hashlib.md5(chunk_text.encode()).hexdigest(),
                    }
                )
                chunk_idx += 1

            if end == len(text):
                break
            start += step

        return chunks

    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a PDF and return all chunks.

        Returns:
            List of all chunks from the PDF
        """
        pages = self.extract_text_from_pdf(pdf_path)
        all_chunks = []

        for page_num, page_text in pages.items():
            chunks = self.chunk_text(
                page_text,
                source=pdf_path.name,
                source_path=str(pdf_path.absolute()),
                page=page_num,
            )
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {pdf_path.name}")
        return all_chunks


# ============================================================================
# Image Extractor (OpenCV-based)
# ============================================================================


class ImageExtractor:
    """Extract images from PDF using OpenCV edge detection."""

    def __init__(self, config: RAGAgentConfig):
        self.config = config
        self._cv2 = None
        self._fitz = None
        self._np = None
        self._init_libraries()

    def _init_libraries(self):
        """Initialize required libraries."""
        try:
            import cv2
            import numpy as np
            import fitz

            self._cv2 = cv2
            self._np = np
            self._fitz = fitz
            logger.info(
                "Image extraction libraries initialized (OpenCV, NumPy, PyMuPDF)"
            )
        except ImportError as e:
            logger.error(
                f"Missing library: {e}. Run: pip install opencv-python numpy pymupdf"
            )
            raise

    def _render_page_to_image(self, doc, page_num: int, dpi: int = 150):
        """Render a PDF page to an image array. Returns numpy ndarray."""
        page = doc[page_num]
        zoom = dpi / 72.0
        mat = self._fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = self._np.frombuffer(pix.samples, dtype=self._np.uint8).reshape(
            pix.height, pix.width, 3
        )
        return self._cv2.cvtColor(img, self._cv2.COLOR_RGB2BGR)

    def detect_borders(self, image, scale_for_detection=True):
        """
        Detect rectangular borders in an image using optimized edge detection.

        Args:
            image: OpenCV image (numpy array)
            scale_for_detection: Whether to scale down for faster detection

        Returns:
            List of bounding boxes [(x, y, w, h), ...] in original image coordinates
        """
        orig_h, orig_w = image.shape[:2]

        # Scale down for faster detection
        if scale_for_detection and self.config.scale_factor < 1.0:
            new_w = int(orig_w * self.config.scale_factor)
            new_h = int(orig_h * self.config.scale_factor)
            # Use INTER_NEAREST for fastest resize (acceptable for detection)
            image = self._cv2.resize(
                image, (new_w, new_h), interpolation=self._cv2.INTER_NEAREST
            )
            scale_ratio = 1.0 / self.config.scale_factor
        else:
            scale_ratio = 1.0

        # Convert to grayscale - use direct numpy for speed if possible
        if len(image.shape) == 3:
            # Fast grayscale using weighted sum (matches cv2.COLOR_BGR2GRAY)
            gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Use faster box blur instead of Gaussian (nearly identical results for edge detection)
        blurred = self._cv2.blur(gray, (5, 5))

        # Optimized edge detection with adaptive thresholds
        edges = self._cv2.Canny(blurred, 30, 100)

        # Dilate edges to close gaps - use pre-compiled kernel
        if _KERNEL_3x3 is not None:
            dilated = self._cv2.dilate(edges, _KERNEL_3x3, iterations=2)
        else:
            kernel = self._np.ones((3, 3), self._np.uint8)
            dilated = self._cv2.dilate(edges, kernel, iterations=2)

        # Find contours with simpler approximation for speed
        contours, _ = self._cv2.findContours(
            dilated, self._cv2.RETR_EXTERNAL, self._cv2.CHAIN_APPROX_SIMPLE
        )

        # Pre-calculate thresholds
        img_area = image.shape[0] * image.shape[1]
        min_area = img_area * self.config.min_diagram_area_pct
        min_dim = int(50 * self.config.scale_factor) if scale_for_detection else 50
        max_diagrams = self.config.max_diagrams_per_page

        # Filter contours by area and aspect ratio - with early exit
        bounding_boxes = []
        for contour in contours:
            # Quick area check before more expensive operations
            area = self._cv2.contourArea(contour)
            if area <= min_area:
                continue

            x, y, w, h = self._cv2.boundingRect(contour)

            # Filter out very thin or very wide boxes (likely page borders)
            if h == 0 or w < min_dim or h < min_dim:
                continue

            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 5:
                # Scale coordinates back to original image size
                if scale_ratio != 1.0:
                    x = int(x * scale_ratio)
                    y = int(y * scale_ratio)
                    w = int(w * scale_ratio)
                    h = int(h * scale_ratio)
                bounding_boxes.append((x, y, w, h))

                # Safety limit to prevent runaway processing
                if len(bounding_boxes) >= max_diagrams * 2:  # Extra buffer before dedup
                    break

        # Remove overlapping boxes (keep larger ones)
        bounding_boxes = self._remove_overlaps_fast(bounding_boxes)

        return bounding_boxes[:max_diagrams]

    def _remove_overlaps_fast(
        self, boxes: List[Tuple[int, int, int, int]], overlap_threshold: float = 0.8
    ) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping bounding boxes using vectorized operations where possible."""
        if not boxes or len(boxes) <= 1:
            return boxes

        # Convert to numpy for faster operations
        boxes_arr = self._np.array(boxes)
        areas = boxes_arr[:, 2] * boxes_arr[:, 3]  # w * h

        # Sort by area (largest first) - get indices
        sorted_indices = self._np.argsort(-areas)

        kept = []
        kept_boxes = []

        for idx in sorted_indices:
            box = boxes[idx]
            x1, y1, w1, h1 = box
            keep = True

            for kept_box in kept_boxes:
                x2, y2, w2, h2 = kept_box

                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)

                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    box_area = w1 * h1

                    if intersection / box_area > overlap_threshold:
                        keep = False
                        break

            if keep:
                kept.append(idx)
                kept_boxes.append(box)

        return kept_boxes

    def detect_text_region(self, region):
        """
        Detect if a region contains text using OpenCV morphological operations.
        Optimized version with early exits.

        Args:
            region: Image region to analyze

        Returns:
            True if text is detected, False otherwise
        """
        # Early dimension check
        h, w = region.shape[:2]
        if h < 10 or w < 10:
            return False

        # Convert to grayscale
        if len(region.shape) == 3:
            gray = self._cv2.cvtColor(region, self._cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        # Apply binary threshold - use simpler threshold for speed
        _, binary = self._cv2.threshold(gray, 127, 255, self._cv2.THRESH_BINARY_INV)

        # Use morphological operations to connect text components - use pre-compiled kernel
        if _KERNEL_MORPH is not None:
            morph = self._cv2.morphologyEx(binary, self._cv2.MORPH_CLOSE, _KERNEL_MORPH)
        else:
            kernel = self._cv2.getStructuringElement(self._cv2.MORPH_RECT, (3, 3))
            morph = self._cv2.morphologyEx(binary, self._cv2.MORPH_CLOSE, kernel)

        # Find contours - limit search with bounding rect
        contours, _ = self._cv2.findContours(
            morph, self._cv2.RETR_EXTERNAL, self._cv2.CHAIN_APPROX_SIMPLE
        )

        # Check for text-like characteristics with early exit
        text_contours = 0
        target_count = 3  # Threshold for text detection

        for contour in contours:
            x, y, cw, ch = self._cv2.boundingRect(contour)
            if ch == 0:
                continue
            aspect_ratio = cw / ch
            area = cw * ch

            # Text typically has certain aspect ratios and sizes
            if 0.1 < aspect_ratio < 10 and 20 < area < 5000:
                text_contours += 1
                # Early exit once we've found enough text-like contours
                if text_contours >= target_count:
                    return True

        return False

    def find_caption_region(self, box, full_image):
        """
        Find and extend the bounding box to include figure captions/titles
        by detecting text regions near the diagram.

        Optimized version with configurable enable/disable.

        Args:
            box: Original bounding box (x, y, w, h)
            full_image: Full page image

        Returns:
            Extended bounding box (x, y, w, h)
        """
        # Skip caption detection if disabled for speed
        if not self.config.enable_caption_detection:
            return box

        x, y, w, h = box
        img_h, img_w = full_image.shape[:2]

        # Define search regions for captions - reduced for speed
        search_height = 100  # Reduced from 150
        scan_step = 20  # Increased from 10 for fewer iterations

        extended_box = [x, y, w, h]

        # Check region below the diagram
        y_below = y + h
        if y_below + 40 < img_h:  # At least 40 pixels to search
            max_search = min(search_height, img_h - y_below)
            region_below = full_image[
                y_below : y_below + max_search, x : min(x + w, img_w)
            ]

            # Quick single-shot text detection instead of scanning
            if region_below.shape[0] >= 30 and self.detect_text_region(
                region_below[:40, :]
            ):
                extended_box[3] = h + 40  # Fixed caption extension

        # Check region above only if below didn't find caption
        if extended_box[3] == h and y > 40:
            region_above = full_image[max(0, y - 40) : y, x : min(x + w, img_w)]
            if region_above.shape[0] >= 30 and self.detect_text_region(region_above):
                extended_box[1] = max(0, y - 40)
                extended_box[3] = h + 40

        return tuple(extended_box)

    def _extract_figure_image(self, img, bbox: List[int], padding: int = 10):
        """Extract a figure region from the page image with padding. Returns numpy ndarray."""
        x, y, w, h = bbox

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        return img[y1:y2, x1:x2]

    def extract_images_from_pdf(
        self, pdf_path: Path, output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF document.

        Returns:
            List of extracted image metadata
        """
        cv2 = self._cv2

        # Create output directory
        images_dir = output_dir / "imgs"
        images_dir.mkdir(parents=True, exist_ok=True)

        doc = self._fitz.open(str(pdf_path))
        all_images = []

        try:
            # Process pages in smaller batches to reduce memory usage
            batch_size = 5  # Process 5 pages at a time
            total_pages = len(doc)

            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)

                for page_num in range(batch_start, batch_end):
                    try:
                        # Render page to image
                        page_img = self._render_page_to_image(
                            doc, page_num, self.config.detection_dpi
                        )

                        # Detect figures using optimized method
                        boxes = self.detect_borders(page_img, scale_for_detection=True)

                        if not boxes:
                            # Clean up page image immediately
                            del page_img
                            continue

                        # Re-render at higher DPI for extraction
                        high_res_img = None
                        if self.config.extraction_dpi != self.config.detection_dpi:
                            scale = (
                                self.config.extraction_dpi / self.config.detection_dpi
                            )
                            high_res_img = self._render_page_to_image(
                                doc, page_num, self.config.extraction_dpi
                            )
                        else:
                            scale = 1.0
                            high_res_img = page_img

                        # Extract and save each figure
                        for i, bbox in enumerate(boxes):
                            # Scale bbox for high-res image if needed
                            if scale != 1.0:
                                x, y, w, h = bbox
                                bbox = (
                                    int(x * scale),
                                    int(y * scale),
                                    int(w * scale),
                                    int(h * scale),
                                )

                            # Find caption region if enabled
                            if self.config.enable_caption_detection:
                                bbox = self.find_caption_region(bbox, high_res_img)

                            fig_img = self._extract_figure_image(
                                high_res_img, list(bbox)
                            )

                            # Generate filename
                            figure_id = f"page{page_num + 1}_fig{i}"
                            fig_filename = f"{pdf_path.stem}_{figure_id}.png"
                            fig_path = images_dir / fig_filename

                            # Save image with path length and encoding fixes
                            try:
                                if fig_img is None or fig_img.size == 0:
                                    logger.warning(
                                        f"Empty image extracted for page {page_num + 1}, figure {i}"
                                    )
                                    continue

                                # Ensure the directory exists
                                fig_path.parent.mkdir(parents=True, exist_ok=True)

                                # Validate image data
                                if (
                                    len(fig_img.shape) < 2
                                    or fig_img.shape[0] == 0
                                    or fig_img.shape[1] == 0
                                ):
                                    logger.warning(
                                        f"Invalid image dimensions {fig_img.shape} for page {page_num + 1}, figure {i}"
                                    )
                                    continue

                                # Check if image region is too small
                                if fig_img.shape[0] < 10 or fig_img.shape[1] < 10:
                                    logger.warning(
                                        f"Image region too small {fig_img.shape} for page {page_num + 1}, figure {i}"
                                    )
                                    continue

                                # Try to fix Windows path issues - use shorter names and absolute paths
                                try:
                                    # Use shorter filename to avoid Windows path length limits
                                    short_filename = (
                                        f"{pdf_path.stem}_{page_num + 1}_{i}.png"
                                    )
                                    short_path = images_dir / short_filename

                                    # Ensure path is not too long (Windows MAX_PATH = 260)
                                    if len(str(short_path)) > 250:
                                        # Use even shorter path
                                        short_filename = f"p{page_num + 1}_f{i}.png"
                                        short_path = images_dir / short_filename

                                    success = cv2.imwrite(str(short_path), fig_img)
                                    if success and short_path.exists():
                                        fig_path = short_path
                                        fig_filename = short_filename
                                    else:
                                        # Try using the original path
                                        success = cv2.imwrite(str(fig_path), fig_img)
                                        if not success:
                                            logger.warning(
                                                f"OpenCV failed to save image with both short and long paths: {fig_filename}"
                                            )
                                            continue
                                except Exception as e:
                                    logger.error(
                                        f"Error with path handling for {fig_filename}: {e}"
                                    )
                                    continue

                                if not fig_path.exists():
                                    logger.warning(
                                        f"Image file not created despite success flag: {fig_path}"
                                    )
                                    continue

                                # Verify file size
                                file_size = fig_path.stat().st_size
                                if file_size == 0:
                                    logger.warning(
                                        f"Empty image file created: {fig_path}"
                                    )
                                    fig_path.unlink()  # Delete empty file
                                    continue

                            except Exception as e:
                                logger.error(f"Error saving image {fig_filename}: {e}")
                                continue

                            # Store metadata
                            fig_metadata = {
                                "bbox": list(bbox),
                                "area": bbox[2] * bbox[3],
                                "aspect_ratio": bbox[2] / bbox[3] if bbox[3] > 0 else 0,
                                "page": page_num + 1,
                                "figure_id": figure_id,
                                "file_path": str(fig_path),
                                "filename": fig_filename,
                                "source_pdf": pdf_path.name,
                                "source_path": str(pdf_path.absolute()),
                                "extracted_at": datetime.now().isoformat(),
                            }

                            all_images.append(fig_metadata)

                            # Clean up figure image immediately
                            del fig_img

                        # Clean up page images
                        if high_res_img is not page_img:
                            del high_res_img
                        del page_img

                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {e}")
                        continue

                # Force garbage collection after each batch
                import gc

                gc.collect()

        finally:
            doc.close()

        logger.info(f"Extracted {len(all_images)} images from {pdf_path.name}")
        return all_images


# ============================================================================
# CLIP Image Embedder
# ============================================================================

# Global CLIP model cache to avoid reloading
_CLIP_CACHE = {
    "model": None,
    "preprocess": None,
    "tokenizer": None,
    "device": None,
    "loaded": False,
}


class CLIPImageEmbedder:
    """Generate CLIP embeddings for images and store in ChromaDB."""

    def __init__(self, config: RAGAgentConfig):
        self.config = config
        self.collection = None
        self._clip_ready = False
        self._init_chromadb()  # ChromaDB is fast, init immediately
        # CLIP is lazy loaded on first use

    @property
    def model(self):
        self._ensure_clip_loaded()
        return _CLIP_CACHE["model"]

    @property
    def preprocess(self):
        self._ensure_clip_loaded()
        return _CLIP_CACHE["preprocess"]

    @property
    def tokenizer(self):
        self._ensure_clip_loaded()
        return _CLIP_CACHE["tokenizer"]

    @property
    def device(self):
        self._ensure_clip_loaded()
        return _CLIP_CACHE["device"] or "cpu"

    def _ensure_clip_loaded(self):
        """Lazy load CLIP model on first use."""
        if _CLIP_CACHE["loaded"]:
            return

        if not self.config.enable_clip_embeddings:
            return

        try:
            import open_clip
            import torch

            logger.info("Loading CLIP model (first use)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Workaround for meta tensor issue in newer PyTorch versions
            # Set default device to CPU before model creation
            original_device = None
            try:
                # Try to set default tensor type to avoid meta device issues
                if hasattr(torch, "set_default_device"):
                    original_device = (
                        torch.get_default_device()
                        if hasattr(torch, "get_default_device")
                        else None
                    )
                    torch.set_default_device("cpu")
            except Exception:
                pass

            try:
                # Method 1: Try with explicit CPU device parameter
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.config.clip_model,
                    pretrained="openai",
                    device=torch.device("cpu"),  # Explicit torch.device object
                    precision="fp32",
                    jit=False,  # Prevent JIT tracing meta tensor issues
                )
                logger.info("CLIP model loaded with device parameter")

                if next(model.parameters()).is_meta:
                    raise RuntimeError("Method 1 returned meta model")

            except (TypeError, RuntimeError) as e:
                logger.info(f"Method 1 failed ({e}), trying method 2...")

                try:
                    # Method 2: Create without pretrained, then load weights separately
                    model, _, preprocess = open_clip.create_model_and_transforms(
                        self.config.clip_model,
                        pretrained=None,
                        jit=False,  # Prevent JIT tracing meta tensor issues
                    )

                    # Check if on meta device
                    first_param = next(model.parameters(), None)
                    if first_param is not None and first_param.is_meta:
                        logger.info("Model on meta device, using to_empty()")
                        model = model.to_empty(device="cpu")

                    # Load pretrained weights
                    pretrained_cfg = open_clip.get_pretrained_cfg(
                        self.config.clip_model, "openai"
                    )
                    if pretrained_cfg and "url" in pretrained_cfg:
                        state_dict = torch.hub.load_state_dict_from_url(
                            pretrained_cfg["url"], map_location="cpu", check_hash=True
                        )
                        model.load_state_dict(state_dict, strict=False)
                    logger.info("CLIP model loaded via separate weight loading")

                except Exception as e2:
                    logger.info(f"Method 2 failed ({e2}), trying method 3...")

                    # Method 3: Use CLIP from transformers if available
                    try:
                        from transformers import CLIPProcessor, CLIPModel

                        model = CLIPModel.from_pretrained(
                            _LOCAL_CLIP_MODEL
                        )
                        preprocess = CLIPProcessor.from_pretrained(
                            _LOCAL_CLIP_MODEL
                        )
                        logger.info("Loaded CLIP from transformers library")
                    except ImportError:
                        raise RuntimeError("All CLIP loading methods failed")

            # Restore original device setting
            if original_device is not None:
                try:
                    torch.set_default_device(original_device)
                except Exception:
                    pass

            first_param = next(model.parameters(), None)

            if device != "cpu":
                if first_param is not None and first_param.is_meta:
                    logger.info("Model is meta — using to_empty()")
                    model = model.to_empty(device=device)
                else:
                    model = model.to(device)

            if "open_clip" in type(model).__module__:
                tokenizer = open_clip.get_tokenizer(self.config.clip_model)
            else:
                tokenizer = preprocess.tokenizer

            model.eval()

            # Cache globally
            _CLIP_CACHE["model"] = model
            _CLIP_CACHE["preprocess"] = preprocess
            _CLIP_CACHE["tokenizer"] = tokenizer
            _CLIP_CACHE["device"] = device
            _CLIP_CACHE["loaded"] = True

            logger.info(
                f"CLIP model loaded ({self.config.clip_model}, device: {device})"
            )
        except ImportError:
            logger.warning("open_clip not installed. CLIP embeddings disabled.")
            self.config.enable_clip_embeddings = False
            _CLIP_CACHE["loaded"] = True  # Mark as "loaded" to avoid retrying
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.config.enable_clip_embeddings = False
            _CLIP_CACHE["loaded"] = True

    def _init_chromadb(self):
        """Initialize ChromaDB for image embeddings."""
        if not self.config.enable_clip_embeddings:
            return

        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.config.image_db_path)
            self.collection = client.get_or_create_collection(
                name="infineon_images", metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"ChromaDB initialized ({self.collection.count()} existing images)"
            )
        except ImportError:
            logger.warning("chromadb not installed. Image storage disabled.")
            self.config.enable_clip_embeddings = False
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.config.enable_clip_embeddings = False

    def embed_image(self, image_path: str) -> Optional[List[float]]:
        """Generate CLIP embedding for a single image."""
        if not self.config.enable_clip_embeddings:
            return None

        self._ensure_clip_loaded()
        if _CLIP_CACHE["model"] is None:
            return None

        try:
            import torch
            from PIL import Image

            img = Image.open(image_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(img_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                return features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.warning(f"Failed to embed image {image_path}: {e}")
            return None

    def embed_images_batch(
        self, image_paths: List[str], batch_size: int = 16
    ) -> List[Optional[List[float]]]:
        """
        Generate CLIP embeddings for multiple images in batches (much faster than one-by-one).

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once (default: 16)

        Returns:
            List of embeddings (None for failed images)
        """
        if not self.config.enable_clip_embeddings or not image_paths:
            return [None] * len(image_paths)

        self._ensure_clip_loaded()
        if _CLIP_CACHE["model"] is None:
            return [None] * len(image_paths)

        import torch
        from PIL import Image

        all_embeddings = []

        # Process in batches
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            # Load and preprocess images
            batch_tensors = []
            valid_indices = []

            for i, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = self.preprocess(img)
                    batch_tensors.append(img_tensor)
                    valid_indices.append(batch_start + i)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")

            if not batch_tensors:
                all_embeddings.extend([None] * len(batch_paths))
                continue

            # Stack into batch tensor and encode
            try:
                batch_tensor = torch.stack(batch_tensors)

                batch_tensor = batch_tensor.to(self.device)

                with torch.no_grad():
                    features = self.model.encode_image(batch_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    batch_embeddings = features.cpu().numpy().tolist()

                # Map embeddings back to original indices
                embedding_idx = 0
                for i, path in enumerate(batch_paths):
                    if (batch_start + i) in valid_indices:
                        all_embeddings.append(batch_embeddings[embedding_idx])
                        embedding_idx += 1
                    else:
                        all_embeddings.append(None)

            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")
                all_embeddings.extend([None] * len(batch_paths))

        return all_embeddings

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate CLIP embedding for text query."""
        if not self.config.enable_clip_embeddings:
            return None

        self._ensure_clip_loaded()
        if _CLIP_CACHE["model"] is None:
            return None

        try:
            import torch

            with torch.no_grad():
                # Tokenize text
                text_tokens = self.tokenizer([text])
                text_tokens = text_tokens.to(self.device)

                features = self.model.encode_text(text_tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                return features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.warning(f"Failed to embed text: {e}")
            return None

    def index_image(self, image_path: str, metadata: Dict[str, Any]) -> bool:
        """Index an image with its CLIP embedding."""
        if self.collection is None:
            return False

        # Generate embedding
        embedding = self.embed_image(image_path)
        if embedding is None:
            return False

        # Create unique ID
        img_id = hashlib.md5(image_path.encode()).hexdigest()

        # Check if already indexed
        existing = self.collection.get(ids=[img_id])
        if existing["ids"]:
            return True  # Already indexed

        # Clean metadata (ChromaDB only accepts str, int, float, bool)
        clean_meta = {
            "file_path": str(image_path),
            "filename": metadata.get("filename", ""),
            "source_pdf": metadata.get("source_pdf", ""),
            "source_path": metadata.get("source_path", ""),
            "page": metadata.get("page", 0),
            "figure_id": metadata.get("figure_id", ""),
        }

        # Add GPT analysis if available
        if "gpt_analysis" in metadata:
            clean_meta["gpt_description"] = str(
                metadata["gpt_analysis"].get("description", "")
            )[:1000]

        self.collection.add(
            ids=[img_id], embeddings=[embedding], metadatas=[clean_meta]
        )

        return True

    def index_images_batch(
        self, images: List[Dict[str, Any]], batch_size: int = 16
    ) -> int:
        """
        Index multiple images with their CLIP embeddings in batches (optimized for ingestion).

        Args:
            images: List of image metadata dicts (must have 'file_path' key)
            batch_size: Number of images to embed at once

        Returns:
            Number of successfully indexed images
        """
        if self.collection is None or not images:
            return 0

        # Extract paths and create IDs
        image_paths = [img.get("file_path", "") for img in images]
        img_ids = [hashlib.md5(p.encode()).hexdigest() for p in image_paths]

        # Batch check which images are already indexed
        try:
            existing = self.collection.get(ids=img_ids)
            existing_ids = set(existing["ids"]) if existing["ids"] else set()
        except Exception:
            existing_ids = set()

        # Filter to only new images
        new_images = []
        new_paths = []
        new_ids = []

        for img, path, img_id in zip(images, image_paths, img_ids):
            if img_id not in existing_ids and path:
                new_images.append(img)
                new_paths.append(path)
                new_ids.append(img_id)

        if not new_images:
            logger.info(f"All {len(images)} images already indexed")
            return 0

        logger.info(
            f"Indexing {len(new_images)} new images (skipping {len(existing_ids)} existing)"
        )

        # Batch embed all new images
        embeddings = self.embed_images_batch(new_paths, batch_size=batch_size)

        # Prepare batch insert data
        valid_ids = []
        valid_embeddings = []
        valid_metadatas = []

        for img, img_id, embedding in zip(new_images, new_ids, embeddings):
            if embedding is None:
                continue

            # Clean metadata (ChromaDB only accepts str, int, float, bool)
            clean_meta = {
                "file_path": str(img.get("file_path", "")),
                "filename": img.get("filename", ""),
                "source_pdf": img.get("source_pdf", ""),
                "source_path": img.get("source_path", ""),
                "page": img.get("page", 0),
                "figure_id": img.get("figure_id", ""),
            }

            # Add GPT analysis if available
            if "gpt_analysis" in img:
                clean_meta["gpt_description"] = str(
                    img["gpt_analysis"].get("description", "")
                )[:1000]

            valid_ids.append(img_id)
            valid_embeddings.append(embedding)
            valid_metadatas.append(clean_meta)

        # Batch insert into ChromaDB
        if valid_ids:
            try:
                self.collection.add(
                    ids=valid_ids,
                    embeddings=valid_embeddings,
                    metadatas=valid_metadatas,
                )
                logger.info(f"Batch indexed {len(valid_ids)} images to ChromaDB")
            except Exception as e:
                logger.error(f"Batch ChromaDB insert failed: {e}")
                return 0

        return len(valid_ids)

    def search_images(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[MetadataFilter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for images similar to a text query with optional metadata filtering.

        Args:
            query: Search query text
            top_k: Number of results to return
            metadata_filter: Optional MetadataFilter for filtering results

        Returns:
            List of matching images sorted by similarity
        """
        if self.collection is None or self.collection.count() == 0:
            return []

        # Limit top_k for performance
        top_k = min(top_k, 20)  # Cap at 20 images for memory efficiency

        embedding = self.embed_text(query)
        if embedding is None:
            return []

        # Build ChromaDB where clause from filter
        where_clause = None
        if metadata_filter:
            where_clause = metadata_filter.to_chromadb_where()

        # For folder filtering, fetch more results and filter in Python
        has_folder_filter = (
            metadata_filter
            and metadata_filter.custom
            and "_target_folders" in metadata_filter.custom
        )
        fetch_count = min(top_k * 3, 50) if has_folder_filter else top_k

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(fetch_count, self.collection.count()),
                include=["metadatas", "distances"],
                where=where_clause,
            )
        except Exception as e:
            # If filtering fails (e.g., no matching documents), try without filter
            logger.warning(f"Filtered image search failed: {e}. Trying without filter.")
            try:
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=min(fetch_count, self.collection.count()),
                    include=["metadatas", "distances"],
                )
            except Exception as e2:
                logger.error(f"ChromaDB fallback query failed: {e2}")
                return []

        formatted = []
        for i in range(len(results["ids"][0])):
            try:
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                similarity = 1 - dist / 2  # Convert cosine distance to similarity

                image_data = {
                    "file_path": meta.get("file_path", ""),
                    "filename": meta.get("filename", ""),
                    "source_pdf": meta.get("source_pdf", ""),
                    "source_path": meta.get("source_path", ""),
                    "page": meta.get("page", 0),
                    "similarity": similarity,
                    "gpt_description": meta.get("gpt_description", ""),
                }

                # Apply folder filtering in Python if needed
                if metadata_filter and metadata_filter.matches(meta):
                    formatted.append(image_data)
                elif not metadata_filter:
                    formatted.append(image_data)

            except (IndexError, KeyError) as e:
                logger.warning(f"Error processing image result {i}: {e}")
                continue

        # Limit to requested top_k after filtering
        return formatted[:top_k]

    def get_available_sources(self) -> List[str]:
        """
        Get list of unique source documents in the image collection.

        Returns:
            List of unique source PDF names
        """
        if self.collection is None or self.collection.count() == 0:
            return []

        try:
            # Get all metadata
            all_data = self.collection.get(include=["metadatas"])
            sources = set()
            for meta in all_data["metadatas"]:
                source = meta.get("source_pdf", "")
                if source:
                    sources.add(source)
            return sorted(list(sources))
        except Exception as e:
            logger.warning(f"Failed to get image sources: {e}")
            return []


# ============================================================================
# Vector Store Manager
# ============================================================================

# Global sentence transformer cache
_SENTENCE_TRANSFORMER_CACHE = {"model": None, "loaded": False}

# Resolve the local model path relative to this file:
# rag_agent/infineon_rag_agent.py → ../../.cache/huggingface/hub/...
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_EMBEDDING_MODEL = os.path.join(
    _WORKSPACE_ROOT,
    "models", ".cache", "huggingface", "hub",
    "models--sentence-transformers--all-MiniLM-L6-v2",
    "snapshots", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
)

_LOCAL_CLIP_MODEL = os.path.join(
    _WORKSPACE_ROOT,
    "models", ".cache", "huggingface", "hub",
    "models--openai--clip-vit-base-patch32",
    "snapshots", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
)


class VectorStoreManager:
    """Manage text chunk embeddings using FAISS."""

    def __init__(self, config: RAGAgentConfig):
        self.config = config
        self.index = None
        self.texts = []
        self.metadata = []
        self._embed_model = None
        # Cached list of metadata indices whose document_type == "AURIX_projects".
        # Built lazily on first access; reset to None whenever add_chunks() is called.
        self._project_chunks_cache: Optional[List[int]] = None

    def _ensure_embeddings_loaded(self):
        """Lazy load sentence transformer on first use."""
        if _SENTENCE_TRANSFORMER_CACHE["loaded"]:
            self._embed_model = _SENTENCE_TRANSFORMER_CACHE["model"]
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence transformer (first use)...")
            _SENTENCE_TRANSFORMER_CACHE["model"] = SentenceTransformer(
                _LOCAL_EMBEDDING_MODEL
            )
            _SENTENCE_TRANSFORMER_CACHE["loaded"] = True
            self._embed_model = _SENTENCE_TRANSFORMER_CACHE["model"]
            logger.info("Sentence transformer loaded for text embeddings")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Text search will be limited."
            )
            _SENTENCE_TRANSFORMER_CACHE["loaded"] = True  # Mark to avoid retry

    def _embed_texts(self, texts, show_progress_bar=False):
        """Embed texts using the model."""
        self._ensure_embeddings_loaded()
        if self._embed_model is None:
            return None
        return self._embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress_bar)

    def search_source_direct(
        self,
        query: str,
        source_pattern: str,
        top_k: int = 5,
        include_table_of_contents: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Directly search within chunks whose 'source' contains *source_pattern*.

        Unlike the main FAISS search this method does NOT rely on global
        approximate nearest-neighbor search — it explicitly finds all chunks
        matching the source filter and computes cosine similarity against them.
        This guarantees that relevant chunks from a sparse source (e.g. a board
        kit manual that is only 43 of 41 000+ total chunks) are found even
        when they would not appear in the global top-k.

        Args:
            query: User query string.
            source_pattern: Case-insensitive substring to match in chunk 'source'.
            top_k: Maximum chunks to return.
            include_table_of_contents: Whether to include ToC pages.

        Returns:
            List of matching chunk dicts sorted by similarity (highest first).
        """
        if not self.texts or self.index is None:
            return []

        source_lower = source_pattern.lower()

        # 1. Collect indices of all chunks from the target source
        source_indices = [
            i
            for i, meta in enumerate(self.metadata)
            if source_lower in (meta.get("source", "") or meta.get("source_pdf", "")).lower()
        ]
        if not source_indices:
            return []

        # 2. Embed the query
        query_embedding = self._embed_texts([query])
        if query_embedding is None:
            return []

        try:
            import numpy as np
            from numpy.linalg import norm

            q_vec = query_embedding[0].astype("float32")
            q_norm = norm(q_vec)

            # 3. Compute similarities for source-filtered chunks.
            #    Strategy A: use FAISS reconstruct() — fast, zero extra inference.
            #    Strategy B: re-embed the texts — fallback if reconstruct fails.
            similarities: List[Tuple[float, int]] = []
            use_reconstruct = hasattr(self.index, "reconstruct")

            if use_reconstruct:
                for i in source_indices:
                    if not include_table_of_contents and self._is_table_of_contents(
                        self.texts[i]
                    ):
                        continue
                    try:
                        stored = self.index.reconstruct(i).astype("float32")
                    except Exception:
                        use_reconstruct = False  # index doesn't support reconstruction
                        break
                    s_norm = norm(stored)
                    sim = float(np.dot(q_vec, stored) / (q_norm * s_norm + 1e-8))
                    similarities.append((sim, i))

            if not use_reconstruct or not similarities:
                # Fallback: re-embed the source texts and compute similarity
                valid_indices = [
                    i
                    for i in source_indices
                    if include_table_of_contents
                    or not self._is_table_of_contents(self.texts[i])
                ]
                if valid_indices:
                    texts_to_embed = [self.texts[i] for i in valid_indices]
                    embeddings = self._embed_texts(texts_to_embed)
                    if embeddings is not None:
                        for j, i in enumerate(valid_indices):
                            s_norm = norm(embeddings[j])
                            sim = float(
                                np.dot(q_vec, embeddings[j]) / (q_norm * s_norm + 1e-8)
                            )
                            similarities.append((sim, i))

            similarities.sort(reverse=True)

            results = []
            for rank, (sim, idx) in enumerate(similarities[:top_k], 1):
                meta = dict(self.metadata[idx])
                meta["score"] = sim
                meta["rank"] = rank
                meta["text"] = self.texts[idx]
                results.append(meta)

            return results

        except Exception as e:
            logger.warning(f"search_source_direct failed: {e}")
            return []

    def search_batch(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        metadata_filter: Optional["MetadataFilter"] = None,
        include_table_of_contents: bool = False,
        detected_architectures: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed *all* queries in a single batch encode call, then run one batched
        FAISS search.  Returns deduplicated results sorted by score.

        Use this instead of calling search() in a loop whenever you have a list
        of supplementary/strategy queries – it avoids N separate model.encode()
        calls and N separate FAISS round-trips.
        """
        if not queries or not self.texts or self.index is None:
            return []

        # Deduplicate while preserving order
        seen_q: set = set()
        unique_queries: List[str] = []
        for q in queries:
            if q not in seen_q:
                seen_q.add(q)
                unique_queries.append(q)

        # --- single batch encode ---
        query_embeddings = self._embed_texts(unique_queries)
        if query_embeddings is None:
            return []

        try:
            import numpy as np

            intent_classifier = get_query_intent_classifier()
            search_k = min(top_k_per_query * 3, len(self.texts), 200)

            # Batched FAISS search: one call for all queries
            # D, I shapes: (n_queries, search_k)
            all_embeddings = np.array(query_embeddings).astype("float32")
            D, I = self.index.search(all_embeddings, search_k)

            seen_indices: set = set()
            results: List[Dict[str, Any]] = []

            for q_idx, (dists, indices) in enumerate(zip(D, I)):
                query = unique_queries[q_idx]
                query_intent = intent_classifier.classify(query)

                for dist, idx in zip(dists, indices):
                    if idx < 0 or idx >= len(self.metadata):
                        continue
                    if idx in seen_indices:
                        continue
                    meta = self.metadata[idx]
                    if metadata_filter and not metadata_filter.matches(meta):
                        continue
                    if not include_table_of_contents and self._is_table_of_contents(
                        self.texts[idx]
                    ):
                        continue

                    seen_indices.add(idx)
                    result = dict(meta)
                    base_score = 1 / (1 + dist)

                    filename = meta.get("source", "") or meta.get("source_pdf", "")
                    doc_boost = intent_classifier.get_boost_for_document(
                        query_intent, filename, meta
                    )

                    if detected_architectures:
                        boost = self._calculate_architecture_boost(
                            filename, detected_architectures, query, meta
                        )
                        result["score"] = base_score * (1 + boost) * (1 + doc_boost)
                        result["architecture_boost"] = boost
                    else:
                        result["score"] = base_score * (1 + doc_boost)

                    result["text"] = self.texts[idx]
                    results.append(result)

            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Error in batch vector search: {e}")
            return []

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Add chunks to the vector store."""
        if not chunks:
            return 0

        # Invalidate project-chunk cache whenever new data is added
        self._project_chunks_cache = None

        # Extract texts
        texts = [c["text"] for c in chunks]

        # Generate embeddings
        embeddings = self._embed_texts(texts)
        if embeddings is None:
            # Store without embeddings
            self.texts.extend(texts)
            self.metadata.extend(chunks)
            return len(texts)

        # Initialize or update FAISS index
        try:
            import faiss
            import numpy as np

            dim = embeddings.shape[1]

            if self.index is None:
                self.index = faiss.IndexFlatL2(dim)

            self.index.add(np.array(embeddings).astype("float32"))
            self.texts.extend(texts)
            self.metadata.extend(chunks)

            return len(texts)

        except ImportError:
            logger.warning("faiss-cpu not installed. Using simple list storage.")
            self.texts.extend(texts)
            self.metadata.extend(chunks)
            return len(texts)

    def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[MetadataFilter] = None,
        include_table_of_contents: bool = False,
        detected_architectures: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks with optional metadata filtering.

        Args:
            query: Search query text
            top_k: Number of results to return
            metadata_filter: Optional MetadataFilter to filter results
            include_table_of_contents: Whether to include table of contents in results

        Returns:
            List of matching chunks sorted by relevance
        """
        if not self.texts:
            return []

        # Limit top_k to reasonable bounds for performance
        top_k = min(top_k, 50)  # Cap at 50 for memory efficiency

        # Initialize query intent classifier and classify intent
        intent_classifier = get_query_intent_classifier()
        query_intent = intent_classifier.classify(query)

        # =====================================================================
        # PROJECT-BASED RETRIEVAL: Forcefully retrieve project chunks when query
        # mentions a specific project (e.g., "GTM TOM 3 Phase Inverter PWM 2")
        # This ensures project-specific information is found even if semantic
        # similarity is low.
        # =====================================================================
        injected_project_chunks = []
        if detected_architectures:
            injected_project_chunks = self._find_matching_project_chunks(
                query, detected_architectures
            )

        # Embed query
        query_embedding = self._embed_texts([query])
        if query_embedding is None:
            return []

        try:
            import numpy as np

            # For filtered search or architecture boosting, use larger candidate pool
            # to ensure relevant documents aren't missed
            if detected_architectures:
                # Larger pool when using architecture boosting for re-ranking
                search_multiplier = 20  # 20× gives good reranking without memory pressure
            elif metadata_filter:
                search_multiplier = 5
            else:
                search_multiplier = 2
            search_k = min(
                top_k * search_multiplier, len(self.texts), 1000
            )  # Increase cap

            if self.index is not None:
                # FAISS search - retrieve candidates efficiently
                D, I = self.index.search(
                    np.array(query_embedding).astype("float32"), search_k
                )

                results = []
                for dist, idx in zip(D[0], I[0]):
                    if idx < len(self.metadata):
                        meta = self.metadata[idx]

                        # Apply metadata filter if provided
                        if metadata_filter and not metadata_filter.matches(meta):
                            continue

                        # Filter out table of contents unless explicitly requested
                        if (
                            not include_table_of_contents
                            and self._is_table_of_contents(self.texts[idx])
                        ):
                            continue

                        result = dict(meta)
                        base_score = 1 / (1 + dist)  # Convert distance to similarity

                        # Apply document-type intent boost
                        filename = meta.get("source", "") or meta.get("source_pdf", "")
                        doc_boost = intent_classifier.get_boost_for_document(
                            query_intent, filename, meta
                        )
                        result["doc_boost"] = doc_boost
                        # Also expose detected doc_type for transparency
                        try:
                            detected_doc_type = intent_classifier.detect_document_type(
                                filename, meta
                            )
                            result["doc_type"] = detected_doc_type
                        except Exception:
                            pass

                        # Apply architecture-based relevance boost
                        if detected_architectures:
                            boost = self._calculate_architecture_boost(
                                meta.get("source", "") or meta.get("source_pdf", ""),
                                detected_architectures,
                                query,  # Pass query for context-aware boosting
                                meta,  # Pass full metadata for project-specific boosting
                            )
                            result["score"] = base_score * (1 + boost) * (1 + doc_boost)
                            result["architecture_boost"] = boost
                        else:
                            result["score"] = base_score * (1 + doc_boost)

                        result["text"] = self.texts[idx]  # Add text for context
                        results.append(result)

                # Re-sort by boosted score if architecture boost was applied
                if detected_architectures:
                    results.sort(key=lambda x: x["score"], reverse=True)

                # =====================================================================
                # MERGE INJECTED PROJECT CHUNKS: Add project chunks that weren't found
                # by vector search but match by metadata
                # =====================================================================
                if injected_project_chunks:
                    # Get indices already in results to avoid duplicates
                    existing_indices = {r.get("chunk_index") for r in results}
                    for chunk in injected_project_chunks:
                        if chunk.get("chunk_index") not in existing_indices:
                            results.append(chunk)
                    # Re-sort after adding injected chunks
                    results.sort(key=lambda x: x.get("score", 0), reverse=True)

                # Limit to top_k after boost-based re-ranking
                results = results[:top_k]

                # Add ranks
                for i, result in enumerate(results):
                    result["rank"] = i + 1

                return results
            else:
                # Fallback: Use batched similarity computation for memory efficiency
                from numpy.linalg import norm

                # Process embeddings in smaller batches to save memory
                batch_size = 100
                similarities = []

                for start_idx in range(0, len(self.texts), batch_size):
                    end_idx = min(start_idx + batch_size, len(self.texts))
                    batch_texts = self.texts[start_idx:end_idx]
                    batch_embeddings = self._embed_texts(batch_texts)

                    for i, emb in enumerate(batch_embeddings):
                        actual_idx = start_idx + i
                        meta = self.metadata[actual_idx]

                        # Apply metadata filter first if provided
                        if metadata_filter and not metadata_filter.matches(meta):
                            continue

                        # Filter out table of contents unless explicitly requested
                        if (
                            not include_table_of_contents
                            and self._is_table_of_contents(self.texts[actual_idx])
                        ):
                            continue

                        sim = np.dot(query_embedding[0], emb) / (
                            norm(query_embedding[0]) * norm(emb) + 1e-8
                        )

                        # Apply document-type intent boost
                        filename = meta.get("source", "") or meta.get("source_pdf", "")
                        doc_boost = intent_classifier.get_boost_for_document(
                            query_intent, filename, meta
                        )
                        sim = sim * (1 + doc_boost)

                        # Apply architecture boost in fallback path too
                        if detected_architectures:
                            boost = self._calculate_architecture_boost(
                                meta.get("source", "") or meta.get("source_pdf", ""),
                                detected_architectures,
                                query,
                                meta,
                            )
                            sim = sim * (1 + boost)

                        similarities.append((sim, actual_idx))

                similarities.sort(reverse=True)

                results = []
                for rank, (sim, idx) in enumerate(similarities[:top_k], 1):
                    result = dict(self.metadata[idx])
                    result["score"] = sim
                    result["rank"] = rank
                    result["text"] = self.texts[idx]  # Add text for context
                    results.append(result)

                return results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _find_matching_project_chunks(
        self, query: str, detected_architectures: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Find project chunks that match the query by metadata (not vector similarity).

        This ensures that when a user queries for a specific project (e.g., "GTM TOM PWM 2"),
        we forcefully retrieve those project chunks even if they have low semantic similarity.

        Args:
            query: User's search query
            detected_architectures: Detected architectures from query

        Returns:
            List of matching project chunks with high boost scores
        """
        matching_chunks = []
        query_lower = query.lower()

        # Extract project keywords from query
        project_keywords = self._extract_project_keywords_from_query(query_lower)
        if not project_keywords or len(project_keywords) < 3:
            return []  # Not enough keywords to identify a specific project

        # Extract suffix number from query (e.g., "PWM 2" -> "2")
        query_suffix_match = re.search(
            r"(?:pwm|adc|can|spi|uart|dma|timer|encoder|demo|shell)[_\s]?(\d)(?:\s|$|[?\.])",
            query_lower,
        )
        query_suffix = query_suffix_match.group(1) if query_suffix_match else None

        # Build architecture matching patterns
        # Map family names to regex patterns that match specific MCU names in project names
        arch_patterns = []
        for arch in detected_architectures:
            arch_lower = arch.lower()
            # Add exact match pattern
            arch_patterns.append(re.escape(arch_lower))

            # Map family names to specific MCU patterns
            if "tc4xx" in arch_lower or "aurix tc4" in arch_lower:
                # AURIX TC4xx family -> match tc4d7, tc4dx, tc4x, tc4 patterns
                arch_patterns.extend(["tc4d7", "tc4dx", "tc4d", "tc4x"])
            elif "tc3xx" in arch_lower or "aurix tc3" in arch_lower:
                # AURIX TC3xx family -> match tc38x, tc387, tc375, etc.
                arch_patterns.extend(
                    ["tc38", "tc37", "tc36", "tc35", "tc39", "tc33", "tc32"]
                )
            elif re.match(r"tc3\d", arch_lower):
                # Specific TC3xx subfamily (e.g., TC38x) -> match specific variants
                prefix = arch_lower[:4]  # e.g., "tc38"
                arch_patterns.append(prefix)
            elif re.match(r"tc\d{3,4}", arch_lower):
                # Specific MCU (e.g., TC387) -> use as-is
                arch_patterns.append(arch_lower)

        # Build (and cache) the list of project-chunk indices so subsequent
        # queries don't pay the cost of a full O(n) metadata scan.
        if self._project_chunks_cache is None:
            self._project_chunks_cache = [
                i
                for i, m in enumerate(self.metadata)
                if m.get("document_type") == "AURIX_projects"
            ]

        # Scan only project chunks (typically a few hundred vs 41K total)
        for idx in self._project_chunks_cache:
            meta = self.metadata[idx]

            project_name = meta.get("project_name", "").lower()
            source = meta.get("source", "").lower()

            if not project_name:
                continue

            # Check architecture match using expanded patterns
            combined_text = project_name + " " + source
            arch_match = any(pattern in combined_text for pattern in arch_patterns)
            if not arch_match:
                continue

            # Count keyword matches
            matches = sum(
                1 for kw in project_keywords if kw in project_name or kw in source
            )
            match_ratio = matches / len(project_keywords) if project_keywords else 0

            # Check suffix match
            project_suffix_match = re.search(r"[_](\d)(?:_readme)?$", project_name)
            project_suffix = (
                project_suffix_match.group(1) if project_suffix_match else None
            )

            # Only include if:
            # 1. High keyword match ratio (>= 60%)
            # 2. Suffix matches (if both have suffixes)
            if match_ratio >= 0.6:
                # If both have suffixes, they must match
                if query_suffix and project_suffix:
                    if query_suffix != project_suffix:
                        continue  # Skip - suffix mismatch

                # Create result with high boost score
                result = dict(meta)
                result["text"] = self.texts[idx]
                result["chunk_index"] = idx

                # Calculate boost score - high for exact matches
                base_score = 0.8  # High base score
                boost = (
                    12.0 if match_ratio >= 0.7 else 8.0
                )  # Very high boost for project match

                # Additional boost for chunks containing pin mapping information
                # This ensures pin tables are prioritized over general project descriptions
                text_content = result["text"]
                has_pin_mapping = False

                # Look for pin patterns like P20.8, P10.3, PHASE_U_HS, etc.
                pin_patterns = [
                    r"P\d+\.\d+",  # Port pins like P20.8, P10.3
                    r"PHASE_[UVW]_[HL]S",  # Phase signals
                    r"TOUT\d+",  # Timer outputs
                    r"Pin\s+Mapping",  # Pin mapping table headers
                ]
                for pattern in pin_patterns:
                    if re.search(pattern, text_content, re.IGNORECASE):
                        has_pin_mapping = True
                        break

                if has_pin_mapping:
                    boost += 3.0  # Extra boost for pin mapping content

                result["score"] = base_score * (1 + boost)
                result["architecture_boost"] = boost
                result["injected_from_metadata"] = True  # Mark as injected
                result["has_pin_mapping"] = has_pin_mapping  # Mark for debugging

                matching_chunks.append(result)

        # Sort by score and return top chunks (limit to prevent too many injections)
        matching_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        return matching_chunks[:20]  # Return up to 20 matching project chunks

    def _calculate_architecture_boost(
        self,
        filename: str,
        detected_architectures: List[str],
        query: str = "",
        metadata: Dict[str, Any] = None,
    ) -> float:
        """
        Calculate relevance boost based on architecture match in filename.

        Args:
            filename: Source document filename
            detected_architectures: List of architectures detected from user query
            query: Original query string for additional context matching
            metadata: Full chunk metadata for project-specific boosting

        Returns:
            Boost factor (0.0 = no boost, up to 10.0 for exact project matches)
        """
        filename_lower = filename.lower()
        query_lower = query.lower() if query else ""
        boost = 0.0

        # =====================================================================
        # PROJECT-SPECIFIC BOOSTING (Highest Priority)
        # When query mentions a specific project, boost matching project readmes
        # =====================================================================
        if metadata and metadata.get("document_type") == "AURIX_projects":
            project_name = metadata.get("project_name", "").lower()
            source = metadata.get("source", "").lower()

            # Extract key project keywords from query
            # e.g., "EGTM ATOM 3 Phase Inverter PWM 1" -> ["egtm", "atom", "3", "phase", "inverter", "pwm", "1"]
            project_keywords = self._extract_project_keywords_from_query(query_lower)

            if project_keywords and project_name:
                # Count how many project keywords match the project name
                matches = sum(
                    1 for kw in project_keywords if kw in project_name or kw in source
                )
                total_keywords = len(project_keywords)

                # CRITICAL: Check for exact project suffix number match (e.g., PWM_1 vs PWM_2)
                # Extract the trailing number from query AFTER specific keywords like PWM, ADC, etc.
                # This avoids matching "3 Phase" when we want "PWM 2"
                # Pattern: Look for version numbers at the END of query, or after common peripheral keywords
                query_suffix_match = re.search(
                    r"(?:pwm|adc|can|spi|uart|dma|timer|encoder|demo|shell)[_\s]?(\d)(?:\s|$|[?\.])",
                    query_lower,
                )
                project_suffix_match = re.search(r"[_](\d)(?:_readme)?$", project_name)

                suffix_matches = True
                if query_suffix_match and project_suffix_match:
                    # Both have a suffix number - they must match exactly
                    query_suffix = query_suffix_match.group(1)
                    project_suffix = project_suffix_match.group(1)
                    suffix_matches = query_suffix == project_suffix
                elif query_suffix_match and not project_suffix_match:
                    # Query has suffix but project doesn't - not a good match
                    suffix_matches = False

                if total_keywords > 0:
                    match_ratio = matches / total_keywords

                    # If suffix doesn't match, significantly reduce boost
                    if not suffix_matches:
                        # Give a small boost for related projects, but not the target
                        if match_ratio >= 0.5:
                            boost = 1.0  # Minor boost for similar but not exact match
                        return boost

                    # Very high boost for strong project name matches WITH correct suffix
                    if match_ratio >= 0.7:  # 70%+ keyword match
                        boost = (
                            10.0  # Highest boost - this is clearly the target project
                        )
                    elif match_ratio >= 0.5:  # 50%+ keyword match
                        boost = 7.0
                    elif match_ratio >= 0.3:  # 30%+ keyword match
                        boost = 4.0
                    elif matches >= 2:  # At least 2 keywords match
                        boost = 2.0

                    # Extra boost if project readme and query asks about pinout/mapping
                    if boost >= 4.0 and any(
                        kw in query_lower for kw in ["pin", "pinout", "mapping"]
                    ):
                        boost += 2.0  # Extra boost for pinout queries

                    if boost > 0:
                        return boost  # Return early - project match takes precedence

        # =====================================================================
        # Standard Architecture Boosting
        # =====================================================================

        # Detect if query is about hardware specs (pin mapping, LED, pinout, etc.)
        hardware_query_keywords = [
            "pin",
            "led",
            "gpio",
            "button",
            "switch",
            "connector",
            "port",
            "pinout",
            "mapping",
            "schematic",
            "hardware",
            "layout",
            "board",
        ]
        is_hardware_query = any(kw in query_lower for kw in hardware_query_keywords)

        # Detect if query is about specific technical topics (PWM, motor control, etc.)
        topic_keywords = {
            "pwm": ["pwm", "pulse width", "pulse-width", "duty cycle"],
            "motor": [
                "motor",
                "inverter",
                "three phase",
                "3-phase",
                "three-phase",
                "bldc",
                "pmsm",
            ],
            "encoder": ["encoder", "quadrature", "incremental encoder"],
            "hall": ["hall sensor", "hall effect", "hall-sensor"],
            "adc": ["adc", "analog to digital", "analog-to-digital"],
            "can": ["can bus", "canbus", "can-bus", "can fd"],
            "spi": ["spi", "serial peripheral"],
            "uart": ["uart", "serial", "asclin"],
            "i2c": ["i2c", "iic", "two wire"],
            "gtm": ["gtm", "generic timer", "tom", "timer output"],
        }

        # Detect topics in query
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_topics.append(topic)

        # Detect if document is a user manual or kit manual (primary hardware documentation)
        is_user_manual = (
            "usermanual" in filename_lower or "user-manual" in filename_lower
        )
        is_kit_manual = "kit" in filename_lower and "manual" in filename_lower
        is_application_note = (
            "applicationnote" in filename_lower or "application-note" in filename_lower
        )
        is_primary_hardware_doc = is_user_manual or is_kit_manual

        # APPLICATION NOTE BOOST: Boost application notes when query matches their topic
        # e.g., "PWM generation" query + "motor-control-power-board-applicationnotes-en.pdf"
        if is_application_note:
            topic_match_score = 0.0
            # Check if application note name matches detected topics
            for topic in detected_topics:
                # motor control application note matches motor, pwm, encoder, hall topics
                if (
                    topic in ["motor", "pwm", "encoder", "hall"]
                    and "motor-control" in filename_lower
                ):
                    topic_match_score = max(topic_match_score, 2.5)
                # General topic match in filename
                elif topic in filename_lower:
                    topic_match_score = max(topic_match_score, 2.0)

            # Check for architecture family match in filename (tc3xx in filename)
            for arch in detected_architectures:
                arch_lower = arch.lower()
                # Extract family (e.g., TC387 -> tc3xx, TC4D7 -> tc4xx)
                if re.match(r"TC([34])", arch, re.IGNORECASE):
                    family_num = re.match(r"TC([34])", arch, re.IGNORECASE).group(1)
                    if f"tc{family_num}xx" in filename_lower:
                        topic_match_score = max(
                            topic_match_score, topic_match_score + 0.5
                        )

            if topic_match_score > 0:
                boost = max(boost, topic_match_score)

        # No architecture detection needed for topic-based boost
        if not detected_architectures:
            return boost

        for arch in detected_architectures:
            arch_lower = arch.lower()

            # HIGHEST boost: Hardware query + exact architecture + user/kit manual
            # e.g., "TC375 LED pins" query + "infineon-aurix-tc375-lite-kit-usermanual-en.pdf"
            if (
                is_hardware_query
                and arch_lower in filename_lower
                and is_primary_hardware_doc
            ):
                boost = max(boost, 3.0)  # Strongest boost for exact hardware doc match
                continue

            # Very high boost: Exact architecture + "lite kit" in both query and filename
            if (
                arch_lower in filename_lower
                and "lite" in filename_lower
                and "kit" in filename_lower
            ):
                if (
                    "lite" in query_lower
                    or "kit" in query_lower
                    or "manual" in query_lower
                ):
                    boost = max(boost, 2.5)  # Very strong boost
                    continue

            # High boost: Exact architecture match + kit/manual in filename
            if arch_lower in filename_lower:
                if is_primary_hardware_doc:
                    boost = max(boost, 2.0)
                elif "kit" in filename_lower or "manual" in filename_lower:
                    boost = max(boost, 1.5)
                else:
                    boost = max(boost, 0.8)
                continue

            # Kit manual boost - if searching for specific arch, boost its kit manuals
            if is_kit_manual:
                # Check if this kit manual relates to the architecture family
                # TC375 -> TC37x family -> tc3x7 kit manual
                # TC387 -> TC38x family -> tc3x7 kit manual (generic family manual)
                if re.match(r"TC(\d{3,4})$", arch, re.IGNORECASE):
                    arch_num = arch[2:]  # "375" from "TC375", "387" from "TC387"
                    family_num = arch_num[:2]  # "37" from "375", "38" from "387"
                    main_family = arch_num[0]  # "3" from "375" or "387"

                    # Check for family pattern in kit manual name
                    # Patterns to match:
                    # - tc3x7, tc3x8, tc3x9 (generic manual covering TC37x, TC38x, TC39x families)
                    # - tc37x (specific to TC37x subfamily)
                    # - tc38x (specific to TC38x subfamily)

                    # Generic family manual pattern - matches tc3x7, tc3x8, tc3x9 for any TC3xx query
                    # This handles cases where tc3x7 manual covers both TC37x AND TC38x
                    generic_pattern = (
                        rf"tc{main_family}x[7-9]"  # tc3x7, tc3x8, tc3x9 for any TC3xx
                    )
                    # Specific subfamily pattern
                    specific_pattern = rf"tc{family_num}[x\d]"  # tc37x, tc38x, etc.

                    if re.search(generic_pattern, filename_lower) or re.search(
                        specific_pattern, filename_lower
                    ):
                        # Extra boost for hardware queries to family kit manuals
                        boost = max(boost, 1.8 if is_hardware_query else 0.8)
                        continue

            # Family match boost (moderate boost)
            # e.g., "TC375" query should boost "tc37x" documents
            if re.match(r"TC(\d{3,4})$", arch, re.IGNORECASE):
                arch_num = arch[2:]  # "375" from "TC375"
                family_pattern = rf"tc{arch_num[:2]}[x\d]"
                if re.search(family_pattern, filename_lower):
                    boost = max(boost, 0.3)

        return boost

    def _extract_project_keywords_from_query(self, query: str) -> List[str]:
        """
        Extract project-relevant keywords from a query for project matching.

        Extracts technical terms like peripheral names (GTM, ATOM, EGTM, ADC, etc.),
        application types (PWM, inverter, encoder), and project-specific identifiers.

        Args:
            query: User's query string (already lowercase)

        Returns:
            List of keywords relevant for project matching
        """
        # Technical peripheral/module keywords
        peripheral_keywords = [
            "egtm",
            "gtm",
            "atom",
            "tom",
            "tim",
            "ccu",
            "gpt",
            "pwm",
            "adc",
            "tmadc",
            "vadc",
            "dsadc",
            "dac",
            "asclin",
            "uart",
            "spi",
            "i2c",
            "lin",
            "can",
            "eth",
            "dma",
            "dmac",
            "cpu",
            "mpu",
            "pms",
            "smu",
            "stm",
            "flash",
            "overlay",
            "interrupt",
            "watchdog",
            "gpio",
            "qspi",
            "sent",
            "psi5",
            "mcmcan",
            "eray",
            "encoder",
            "inverter",
            "motor",
            "blinky",
            "led",
            "button",
            "multicore",
            "virtualization",
            "hypervisor",
            "cdsp",
            "trap",
        ]

        # Application/feature keywords
        application_keywords = [
            "phase",
            "pwm",
            "capture",
            "timestamp",
            "filtering",
            "shell",
            "printf",
            "demo",
            "master",
            "slave",
            "send",
            "receive",
            "protection",
            "memory",
            "alarm",
            "single",
            "multiple",
            "channel",
            "incremental",
            "assembly",
            "performance",
            "counter",
            "linked",
            "list",
        ]

        keywords = []
        query_words = re.split(r"[\s_\-]+", query)

        # Extract matching peripheral keywords
        for word in query_words:
            word_clean = re.sub(r"[^a-z0-9]", "", word)
            if word_clean in peripheral_keywords:
                keywords.append(word_clean)
            elif word_clean in application_keywords:
                keywords.append(word_clean)
            # Also capture numbers that might be project identifiers (e.g., "1", "2")
            elif word_clean.isdigit() and 1 <= int(word_clean) <= 9:
                keywords.append(word_clean)
            # Capture compound matches like "3phase" or "3_phase"
            elif any(pk in word_clean for pk in peripheral_keywords):
                for pk in peripheral_keywords:
                    if pk in word_clean:
                        keywords.append(pk)

        # Special handling for common patterns
        if "3" in query and "phase" in query:
            keywords.append("3")
            keywords.append("phase")

        return list(set(keywords))  # Remove duplicates

    def _is_table_of_contents(self, text: str) -> bool:
        """
        Detect if a text chunk is likely from a table of contents.

        Common TOC patterns:
        - Contains mostly page numbers and chapter titles
        - Has patterns like "Chapter 1... 5", "Section 2.1... 23"
        - Contains dots or lines connecting titles to page numbers
        - Short lines with numbers at the end
        - Keywords like "Contents", "Table of Contents", "Index"

        Args:
            text: Text chunk to analyze

        Returns:
            True if text appears to be table of contents
        """
        if not text or len(text.strip()) < 20:
            return False

        text_lower = text.lower().strip()

        # Direct TOC keywords
        toc_keywords = [
            "table of contents",
            "contents",
            "table of content",
            "list of figures",
            "list of tables",
            "index",
        ]

        if any(keyword in text_lower for keyword in toc_keywords):
            return True

        lines = text.split("\n")
        total_lines = len([line for line in lines if line.strip()])

        if total_lines < 3:  # Need minimum lines to detect patterns
            return False

        # Count lines that look like TOC entries
        toc_pattern_lines = 0
        page_number_lines = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Pattern: text followed by dots and page number
            # e.g., "Introduction........................5"
            if re.search(r"[.\s]{3,}\d+\s*$", line):
                toc_pattern_lines += 1
                continue

            # Pattern: ends with page number (possibly with spaces/tabs)
            # e.g., "Chapter 1    15", "Section 2.1\t\t\t23"
            if re.search(r"\s+\d+\s*$", line) and len(line) > 10:
                page_number_lines += 1
                continue

            # Pattern: chapter/section numbering
            # e.g., "1. Introduction", "2.1 Overview", "Chapter 3"
            if re.search(r"^\s*(?:chapter\s+)?\d+(?:\.\d+)*[.\s]", line.lower()):
                toc_pattern_lines += 1
                continue

        # High ratio of TOC-like lines suggests this is a TOC
        toc_ratio = (toc_pattern_lines + page_number_lines) / total_lines

        # Also check for common TOC section headers
        section_headers = [
            "abstract",
            "introduction",
            "overview",
            "summary",
            "conclusion",
            "appendix",
            "references",
            "bibliography",
            "glossary",
            "acronyms",
            "abbreviations",
        ]

        header_matches = sum(
            1
            for line in lines
            for header in section_headers
            if header in line.lower().strip()
        )

        # Multiple section headers + high TOC pattern ratio = likely TOC
        if header_matches >= 2 and toc_ratio > 0.4:
            return True

        # High TOC pattern ratio alone
        if toc_ratio > 0.6:
            return True

        return False

    def get_available_sources(self) -> List[str]:
        """
        Get list of unique source documents in the vector store.
        Useful for building filter UI options.

        Returns:
            List of unique source document names
        """
        sources = set()
        for meta in self.metadata:
            source = meta.get("source", "")
            if source:
                sources.add(source)
        return sorted(list(sources))

    def get_page_range(self, source: Optional[str] = None) -> Tuple[int, int]:
        """
        Get the page range for documents in the vector store.

        Args:
            source: Optional source filter to get range for specific document

        Returns:
            Tuple of (min_page, max_page)
        """
        pages = []
        for meta in self.metadata:
            if source and meta.get("source", "") != source:
                continue
            page = meta.get("page", 0)
            if isinstance(page, int):
                pages.append(page)

        if not pages:
            return (0, 0)
        return (min(pages), max(pages))

    def save(self, output_dir: Path):
        """Save vector store to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save texts and metadata
        with open(output_dir / "texts.json", "w", encoding="utf-8") as f:
            json.dump(self.texts, f, indent=2)

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        # Save FAISS index
        if self.index is not None:
            try:
                import faiss

                faiss.write_index(self.index, str(output_dir / "faiss_index.bin"))
            except:
                pass

        logger.info(f"Vector store saved to {output_dir}")

    def load(self, output_dir: Path) -> bool:
        """Load vector store from disk."""
        texts_path = output_dir / "texts.json"
        meta_path = output_dir / "metadata.json"

        if not texts_path.exists() or not meta_path.exists():
            return False

        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Load FAISS index
        index_path = output_dir / "faiss_index.bin"
        if index_path.exists():
            try:
                import faiss

                self.index = faiss.read_index(str(index_path))
            except:
                pass
        else:
            # Rebuild FAISS index from stored texts (one-time cost, then persisted)
            logger.info(
                f"faiss_index.bin not found - rebuilding FAISS index from "
                f"{len(self.texts)} chunks (this happens once)..."
            )
            try:
                import os
                import faiss
                import numpy as np
                import torch

                # Maximise CPU parallelism for PyTorch inference
                num_threads = os.cpu_count() or 4
                torch.set_num_threads(num_threads)
                logger.info(f"  Using {num_threads} CPU threads for embedding")

                # Ensure model is loaded before timing the encode pass
                self._ensure_embeddings_loaded()

                if self._embed_model is not None:
                    # Let sentence_transformers handle batching internally –
                    # one encode() call is significantly faster than many small
                    # calls because it avoids repeated Python/PyTorch overhead.
                    logger.info("  Encoding all chunks (single pass, batch_size=256)...")
                    all_embeddings = self._embed_model.encode(
                        self.texts,
                        batch_size=256,
                        show_progress_bar=True,
                        convert_to_numpy=True,
                    ).astype("float32")

                    dim = all_embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dim)
                    self.index.add(all_embeddings)
                    # Persist so future loads skip this rebuild
                    faiss.write_index(self.index, str(index_path))
                    logger.info(
                        f"FAISS index built and saved to {index_path} "
                        f"({len(self.texts)} vectors)"
                    )
            except ImportError:
                logger.warning("faiss-cpu not installed. Queries will be slow.")
            except Exception as e:
                logger.warning(f"Failed to rebuild FAISS index: {e}")

        logger.info(f"Loaded {len(self.texts)} chunks from {output_dir}")
        return True


# ============================================================================
# Main RAG Agent
# ============================================================================


class InfineonRAGAgent:
    """
    Unified RAG Agent for Infineon board documentation.

    Combines text chunking, image extraction, CLIP embeddings,
    and GPT Vision analysis for comprehensive document querying.
    """

    def __init__(self, config: Optional[RAGAgentConfig] = None):
        """Initialize the RAG Agent."""
        self.config = config or RAGAgentConfig()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.text_chunker = TextChunker(self.config)
        self.image_extractor = (
            ImageExtractor(self.config) if self.config.enable_image_extraction else None
        )
        self.clip_embedder = (
            CLIPImageEmbedder(self.config)
            if self.config.enable_clip_embeddings
            else None
        )
        self.vector_store = VectorStoreManager(self.config)

        # OpenAI client for answer generation
        self.openai_client = None
        self._init_openai()

        # Track processed documents
        self.processed_docs: Dict[str, Dict[str, Any]] = {}

        # Load board-to-package mapping
        self.board_package_mapping = self._load_board_package_mapping()

        # Folder manager for architecture detection (shared by query and interactive mode)
        self.folder_manager = DocumentFolderManager(self.config.documents_dir)

        # Load existing index if available
        self._load_existing_index()

        # Load processed documents registry
        self._load_processed_registry()

        logger.info("Infineon RAG Agent initialized")

    def _load_board_package_mapping(self) -> Dict[str, Any]:
        """Load board-to-package mapping from proj_package_mapping.json."""
        mapping_path = SCRIPT_DIR / "proj_package_mapping.json"
        if not mapping_path.exists():
            logger.warning(f"Board package mapping file not found: {mapping_path}")
            return {}

        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(
                    f"Loaded board-to-package mapping: {len(data.get('project_mappings', {}))} architectures"
                )
                return data.get("project_mappings", {})
        except Exception as e:
            logger.error(f"Failed to load board package mapping: {e}")
            return {}

    def _extract_board_info(self, query: str) -> Optional[Dict[str, str]]:
        """Extract board model and package info from query.

        Detection priority:
          1. Exact board name match (e.g. KIT_A2G_TC387_5V_TFT in query)
          2. Alias match (e.g. KIT_TC375_LK in query)
          3. Bare MCU name match (e.g. "TC387", "TC375", "TC4D7" in query)
             -> resolved dynamically against proj_package_mapping.json

        Args:
            query: User query text

        Returns:
            Dict with 'board', 'package', 'architecture', 'kit_manual', 'aliases'
            and optionally 'detected_mcu' if resolved via bare MCU name.
            Returns None if nothing could be matched.
        """
        if not self.board_package_mapping:
            return None

        query_upper = query.upper()

        # ------------------------------------------------------------------ #
        # Priority 1 & 2: exact board name or alias present in query          #
        # ------------------------------------------------------------------ #
        for arch, boards in self.board_package_mapping.items():
            for board_name, board_info in boards.items():
                # Check if board name is in query
                if board_name.upper() in query_upper:
                    return {
                        "board": board_name,
                        "package": board_info.get("package", "Unknown"),
                        "architecture": arch,
                        "kit_manual": board_info.get("kit_manual", ""),
                        "aliases": board_info.get("aliases", []),
                        "notes": board_info.get("notes", ""),
                    }

                # Check aliases
                for alias in board_info.get("aliases", []):
                    if alias.upper() in query_upper:
                        return {
                            "board": board_name,
                            "package": board_info.get("package", "Unknown"),
                            "architecture": arch,
                            "kit_manual": board_info.get("kit_manual", ""),
                            "aliases": board_info.get("aliases", []),
                            "notes": board_info.get("notes", ""),
                        }

        # ------------------------------------------------------------------ #
        # Priority 3: bare MCU name in query (e.g. "TC387", "TC375", "TC4D7") #
        # Map MCU name → arch family → best board → kit_manual               #
        # ------------------------------------------------------------------ #
        # Pattern: TC followed by a digit then 2-3 more alphanumeric chars
        # Matches: TC375, TC387, TC4D7, TC397, TC334, TC367, TC377, etc.
        mcu_name_re = re.compile(r'\bTC(\d[A-Z0-9]{2,3})\b', re.IGNORECASE)

        best_board_name: Optional[str] = None
        best_board_data: Optional[Dict] = None
        best_arch_key: Optional[str] = None
        best_mcu_full: Optional[str] = None
        best_score: float = -1.0

        for mcu_match in mcu_name_re.finditer(query):
            mcu_full = mcu_match.group(0).upper()       # "TC387", "TC375", "TC4D7"
            mcu_suffix = mcu_match.group(1).upper()     # "387",  "375",  "4D7"

            # Build a compact digit-only prefix to match arch families:
            #   "387" → "38"  (TC38x family)
            #   "375" → "37"  (TC37x family)
            #   "4D7" → "4"   (TC4xx family — only first digit is numeric)
            digits_only = ''.join(c for c in mcu_suffix if c.isdigit())
            if len(digits_only) >= 2:
                family_prefix = digits_only[:2]   # "38", "37", "39", …
            elif len(digits_only) == 1:
                family_prefix = digits_only        # "4"
            else:
                continue  # no digits found — skip

            for arch_key, boards_in_arch in self.board_package_mapping.items():
                # Extract digits from arch key: "TC38x" → "38", "TC4xx" → "4"
                arch_digits = ''.join(c for c in arch_key[2:] if c.isdigit())
                if not arch_digits:
                    continue

                # Family must match: MCU "38…" ↔ arch "38", MCU "4…" ↔ arch "4"
                if not (family_prefix.startswith(arch_digits)
                        or arch_digits.startswith(family_prefix)):
                    continue

                # Found a matching arch family — score each board inside it
                for board_name, binfo in boards_in_arch.items():
                    score: float = 0.0

                    # Strong match: board name literally contains the MCU number
                    if mcu_full in board_name.upper():
                        score = 3.0
                    elif mcu_suffix in board_name.upper():
                        score = 2.0
                    else:
                        # Check aliases as well
                        for alias in binfo.get("aliases", []):
                            if mcu_full in alias.upper() or mcu_suffix in alias.upper():
                                score = max(score, 1.5)
                        # Weak match: arch family matches but no direct MCU hit
                        if score == 0.0:
                            score = 0.5

                    # Slight preference for boards that have a kit_manual defined
                    if binfo.get("kit_manual"):
                        score += 0.1

                    if score > best_score:
                        best_score = score
                        best_board_name = board_name
                        best_board_data = binfo
                        best_arch_key = arch_key
                        best_mcu_full = mcu_full

        if best_board_name and best_board_data is not None:
            return {
                "board": best_board_name,
                "package": best_board_data.get("package", "Unknown"),
                "architecture": best_arch_key,
                "kit_manual": best_board_data.get("kit_manual", ""),
                "aliases": best_board_data.get("aliases", []),
                "notes": best_board_data.get("notes", ""),
                "detected_mcu": best_mcu_full,
            }

        return None

    def _apply_kit_manual_routing(
        self,
        text_results: List[Dict[str, Any]],
        target_kit_manual: str,
    ) -> List[Dict[str, Any]]:
        """
        Dynamically route retrieval results toward the correct board kit manual.

        When a specific MCU or board is mentioned in the query the correct kit
        manual is known (e.g. TC387 → infineon-applicationkitmanual-tc3x7…).
        This method:
          - Boosts (×3.0) chunks from the *target* kit manual
          - Penalises (×0.15) chunks from *other* board-specific kit manuals
            so conflicting board docs (e.g. the TC375 lite-kit manual when
            answering a TC387 question) no longer dominate results.

        Generic cross-architecture documents (datasheets, app-notes, user guides
        without a specific kit-manual entry) are left untouched.

        Args:
            text_results: Retrieved text/chunk list from VectorStoreManager.
            target_kit_manual: Filename of the authoritative kit manual for the
                               detected board (from proj_package_mapping.json).

        Returns:
            Re-scored and re-sorted list of chunks.
        """
        if not target_kit_manual or not text_results:
            return text_results

        target_lower = target_kit_manual.lower().replace(".pdf", "")

        # Collect all kit_manual filenames that belong to OTHER boards
        other_kit_manuals: set = set()
        for arch, boards in self.board_package_mapping.items():
            for board_name, binfo in boards.items():
                km = binfo.get("kit_manual", "")
                if km:
                    km_lower = km.lower().replace(".pdf", "")
                    if km_lower and km_lower != target_lower:
                        other_kit_manuals.add(km_lower)

        routed: List[Dict[str, Any]] = []
        for chunk in text_results:
            source = (
                chunk.get("source", "") or chunk.get("source_pdf", "")
            ).lower().replace(".pdf", "")

            # Check if chunk belongs to the correct kit manual
            is_target = target_lower in source or source in target_lower

            # Check if chunk belongs to another board's kit manual
            is_wrong_board = not is_target and any(
                other_km in source or source in other_km
                for other_km in other_kit_manuals
            )

            if is_target or is_wrong_board:
                chunk = dict(chunk)  # shallow copy to avoid mutating shared dicts
                if is_target:
                    chunk["score"] = chunk.get("score", 0) * 3.0
                    chunk["_kit_manual_boosted"] = True
                else:
                    chunk["score"] = chunk.get("score", 0) * 0.15
                    chunk["_kit_manual_penalized"] = True

            routed.append(chunk)

        routed.sort(key=lambda c: c.get("score", 0), reverse=True)
        return routed

    def _init_openai(self):
        """Initialize OpenAI client for answer generation."""
        # Fallback to standard OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for answer generation")
            except ImportError:
                logger.warning("openai package not installed")

    def _load_existing_index(self):
        """Load existing vector store if available."""
        index_dir = self.config.output_dir / "index"
        if index_dir.exists():
            self.vector_store.load(index_dir)

    def _load_processed_registry(self):
        """Load registry of already processed documents."""
        registry_path = self.config.output_dir / "processed_documents.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    self.processed_docs = json.load(f)
                logger.info(
                    f"Loaded registry: {len(self.processed_docs)} documents previously processed"
                )
            except Exception as e:
                logger.warning(f"Failed to load processed registry: {e}")
                self.processed_docs = {}

    def _save_processed_registry(self):
        """Save registry of processed documents."""
        registry_path = self.config.output_dir / "processed_documents.json"
        try:
            with open(registry_path, "w") as f:
                json.dump(self.processed_docs, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save processed registry: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _is_document_processed(self, pdf_path: Path) -> bool:
        """Check if document was already processed."""
        if self.config.force_reprocess:
            return False

        doc_key = str(pdf_path.absolute())
        if doc_key not in self.processed_docs:
            return False

        # Check if file hash matches (file might have been modified)
        try:
            current_hash = self._compute_file_hash(pdf_path)
            stored_hash = self.processed_docs[doc_key].get("file_hash", "")
            return current_hash == stored_hash
        except Exception as e:
            logger.warning(f"Failed to verify file hash for {pdf_path.name}: {e}")
            return False

    def _extract_architecture_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract architecture information from document path and filename.

        This enriches document metadata during ingestion for better filtering.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with architecture tags: product_family, architecture, document_type
        """
        import re

        path_str = str(pdf_path).replace("\\", "/").lower()
        filename = pdf_path.name.lower()

        metadata = {"product_family": None, "architecture": None, "document_type": None}

        # Detect product family from path
        if "tc3xx" in path_str or "tc3" in path_str:
            metadata["product_family"] = "TC3xx"
        elif "tc4xx" in path_str or "tc4" in path_str:
            metadata["product_family"] = "TC4xx"

        # Detect specific architecture from path or filename
        # TC3xx family architectures
        arch_patterns = {
            r"\btc38[x0-9]": "TC38x",
            r"\btc39[x0-9]": "TC39x",
            r"\btc37[x0-9]": "TC37x",
            r"\btc36[x0-9]": "TC36x",
            r"\btc35[x0-9]": "TC35x",
            r"\btc33[x0-9]": "TC33x",
            r"\btc32[x0-9]": "TC32x",
            # TC4xx family
            r"\btc4[d0-9][x0-9]": "TC4xx",
        }

        for pattern, arch_name in arch_patterns.items():
            if re.search(pattern, path_str) or re.search(pattern, filename):
                metadata["architecture"] = arch_name
                break

        # Detect document type from filename
        doc_type_patterns = {
            "datasheet": r"datasheet",
            "user_manual": r"usermanual|user-manual",
            "application_note": r"applicationnote|application-note",
            "errata": r"errata",
            "quick_training": r"quick[_-]?training",
            "expert_training": r"expert[_-]?training",
            "kit_manual": r"kit[_-]?manual",
        }

        for doc_type, pattern in doc_type_patterns.items():
            if re.search(pattern, filename):
                metadata["document_type"] = doc_type
                break

        return metadata

    def ingest_documents(
        self, subfolder: str = ".", file_pattern: str = "**/*.pdf"
    ) -> Dict[str, Any]:
        """
        Ingest documents from a subfolder or absolute path (recursively finds all PDFs).

        Args:
            subfolder: Subfolder within documents_dir (e.g., "kit_manual", "datasheets/TC3xx")
                      OR absolute path to a custom folder (e.g., "C:/Users/Documents/PDFs")
                      OR "." to search entire documents_dir (default)
            file_pattern: Glob pattern for files to process (default: **/*.pdf for recursive search)

        Returns:
            Ingestion statistics
        """
        # Determine if subfolder is absolute path or relative
        input_path = Path(subfolder)

        if input_path.is_absolute():
            # User provided absolute path
            docs_path = input_path
            logger.info(f"Using absolute path: {docs_path}")
        elif subfolder == ".":
            # Search entire documents_dir
            docs_path = self.config.documents_dir
            logger.info(f"Searching entire documents directory: {docs_path}")
        else:
            # Relative to documents_dir
            docs_path = self.config.documents_dir / subfolder
            logger.info(f"Using relative path: {docs_path}")

        if not docs_path.exists():
            raise FileNotFoundError(f"Documents folder not found: {docs_path}")

        # Recursively find all PDF files using rglob (recursive glob)
        pdf_files = list(docs_path.rglob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {docs_path} (searched recursively)"
            )

        logger.info(
            f"Found {len(pdf_files)} PDF documents to process (searched recursively)"
        )

        stats = {
            "input_path": str(docs_path),
            "is_absolute_path": input_path.is_absolute(),
            "files_found": len(pdf_files),
            "documents_processed": 0,
            "documents_skipped": 0,
            "total_chunks": 0,
            "total_images": 0,
            "chunks_with_images": 0,
            "details": [],
            "skipped": [],
        }

        for pdf_path in pdf_files:
            # Check if already processed
            if self._is_document_processed(pdf_path):
                logger.info(f"Skipping already processed: {pdf_path.name}")
                stats["documents_skipped"] += 1
                stats["skipped"].append(str(pdf_path.name))
                continue

            doc_stats = self._process_document(pdf_path)
            stats["details"].append(doc_stats)
            stats["documents_processed"] += 1
            stats["total_chunks"] += doc_stats.get("chunks_created", 0)
            stats["total_images"] += doc_stats.get("images_extracted", 0)
            stats["chunks_with_images"] += doc_stats.get("chunks_with_images", 0)

        # Save vector store
        self.vector_store.save(self.config.output_dir / "index")

        # Save processed documents registry
        self._save_processed_registry()

        # Save processing stats
        with open(self.config.output_dir / "ingestion_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(
            f"Ingestion complete: {stats['documents_processed']} processed, "
            f"{stats['documents_skipped']} skipped, "
            f"{stats['total_chunks']} chunks, {stats['total_images']} images"
        )

        return stats

    def ingest_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single PDF file.

        Args:
            file_path: Absolute path to the PDF file

        Returns:
            Ingestion statistics
        """
        pdf_path = Path(file_path)

        # Validate file exists and is PDF
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF: {pdf_path}")

        logger.info(f"Processing single file: {pdf_path.name}")

        # Process the document
        doc_stats = self._process_document(pdf_path)

        # Save vector store
        self.vector_store.save(self.config.output_dir / "index")

        # Create overall stats
        stats = {
            "mode": "single_file",
            "file_path": str(pdf_path),
            "documents_processed": 1,
            "total_chunks": doc_stats.get("chunks_created", 0),
            "total_images": doc_stats.get("images_extracted", 0),
            "details": [doc_stats],
        }

        # Save processing stats
        with open(self.config.output_dir / "ingestion_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(
            f"Single file ingestion complete: {stats['total_chunks']} chunks, "
            f"{stats['total_images']} images"
        )

        return stats

    def ingest_project_readmes(self, force: bool = False) -> Dict[str, Any]:
        """
        Ingest project README files from TC3xx_projects and TC4xx_projects folders.

        Features:
        - Detects and skips already ingested projects
        - Stores project name in metadata for duplicate detection
        - Supports both TC3xx and TC4xx project structures

        Args:
            force: If True, re-ingest all projects even if already processed

        Returns:
            Dict with ingestion statistics
        """
        import re

        # Define project folder paths
        project_folders = [
            self.config.documents_dir / "AURIX" / "AURIX TC3xx" / "TC3xx_projects",
            self.config.documents_dir / "AURIX" / "AURIX TC4xx" / "TC4xx_projects",
        ]

        stats = {
            "mode": "project_readmes",
            "folders_scanned": [],
            "projects_found": 0,
            "projects_ingested": 0,
            "projects_skipped": 0,
            "total_chunks": 0,
            "skipped_projects": [],
            "ingested_projects": [],
            "errors": [],
        }

        # Load existing processed projects registry
        projects_registry_path = self.config.output_dir / "processed_projects.json"
        processed_projects = {}
        if projects_registry_path.exists() and not force:
            try:
                with open(projects_registry_path, "r") as f:
                    processed_projects = json.load(f)
                logger.info(
                    f"Loaded registry: {len(processed_projects)} projects previously processed"
                )
            except Exception as e:
                logger.warning(f"Failed to load projects registry: {e}")

        all_readme_files = []

        # Scan for project readme files
        for proj_folder in project_folders:
            if not proj_folder.exists():
                logger.warning(f"Project folder not found: {proj_folder}")
                continue

            stats["folders_scanned"].append(str(proj_folder))
            logger.info(f"Scanning for projects in: {proj_folder}")

            # Recursively find readme files (they don't have .md extension in this case)
            for readme_path in proj_folder.rglob("*_readme"):
                all_readme_files.append(readme_path)

            # Also look for README.md files
            for readme_path in proj_folder.rglob("README.md"):
                all_readme_files.append(readme_path)
            for readme_path in proj_folder.rglob("readme.md"):
                all_readme_files.append(readme_path)

        stats["projects_found"] = len(all_readme_files)
        logger.info(f"Found {len(all_readme_files)} project readme files")

        for readme_path in all_readme_files:
            # Extract project name from path
            # e.g., .../TC387/iLLD_TC387_ADS_GTM_TOM_3_Phase_Inverter_PWM_1/readme
            project_name = readme_path.parent.name
            project_key = str(readme_path.absolute())

            # Check if already processed (by project name or path)
            if not force and (
                project_key in processed_projects or project_name in processed_projects
            ):
                logger.info(f"Skipping already ingested project: {project_name}")
                stats["projects_skipped"] += 1
                stats["skipped_projects"].append(project_name)
                continue

            try:
                # Read readme content
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    logger.warning(f"Empty readme file: {readme_path}")
                    stats["errors"].append(f"Empty file: {project_name}")
                    continue

                # Determine architecture from path
                path_str = str(readme_path).replace("\\", "/").lower()
                product_family = (
                    "TC4xx" if "tc4xx" in path_str or "tc4d" in path_str else "TC3xx"
                )

                # Extract more specific architecture
                architecture = None
                arch_match = re.search(r"tc(\d{3})", path_str, re.IGNORECASE)
                if arch_match:
                    arch_num = arch_match.group(1)
                    architecture = (
                        f"TC{arch_num[0]}{arch_num[1]}x"  # e.g., TC387 -> TC38x
                    )

                # Create chunks from readme content
                # Split by headers or paragraphs for better chunking
                chunks = self._chunk_readme_content(
                    content,
                    project_name=project_name,
                    source_path=str(readme_path),
                    product_family=product_family,
                    architecture=architecture,
                )

                # Add chunks to vector store
                self.vector_store.add_chunks(chunks)

                # Record as processed
                processed_projects[project_key] = {
                    "project_name": project_name,
                    "ingested_at": str(Path(readme_path).stat().st_mtime),
                    "chunks_created": len(chunks),
                    "product_family": product_family,
                    "architecture": architecture,
                }
                # Also store by project name for quick lookup
                processed_projects[project_name] = processed_projects[project_key]

                stats["projects_ingested"] += 1
                stats["total_chunks"] += len(chunks)
                stats["ingested_projects"].append(project_name)

                logger.info(f"Ingested project: {project_name} ({len(chunks)} chunks)")

            except Exception as e:
                logger.error(f"Error processing {project_name}: {e}")
                stats["errors"].append(f"{project_name}: {str(e)}")

        # Save vector store
        self.vector_store.save(self.config.output_dir / "index")

        # Save processed projects registry
        try:
            with open(projects_registry_path, "w") as f:
                json.dump(processed_projects, f, indent=2, default=str)
            logger.info(f"Saved projects registry: {len(processed_projects)} projects")
        except Exception as e:
            logger.error(f"Failed to save projects registry: {e}")

        # Save ingestion stats
        with open(self.config.output_dir / "project_ingestion_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(
            f"Project ingestion complete: {stats['projects_ingested']} ingested, "
            f"{stats['projects_skipped']} skipped, {stats['total_chunks']} chunks"
        )

        return stats

    def _chunk_readme_content(
        self,
        content: str,
        project_name: str,
        source_path: str,
        product_family: str = None,
        architecture: str = None,
        max_chunk_size: int = 1500,
        overlap: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Chunk readme content into smaller pieces for vector storage.

        Args:
            content: Raw readme text content
            project_name: Name of the project
            source_path: Path to the source file
            product_family: TC3xx or TC4xx
            architecture: Specific architecture (e.g., TC38x)
            max_chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks

        Returns:
            List of chunk dictionaries with metadata
        """
        import re

        chunks = []

        # Try to split by markdown headers first
        # Match ## or # headers
        header_pattern = r"^(#{1,3})\s+(.+)$"
        sections = re.split(r"(?=^#{1,3}\s+)", content, flags=re.MULTILINE)

        chunk_id = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Extract header if present
            header_match = re.match(header_pattern, section, re.MULTILINE)
            section_title = header_match.group(2).strip() if header_match else None

            # If section is too long, split further
            if len(section) > max_chunk_size:
                # Split by paragraphs
                paragraphs = section.split("\n\n")
                current_chunk = ""

                for para in paragraphs:
                    if len(current_chunk) + len(para) > max_chunk_size:
                        if current_chunk:
                            chunks.append(
                                self._create_project_chunk(
                                    text=current_chunk.strip(),
                                    chunk_id=chunk_id,
                                    project_name=project_name,
                                    source_path=source_path,
                                    section_title=section_title,
                                    product_family=product_family,
                                    architecture=architecture,
                                )
                            )
                            chunk_id += 1
                        current_chunk = para
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para

                if current_chunk.strip():
                    chunks.append(
                        self._create_project_chunk(
                            text=current_chunk.strip(),
                            chunk_id=chunk_id,
                            project_name=project_name,
                            source_path=source_path,
                            section_title=section_title,
                            product_family=product_family,
                            architecture=architecture,
                        )
                    )
                    chunk_id += 1
            else:
                chunks.append(
                    self._create_project_chunk(
                        text=section,
                        chunk_id=chunk_id,
                        project_name=project_name,
                        source_path=source_path,
                        section_title=section_title,
                        product_family=product_family,
                        architecture=architecture,
                    )
                )
                chunk_id += 1

        # If no chunks were created (no headers), create one or more chunks from content
        if not chunks:
            if len(content) > max_chunk_size:
                # Simple character-based chunking with overlap
                for i in range(0, len(content), max_chunk_size - overlap):
                    chunk_text = content[i : i + max_chunk_size]
                    if chunk_text.strip():
                        chunks.append(
                            self._create_project_chunk(
                                text=chunk_text.strip(),
                                chunk_id=chunk_id,
                                project_name=project_name,
                                source_path=source_path,
                                section_title=None,
                                product_family=product_family,
                                architecture=architecture,
                            )
                        )
                        chunk_id += 1
            else:
                chunks.append(
                    self._create_project_chunk(
                        text=content.strip(),
                        chunk_id=0,
                        project_name=project_name,
                        source_path=source_path,
                        section_title=None,
                        product_family=product_family,
                        architecture=architecture,
                    )
                )

        return chunks

    def _create_project_chunk(
        self,
        text: str,
        chunk_id: int,
        project_name: str,
        source_path: str,
        section_title: str = None,
        product_family: str = None,
        architecture: str = None,
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with project metadata."""
        chunk = {
            "text": text,
            "chunk_id": chunk_id,
            "source": f"{project_name}_readme",
            "source_path": source_path,
            "project_name": project_name,
            "document_type": "AURIX_projects",
            "is_project_readme": True,
        }

        if section_title:
            chunk["section_title"] = section_title
        if product_family:
            chunk["product_family"] = product_family
        if architecture:
            chunk["architecture"] = architecture

        return chunk

    def _process_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single document and enrich with architecture metadata."""
        doc_name = pdf_path.stem
        logger.info(f"Processing: {pdf_path.name}")

        # Compute file hash for deduplication
        file_hash = self._compute_file_hash(pdf_path)

        # Extract architecture information from path and filename
        architecture_tags = self._extract_architecture_metadata(pdf_path)

        doc_output_dir = self.config.output_dir / doc_name
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "document": pdf_path.name,
            "architecture_tags": architecture_tags,
            "chunks_created": 0,
            "images_extracted": 0,
            "images_analyzed": 0,
            "chunks_with_images": 0,
        }

        # Step 1: Extract and chunk text
        chunks = self.text_chunker.process_pdf(pdf_path)

        # Enrich chunks with architecture metadata
        for chunk in chunks:
            chunk.update(architecture_tags)

        stats["chunks_created"] = len(chunks)

        # Step 2: Extract images
        images = []
        if self.image_extractor:
            images = self.image_extractor.extract_images_from_pdf(
                pdf_path, doc_output_dir
            )
            stats["images_extracted"] = len(images)

        # Step 3: Analyze images with GPT Vision (if enabled) and index with CLIP
        if images:

            # CLIP embedding and batch indexing (optimized)
            if self.clip_embedder and self.config.enable_clip_embeddings:
                indexed_count = self.clip_embedder.index_images_batch(
                    images, batch_size=16
                )
                stats["images_indexed"] = indexed_count

        # Step 4: Associate images with chunks (by page number)
        if self.config.associate_images_with_chunks and images:
            chunks, chunks_with_images = self._associate_images_with_chunks(
                chunks, images, self.config.include_nearby_pages
            )
            stats["chunks_with_images"] = chunks_with_images
            logger.info(f"Associated images with {chunks_with_images} chunks")

        # Add to vector store (after image association)
        self.vector_store.add_chunks(chunks)

        # Save chunks to JSON (includes image references)
        with open(doc_output_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        # Save image metadata
        if images:
            with open(doc_output_dir / "images.json", "w", encoding="utf-8") as f:
                json.dump(images, f, indent=2, default=str)

        # Track processed document with hash
        doc_key = str(pdf_path.absolute())
        self.processed_docs[doc_key] = {
            "path": str(pdf_path),
            "file_hash": file_hash,
            "chunks": len(chunks),
            "images": len(images),
            "processed_at": datetime.now().isoformat(),
        }

        return stats

    def _associate_images_with_chunks(
        self,
        chunks: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        nearby_pages: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Associate images with text chunks based on page numbers.

        Args:
            chunks: List of text chunks with 'page' field
            images: List of image metadata with 'page' field
            nearby_pages: Include images from N pages before/after (0 = same page only)

        Returns:
            Tuple of (updated chunks with 'images' field, count of chunks with images)
        """
        if not images:
            return chunks, 0

        # Build page-to-images mapping for fast lookup
        page_images: Dict[int, List[Dict[str, Any]]] = {}
        for img in images:
            page = img.get("page", 0)
            if page not in page_images:
                page_images[page] = []
            # Store minimal image info to keep chunk size manageable
            page_images[page].append(
                {
                    "file_path": img.get("file_path", ""),
                    "filename": img.get("filename", ""),
                    "figure_id": img.get("figure_id", ""),
                    "page": page,
                    "source_path": img.get("source_path", ""),
                    "gpt_description": (
                        img.get("gpt_analysis", {}).get("description", "")[:500]
                        if img.get("gpt_analysis")
                        else ""
                    ),
                }
            )

        chunks_with_images = 0

        for chunk in chunks:
            chunk_page = chunk.get("page", 0)
            chunk_images = []

            # Collect images from the chunk's page and nearby pages
            for page_offset in range(-nearby_pages, nearby_pages + 1):
                target_page = chunk_page + page_offset
                if target_page in page_images:
                    chunk_images.extend(page_images[target_page])

            # Add images to chunk (empty list if no images found)
            chunk["images"] = chunk_images
            if chunk_images:
                chunks_with_images += 1

        return chunks, chunks_with_images

    def _collect_images_from_chunks(
        self, text_sources: List[Dict[str, Any]], max_images: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Collect unique images from retrieved text chunks.

        Args:
            text_sources: Retrieved text chunks with 'images' field
            max_images: Maximum number of images to return

        Returns:
            List of unique image metadata
        """
        # Cap max_images for performance
        max_images = min(max_images, 6)

        seen_paths = set()
        collected_images = []

        # Limit processing of text sources for performance
        for source in text_sources[
            : min(len(text_sources), 10)
        ]:  # Process max 10 sources
            chunk_images = source.get("images", [])
            # Limit chunk images processed per source
            for img in chunk_images[
                : min(len(chunk_images), 5)
            ]:  # Max 5 images per chunk
                img_path = img.get("file_path", "")
                if img_path and img_path not in seen_paths:
                    seen_paths.add(img_path)
                    collected_images.append(
                        {
                            "file_path": img_path,
                            "filename": img.get("filename", ""),
                            "page": img.get("page", 0),
                            "source_chunk_page": source.get("page", 0),
                            "source_path": img.get("source_path", ""),
                            "gpt_description": img.get("gpt_description", ""),
                            "similarity": source.get(
                                "score", 0
                            ),  # Use chunk's relevance score
                        }
                    )

                    if len(collected_images) >= max_images:
                        return collected_images

        return collected_images

    def _prioritize_datasheet_sources(
        self,
        text_results: List[Dict[str, Any]],
        datasheet_sources: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Prioritize datasheet sources in search results for pin-related queries.

        Datasheets contain authoritative pin function tables that should be
        prioritized when answering pin/pinout related questions. This method
        boosts the ranking of chunks from datasheet sources.

        IMPORTANT: If there's a strongly-matched project (boost >= 10.0),
        project sources take precedence over datasheets, as they contain
        the specific pin mappings for that example project.

        Args:
            text_results: Retrieved text chunks from vector search
            datasheet_sources: List of datasheet filename patterns to prioritize
            top_k: Maximum number of results to return

        Returns:
            Reordered list with appropriate prioritization
        """
        if not datasheet_sources or not text_results:
            return text_results

        # CRITICAL: Check if we have strongly-matched project sources
        # If so, project sources should take priority over datasheets
        # because they contain project-specific pin mappings
        strongly_matched_projects = [
            r
            for r in text_results
            if r.get("document_type") == "AURIX_projects"
            and r.get("architecture_boost", 0) >= 10.0
        ]

        if strongly_matched_projects:
            # Project sources found - DON'T prioritize datasheets
            # The project readme has the specific pin mappings we need
            # Just return results as-is, sorted by score (project chunks already boosted)
            results = sorted(
                text_results, key=lambda x: x.get("score", 0), reverse=True
            )
            return results[:top_k]

        # No strongly-matched projects - proceed with datasheet prioritization
        # Separate datasheet results from others
        datasheet_results = []
        other_results = []

        for result in text_results:
            source = result.get("source", "").lower()
            is_datasheet = any(ds.lower() in source for ds in datasheet_sources)

            if is_datasheet:
                # Boost the score for datasheet sources (multiply by 1.5)
                result = result.copy()
                result["score"] = result.get("score", 0) * 1.5
                result["_datasheet_boosted"] = True
                datasheet_results.append(result)
            else:
                other_results.append(result)

        # Sort datasheet results by boosted score (descending)
        datasheet_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Ensure at least some datasheet results are included at the top
        # Take up to 1/3 of results from datasheet, rest from other sources
        datasheet_count = min(len(datasheet_results), max(top_k // 3, 2))
        other_count = top_k - datasheet_count

        # Combine prioritized results
        prioritized_results = datasheet_results[:datasheet_count]
        prioritized_results.extend(other_results[:other_count])

        # If we still have room, add more from either source
        remaining = top_k - len(prioritized_results)
        if remaining > 0:
            # Add remaining datasheet results
            prioritized_results.extend(
                datasheet_results[datasheet_count : datasheet_count + remaining]
            )
            remaining = top_k - len(prioritized_results)
        if remaining > 0:
            # Add remaining other results
            prioritized_results.extend(
                other_results[other_count : other_count + remaining]
            )

        # Final sort by score while keeping some datasheet entries at top
        # Keep first 2 datasheet entries at top, then sort rest by score
        if len(prioritized_results) > 2:
            top_datasheet = [
                r for r in prioritized_results[:4] if r.get("_datasheet_boosted")
            ][:2]
            rest = [r for r in prioritized_results if r not in top_datasheet]
            rest.sort(key=lambda x: x.get("score", 0), reverse=True)
            prioritized_results = top_datasheet + rest

        return prioritized_results[:top_k]

    def query(
        self,
        question: str,
        text_top_k: Optional[int] = None,
        image_top_k: Optional[int] = None,
        mode: str = "hybrid",
        include_chunk_images: bool = True,
        metadata_filter: Optional[MetadataFilter] = None,
        include_table_of_contents: bool = False,
        target_folders: Optional[List[Path]] = None,
        detected_architectures: Optional[List[str]] = None,
        is_migration: bool = False,
        migration_architectures: Optional[List[str]] = None,
        is_pinout_query: bool = False,
        datasheet_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG system with optional metadata filtering and folder targeting.

        Args:
            question: User's question
            text_top_k: Number of text chunks to retrieve
            image_top_k: Number of images to retrieve
            mode: "hybrid" (text + images), "text" (text only), "image" (images only)
            include_chunk_images: Also include images associated with retrieved text chunks
            metadata_filter: Optional MetadataFilter to restrict search to specific documents/pages
            include_table_of_contents: Whether to include table of contents in results (default: False)
            target_folders: Optional list of folder Paths to restrict retrieval to specific architecture folders
            detected_architectures: Optional list of detected architecture names for relevance boosting
            is_migration: Whether this is a migration query between architectures
            migration_architectures: List of architectures involved in migration (e.g., ["TC387", "TC4D7"])
            is_pinout_query: Whether this is a pin/pinout related query requiring datasheet verification
            datasheet_sources: List of datasheet filenames to prioritize for pin verification

        Returns:
            Query result with answer and sources

        Examples:
            # Query all documents
            result = agent.query("What are the GPIO pins?")

            # Query specific document
            result = agent.query(
                "What are the GPIO pins?",
                metadata_filter=MetadataFilter(source="tc375_manual.pdf")
            )

            # Query specific page range
            result = agent.query(
                "Explain the pin mapping",
                metadata_filter=MetadataFilter(page_min=50, page_max=100)
            )

            # Query specific architecture folders
            result = agent.query(
                "What is the clock configuration?",
                target_folders=[Path("docs/AURIX/AURIX TC3xx/TC38x")]
            )

            # Migration query
            result = agent.query(
                "How to migrate PWM code?",
                is_migration=True,
                migration_architectures=["TC387", "TC4D7"]
            )
        """
        text_top_k = text_top_k or self.config.text_top_k
        image_top_k = image_top_k or self.config.image_top_k

        # Limit for performance - prevent memory crashes
        text_top_k = min(text_top_k, 12)
        image_top_k = min(image_top_k, 8)

        # For migration queries, retrieve more context
        if is_migration:
            text_top_k = min(text_top_k + 4, 16)  # More context for migration

        # For pinout queries, also retrieve more context to ensure datasheet coverage
        if is_pinout_query:
            text_top_k = min(text_top_k + 4, 16)  # More context for pin verification

        # Build source path filter from target folders
        if target_folders and not metadata_filter:
            # Create filter to match any of the target folder paths
            folder_paths = [str(f) for f in target_folders]
            # Use custom filter to match folder paths
            metadata_filter = MetadataFilter(custom={"_target_folders": folder_paths})
        elif target_folders and metadata_filter:
            # Merge target folders with existing filter
            if metadata_filter.custom is None:
                metadata_filter.custom = {}
            metadata_filter.custom["_target_folders"] = [str(f) for f in target_folders]

        result = {
            "question": question,
            "answer": "",
            "text_sources": [],
            "image_sources": [],
            "chunk_images": [],  # Images associated with retrieved chunks
            "mode": mode,
            "filter_applied": metadata_filter is not None,
            "target_folders": (
                [str(f) for f in target_folders] if target_folders else None
            ),
            "filter_statistics": None,  # Will be populated if filtering is applied
            "is_migration": is_migration,
            "migration_architectures": migration_architectures,
            "is_pinout_query": is_pinout_query,
            "datasheet_sources": datasheet_sources,
        }

        # Add filter info to result if applied
        if metadata_filter:
            result["filter_info"] = metadata_filter.get_filter_summary()

        # Extract board info early (for all query types, especially pinout queries)
        board_info = self._extract_board_info(question)
        if board_info:
            result["board_info"] = board_info
            logger.info(
                f"Board detected: {board_info['board']} ({board_info['package']} package)"
                + (f" [MCU: {board_info['detected_mcu']}]" if board_info.get("detected_mcu") else "")
            )

            # ------------------------------------------------------------------ #
            # Inject kit_manual into datasheet_sources so _prioritize_datasheet  #
            # boosts chunks from the board-specific manual (e.g. TC387 kit       #
            # manual vs TC375 kit manual).  We do NOT alter detected_architectures#
            # here: architecture-based boosting would favour chip-level docs     #
            # (TC37x datasheets) which can dilute the kit-manual content that    #
            # actually contains the board LED/pin mappings.  The kit_manual      #
            # routing step below is sufficient for board-level queries.          #
            # ------------------------------------------------------------------ #
            kit_manual = board_info.get("kit_manual", "")
            if kit_manual:
                if not datasheet_sources:
                    datasheet_sources = [kit_manual]
                elif kit_manual not in datasheet_sources:
                    datasheet_sources = [kit_manual] + list(datasheet_sources)
                result["datasheet_sources"] = datasheet_sources

        # Retrieve text chunks with optional filtering and architecture boosting
        if mode in ["hybrid", "text"]:
            text_results = self.vector_store.search(
                question,
                top_k=text_top_k,
                metadata_filter=metadata_filter,
                include_table_of_contents=include_table_of_contents,
                detected_architectures=detected_architectures,
            )

            # ------------------------------------------------------------------ #
            # Kit-manual targeted supplementary search                            #
            # When a specific board kit_manual is known, run a secondary DIRECT  #
            # search within that source — bypassing the global FAISS ANN so the  #
            # authoritative board manual is always represented in results, even   #
            # when its raw FAISS rank would fall outside the normal top-k window. #
            # ------------------------------------------------------------------ #
            if board_info and board_info.get("kit_manual"):
                kit_top_k = max(text_top_k // 3, 4)
                kit_results = self.vector_store.search_source_direct(
                    question,
                    source_pattern=board_info["kit_manual"],
                    top_k=kit_top_k,
                )
                if kit_results:
                    seen_hashes = {
                        r.get("hash") for r in text_results if r.get("hash")
                    }
                    added = 0
                    for kr in kit_results:
                        if kr.get("hash") not in seen_hashes:
                            seen_hashes.add(kr.get("hash"))
                            text_results.append(kr)
                            added += 1
                    if added:
                        logger.debug(
                            f"Injected {added} supplementary chunks from "
                            f"kit_manual '{board_info['kit_manual']}'"
                        )

            # For pinout queries, prioritize datasheet sources by boosting their scores
            # and ensuring they appear in results
            if is_pinout_query and datasheet_sources:
                # First try to boost existing datasheet results
                text_results = self._prioritize_datasheet_sources(
                    text_results, datasheet_sources, text_top_k
                )

                # Extract port numbers and full pin references from the question
                # e.g., P10.3 -> port_num=10, full_pins=['P10.3']
                pin_matches = re.findall(
                    r"\b(P(\d{1,2})\.(\d{1,2}))\b", question, re.IGNORECASE
                )
                port_nums = list(set([m[1] for m in pin_matches]))  # ['10']
                full_pins = [m[0].upper() for m in pin_matches]  # ['P10.3', 'P10.5']

                # Use board_info extracted earlier
                detected_package = None

                if board_info:
                    # Use package from mapping (e.g., "BGA-292" -> search for "BGA292" or "BGA292_COM")
                    package_str = board_info["package"]
                    # Normalize package name for search (remove hyphens, handle variations)
                    package_normalized = package_str.replace("-", "").replace(" ", "")
                    detected_package = package_normalized
                    logger.info(
                        f"Board detected: {board_info['board']} -> Package: {package_str} (normalized: {package_normalized})"
                    )
                else:
                    # Fallback: Try to detect package from query text directly
                    package_patterns = [
                        (r"SAK-TC4D7XP-20MF500MC", "BGA292_COM"),
                        (r"BGA292[_\s]?COM", "BGA292_COM"),
                        (r"BGA292", "BGA292"),
                        (r"LFBGA[\-_]?292", "LFBGA292"),
                        (r"QFP[\-_]?176", "QFP176"),
                        (r"LQFP", "LQFP"),
                        (r"TC4D7", "BGA292_COM"),  # TC4D7 defaults to BGA292_COM
                    ]
                    for pattern, package in package_patterns:
                        if re.search(pattern, question, re.IGNORECASE):
                            detected_package = package
                            logger.info(f"Package detected from query text: {package}")
                            break

                # Always do targeted supplementary searches for pin queries
                # to ensure we get the specific port function tables
                if port_nums:
                    datasheet_filter = (
                        MetadataFilter(source=datasheet_sources[0])
                        if datasheet_sources
                        else None
                    )
                    # -------------------------------------------------------
                    # Collect ALL supplementary query strings first, then do
                    # ONE batched embed+search instead of N serial searches.
                    # -------------------------------------------------------
                    all_supp_queries: List[str] = []

                    # Port-level search strategies
                    for port_num in port_nums:
                        if detected_package:
                            all_supp_queries.append(f"Table {detected_package} port {port_num}")
                            all_supp_queries.append(f"{detected_package} P{port_num}")
                        all_supp_queries += [
                            f"BGA292_COM Port {port_num} functions",
                            f"Table BGA292_COM P{port_num}",
                            f"Table Port {port_num} functions",
                            f"continued Port {port_num} functions",
                            f"P{port_num}.2 P{port_num}.3 P{port_num}.4 P{port_num}.5",
                            f"P{port_num}.0 P{port_num}.1 General-purpose",
                            f"EGTM_TOUT P{port_num}",
                            f"Ball Symbol Ctrl Buffer type Function P{port_num}",
                            f"CAN_TXD P{port_num} QSPI ASCLIN",
                        ]

                    # Pin-level search strategies
                    for pin in full_pins:
                        if detected_package:
                            all_supp_queries.append(f"{detected_package} {pin} EGTM_TOUT")
                            all_supp_queries.append(f"{detected_package} {pin} function")
                        all_supp_queries += [
                            f"{pin} General-purpose output EGTM",
                            f"{pin} CAN TXD QSPI alternate",
                            f"{pin} O0 O1 O2 O3 O4 O5 O6 O7 O8 O9",
                        ]

                    # Generic port-function-table searches
                    if detected_package:
                        all_supp_queries.append(f"{detected_package} port functions table")
                        all_supp_queries.append(f"{detected_package} EGTM_TOUT alternate")
                    all_supp_queries += [
                        "EGTM_TOUT104 EGTM_TOUT105 eGTM muxed output",
                        "EGTM_TOUT105 EGTM_TOUT106 EGTM_TOUT107",
                        "General-purpose output Reserved CAN transmit",
                        "Master slave select output Shift clock output",
                        "FAST PU1 VDDEXT ES input output",
                    ]

                    # Single batched embed+search call (replaces 20-30 serial calls)
                    all_supplementary = self.vector_store.search_batch(
                        all_supp_queries,
                        top_k_per_query=3,
                        metadata_filter=datasheet_filter,
                        include_table_of_contents=False,
                        detected_architectures=detected_architectures,
                    )

                    # Deduplicate and add supplementary results
                    seen_hashes = set(
                        r.get("hash") for r in text_results if r.get("hash")
                    )
                    for supp in all_supplementary:
                        source = supp.get("source", "").lower()
                        supp_hash = supp.get("hash")
                        if any(ds.lower() in source for ds in datasheet_sources):
                            if supp_hash and supp_hash not in seen_hashes:
                                seen_hashes.add(supp_hash)
                                supp["_datasheet_boosted"] = True
                                supp["_supplementary"] = True
                                # Boost score for chunks containing the specific pins
                                text_content = supp.get("text", "").upper()
                                for pin in full_pins:
                                    if pin in text_content:
                                        supp["score"] = supp.get("score", 0) * 2.0
                                        break
                                # Boost for package-specific content
                                if (
                                    detected_package
                                    and detected_package.upper() in text_content
                                ):
                                    supp["score"] = supp.get("score", 0) * 1.5
                                text_results.insert(0, supp)

                    # Re-sort to put most relevant (containing target pins) at top
                    def relevance_score(chunk):
                        text = chunk.get("text", "").upper()
                        score = chunk.get("score", 0)
                        # Big boost for chunks containing our target pins
                        for pin in full_pins:
                            if pin in text:
                                score += 10
                        # Extra boost for package-specific content (BGA292_COM, etc.)
                        if detected_package and detected_package.upper() in text:
                            score += 8
                        # Boost for port function table content
                        for port_num in port_nums:
                            if (
                                f"PORT {port_num} FUNCTIONS" in text
                                or f"P{port_num}." in text
                            ):
                                score += 5
                        # Boost for EGTM_TOUT content
                        if "EGTM_TOUT" in text:
                            score += 3
                        return score

                    text_results.sort(key=relevance_score, reverse=True)
                    text_results = text_results[:text_top_k]

            # ------------------------------------------------------------------ #
            # Kit-manual routing: boost the correct board's manual, penalise     #
            # chunks from other boards' kit manuals.  Applied for all queries    #
            # (pinout and non-pinout) when board_info is available.              #
            # ------------------------------------------------------------------ #
            if board_info and board_info.get("kit_manual"):
                text_results = self._apply_kit_manual_routing(
                    text_results, board_info["kit_manual"]
                )
                text_results = text_results[:text_top_k]

            result["text_sources"] = text_results

            # Collect images associated with retrieved chunks
            if include_chunk_images and text_results:
                chunk_images = self._collect_images_from_chunks(
                    text_results, max_images=image_top_k
                )
                result["chunk_images"] = chunk_images

        # Retrieve images via CLIP similarity search with optional filtering
        if mode in ["hybrid", "image"] and self.clip_embedder:
            image_results = self.clip_embedder.search_images(
                question, top_k=image_top_k, metadata_filter=metadata_filter
            )
            result["image_sources"] = image_results

        # Merge and deduplicate image sources (chunk_images + CLIP results)
        all_images = self._merge_image_sources(
            result.get("chunk_images", []),
            result.get("image_sources", []),
            max_images=image_top_k * 2,  # Allow more images from combined sources
        )

        # Generate answer
        if not result["text_sources"] and not all_images:
            if metadata_filter:
                result["answer"] = (
                    f"No relevant information found matching your filter criteria. Try broadening your search or removing filters."
                )
            else:
                result["answer"] = (
                    "No relevant information found. Please ensure documents have been ingested."
                )
            return result

        # Use merged images for answer generation (pass migration and pinout context)
        logger.debug(
            f"Generating answer: {len(result['text_sources'])} text chunk(s), "
            f"{len(all_images)} image(s)"
        )
        result["answer"] = self._generate_answer(
            question,
            result["text_sources"],
            all_images,  # Use merged and deduplicated images
            is_migration=is_migration,
            migration_architectures=migration_architectures,
            is_pinout_query=is_pinout_query,
            board_info=board_info if is_pinout_query else None,
        )

        # Final safety net: if generation returned None / empty but we do have
        # retrieved chunks, expose those chunks directly so the caller always
        # receives useful content.
        if not result["answer"] and result["text_sources"]:
            logger.warning(
                "_generate_answer returned empty — building raw context fallback "
                f"from {len(result['text_sources'])} retrieved chunk(s)."
            )
            raw_parts = [
                f"[{s.get('source', 'unknown')} p.{s.get('page', '?')}]\n{s.get('text', '')}"
                for s in result["text_sources"][:8]
            ]
            result["answer"] = "\n\n---\n\n".join(raw_parts)

        # Store merged images in result
        result["all_images"] = all_images

        # Extract source PDF information
        result["source_pdfs"] = self._extract_source_pdfs(
            result["text_sources"], all_images
        )

        return result

    def get_available_sources(self) -> Dict[str, Any]:
        """
        Get available sources for metadata filtering.

        Returns:
            Dict with 'text_sources' and 'image_sources' lists,
            plus page range info per source
        """
        result = {
            "text_sources": self.vector_store.get_available_sources(),
            "image_sources": [],
            "source_details": {},
        }

        # Get image sources if CLIP is available
        if self.clip_embedder:
            result["image_sources"] = self.clip_embedder.get_available_sources()

        # Get page range for each text source
        for source in result["text_sources"]:
            min_page, max_page = self.vector_store.get_page_range(source)
            result["source_details"][source] = {
                "page_min": min_page,
                "page_max": max_page,
            }

        return result

    def _merge_image_sources(
        self,
        chunk_images: List[Dict[str, Any]],
        clip_images: List[Dict[str, Any]],
        max_images: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate images from chunk associations and CLIP search.
        Prioritizes chunk-associated images (more contextually relevant).

        Args:
            chunk_images: Images associated with retrieved text chunks
            clip_images: Images from CLIP similarity search
            max_images: Maximum number of images to return

        Returns:
            Merged and deduplicated list of images
        """
        # Strict performance limits
        max_images = min(max_images, 8)  # Hard cap for memory

        seen_paths = set()
        merged = []

        # First, add chunk-associated images (higher priority - contextual relevance)
        # Limit chunk images processed
        for img in chunk_images[: min(len(chunk_images), 5)]:
            img_path = img.get("file_path", "")
            if img_path and img_path not in seen_paths:
                seen_paths.add(img_path)
                img["source_type"] = "chunk_associated"
                merged.append(img)
                if len(merged) >= max_images:
                    return merged

        # Then, add CLIP results (semantic similarity)
        # Limit CLIP images processed
        for img in clip_images[: min(len(clip_images), 5)]:
            img_path = img.get("file_path", "")
            if img_path and img_path not in seen_paths:
                seen_paths.add(img_path)
                img["source_type"] = "clip_similarity"
                merged.append(img)
                if len(merged) >= max_images:
                    return merged

        return merged

    def _extract_source_pdfs(
        self, text_sources: List[Dict[str, Any]], image_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract unique source PDF information from text and image sources with page references."""
        source_map: Dict[str, Dict[str, Any]] = {}

        # From text sources
        for source in text_sources:
            source_path = source.get("source_path", "")
            if not source_path:
                continue
            entry = source_map.setdefault(
                source_path,
                {
                    "filename": source.get("source", ""),
                    "filepath": source_path,
                    "type": "text_source",
                    "pages": set(),
                },
            )
            page = source.get("page")
            if isinstance(page, int) and page > 0:
                entry["pages"].add(page)

        # From image sources
        for source in image_sources:
            source_path = source.get("source_path", "")
            if not source_path:
                continue
            entry = source_map.setdefault(
                source_path,
                {
                    "filename": source.get("source_pdf", ""),
                    "filepath": source_path,
                    "type": "image_source",
                    "pages": set(),
                },
            )
            page = source.get("page")
            if isinstance(page, int) and page > 0:
                entry["pages"].add(page)

        # Normalize pages and build final list
        source_pdfs = []
        for entry in source_map.values():
            pages = sorted(entry.get("pages", []))
            entry["pages"] = pages
            if pages:
                entry["page_range"] = (
                    f"{pages[0]}-{pages[-1]}"
                    if pages[0] != pages[-1]
                    else f"{pages[0]}"
                )
            else:
                entry["page_range"] = None
            source_pdfs.append(entry)

        return source_pdfs

    def _generate_answer(
        self,
        question: str,
        text_sources: List[Dict[str, Any]],
        image_sources: List[Dict[str, Any]],
        is_migration: bool = False,
        migration_architectures: Optional[List[str]] = None,
        is_pinout_query: bool = False,
        board_info: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate an answer using retrieved context.

        Args:
            question: User's question
            text_sources: Retrieved text chunks
            image_sources: Retrieved images
            is_migration: Whether this is a migration query
            migration_architectures: List of architectures involved in migration
            is_pinout_query: Whether this is a pin/pinout verification query
            board_info: Optional board information including package type
        """
        if not self.openai_client:
            # Fallback: return raw context
            context = "\n\n---\n\n".join([s.get("text", "") for s in text_sources[:5]])
            return f"Context found (no OpenAI API for generation):\n\n{context}"

        # Fast path: text-only (no images) uses simpler API call
        if not image_sources:
            return self._generate_text_only_answer(
                question,
                text_sources,
                is_migration=is_migration,
                migration_architectures=migration_architectures,
                is_pinout_query=is_pinout_query,
                board_info=board_info,
            )

        import base64

        # Build context - limit text sources for memory efficiency
        text_context = ""
        if text_sources:
            # Limit to first 8 sources and truncate long text
            limited_sources = text_sources[:8]
            text_parts = []
            for s in limited_sources:
                source_text = s.get("text", "")[:1500]  # Truncate to 1500 chars max
                text_parts.append(
                    f"[Source: {s.get('source', 'unknown')}, Page {s.get('page', '?')}]\n{source_text}"
                )
            text_context = "\n\n---\n\n".join(text_parts)

        # Build message content
        content = []

        # Add text context
        if text_context:
            content.append(
                {"type": "text", "text": f"## Text Context:\n\n{text_context}\n\n---\n"}
            )

        # Add images (strict limits for performance)
        images_added = 0
        max_images = min(self.config.max_images_in_answer, 5)  # Hard cap at 5 images
        for img in image_sources[:max_images]:
            img_path = img.get("file_path", "")
            if not os.path.exists(img_path):
                continue

            # Skip large images to save memory
            try:
                img_size = os.path.getsize(img_path)
                if img_size > 5 * 1024 * 1024:  # Skip images larger than 5MB
                    logger.debug(f"Skipping large image {img_path}: {img_size} bytes")
                    continue
            except:
                continue

            try:
                with open(img_path, "rb") as f:
                    b64_image = base64.b64encode(f.read()).decode("utf-8")

                # Include GPT description if available (faster than re-analyzing)
                gpt_desc = img.get("gpt_description", "")
                desc_text = (
                    f" - {gpt_desc[:100]}" if gpt_desc else ""
                )  # Truncate description

                content.append(
                    {
                        "type": "text",
                        "text": f"[Image: {img.get('filename', 'unknown')} from {img.get('source_pdf', 'unknown')}{desc_text}]",
                    }
                )

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": "low",  # Use low detail for faster processing
                        },
                    }
                )
                images_added += 1
            except Exception as e:
                logger.warning(f"Failed to include image {img_path}: {e}")
                continue

            # Memory break: if too many images, stop
            if images_added >= 3:  # Even more restrictive cap
                break

        # Add board/package context if available (for pinout queries)
        if board_info and is_pinout_query:
            board_context = (
                f"\n\n## Board Information (from proj_package_mapping.json):\n"
            )
            board_context += f"- Board Model: {board_info['board']}\n"
            board_context += f"- Package Type: {board_info['package']}\n"
            board_context += f"- Architecture: {board_info['architecture']}\n"
            if board_info.get("notes"):
                board_context += f"- Notes: {board_info['notes']}\n"
            content.append({"type": "text", "text": board_context})

        # Add question with migration context if applicable
        if is_migration and migration_architectures:
            arch_str = " → ".join(migration_architectures)
            content.append(
                {
                    "type": "text",
                    "text": f"\n\n## Migration Query ({arch_str}):\n{question}\n\nProvide detailed migration guidance based on the context and images above. Follow the structured migration approach.",
                }
            )
        else:
            content.append(
                {
                    "type": "text",
                    "text": f"\n\n## Question:\n{question}\n\nProvide a comprehensive answer based on the context and images above.",
                }
            )

        # Select appropriate system prompt
        if is_pinout_query:
            system_prompt = PINOUT_SYSTEM_PROMPT
        elif is_migration:
            system_prompt = MIGRATION_SYSTEM_PROMPT
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        try:
            # Build Responses API input: input_text / input_image blocks must be
            # nested inside a "message" item — they are NOT valid top-level types.
            content_blocks = [
                {"type": "input_text", "text": b["text"]}
                if b["type"] == "text"
                else {"type": "input_image", "image_url": b["image_url"]["url"]}
                for b in content
            ]
            response = self.openai_client.responses.create(
                model=self.config.vlm_model,
                instructions=system_prompt,
                input=[
                    {"type": "message", "role": "user", "content": content_blocks}
                ],
                max_output_tokens=(
                    self.config.max_answer_tokens if not is_migration else 8000
                ),
                reasoning={"effort": "medium"},
            )

            answer = response.output_text or ""
            # Reasoning models (o-series / gpt-5) can return None / empty content
            # when all budget is consumed by reasoning tokens.  Fall back to raw
            # context so the caller always receives something useful.
            if not answer and text_context:
                logger.warning(
                    "LLM returned empty content (reasoning model / token budget) — "
                    "returning raw context chunks as fallback."
                )
                return f"[Retrieved context — generation unavailable]\n\n{text_context[:3000]}"
            return answer

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return (
                f"Error generating answer: {e}\n\nContext found:\n{text_context[:1000]}"
            )

    def _generate_text_only_answer(
        self,
        question: str,
        text_sources: List[Dict[str, Any]],
        is_migration: bool = False,
        migration_architectures: Optional[List[str]] = None,
        is_pinout_query: bool = False,
        board_info: Optional[Dict[str, str]] = None,
    ) -> str:
        """Fast text-only answer generation.

        Args:
            question: User's question
            text_sources: Retrieved text chunks
            is_migration: Whether this is a migration query
            migration_architectures: List of architectures involved in migration
            is_pinout_query: Whether this is a pin/pinout verification query
            board_info: Optional board information including package type
        """
        # Separate project readme sources from other sources
        # Project readme sources contain implementation-specific pin assignments that should be prioritized
        project_sources = []
        datasheet_sources = []

        for s in text_sources[: self.config.text_top_k]:
            source_name = s.get("source", "").lower()
            doc_type = s.get("document_type", "")

            # Check if this is a project readme source
            if (
                doc_type == "AURIX_projects"
                or "_readme" in source_name
                or "project_name" in s
            ):
                project_sources.append(s)
            else:
                datasheet_sources.append(s)

        # Build context with project sources first (highest priority)
        context_parts = []

        if project_sources:
            context_parts.append(
                "=== PROJECT IMPLEMENTATION REFERENCE (HIGHEST PRIORITY - USE THESE PIN ASSIGNMENTS) ==="
            )
            for s in project_sources:
                project_name = s.get("project_name", s.get("source", "unknown"))
                context_parts.append(f"[PROJECT: {project_name}]\n{s.get('text', '')}")
            context_parts.append("=== END PROJECT REFERENCE ===\n")

        if datasheet_sources:
            context_parts.append("=== DATASHEET/DOCUMENTATION REFERENCE ===")
            for s in datasheet_sources:
                context_parts.append(
                    f"[Source: {s.get('source', 'unknown')}, Page {s.get('page', '?')}]\n{s.get('text', '')}"
                )

        text_context = "\n\n---\n\n".join(context_parts)

        # Select appropriate system prompt
        # Add guidance to prioritize project sources when available
        project_priority_note = ""
        if project_sources:
            project_priority_note = "\n\nIMPORTANT: When PROJECT IMPLEMENTATION REFERENCE sources are available, prioritize their pin assignments and configurations over generic datasheet information. Project readmes contain tested, working implementations for specific boards."

        # Add board/package context if available
        if board_info and is_pinout_query:
            board_context = (
                f"\n\n## Board Information (from proj_package_mapping.json):\n"
            )
            board_context += f"- Board Model: {board_info['board']}\n"
            board_context += f"- Package Type: {board_info['package']}\n"
            board_context += f"- Architecture: {board_info['architecture']}\n"
            if board_info.get("notes"):
                board_context += f"- Notes: {board_info['notes']}\n"
            board_context += "\nIMPORTANT: Use this package type to find the correct pin function tables in the datasheet.\n"
            board_context += "Different packages have different pin availability. Reference tables should specify this package.\n"
            text_context = board_context + "\n\n" + text_context

        if is_pinout_query:
            system_prompt = PINOUT_SYSTEM_PROMPT + project_priority_note
            user_prompt = f"""Context:\n{text_context}\n\n---\n\nPin/Pinout Verification Query: {question}\n\nVerify the pin configuration by checking the datasheet port function tables. List all alternate functions for each pin mentioned and provide a clear verdict on whether the claimed configuration is valid."""
        elif is_migration:
            system_prompt = MIGRATION_SYSTEM_PROMPT + project_priority_note
            arch_str = (
                " → ".join(migration_architectures)
                if migration_architectures
                else "unknown"
            )
            user_prompt = f"""Context:\n{text_context}\n\n---\n\nMigration Query ({arch_str}): {question}\n\nProvide detailed migration guidance following the structured migration approach. Include comparison tables, code examples, and verification steps."""
        else:
            system_prompt = f"""You are an expert technical assistant for Infineon AURIX microcontroller documentation.
Answer based only on the provided context. Be concise and accurate.{project_priority_note}"""
            user_prompt = f"""Context:\n{text_context}\n\n---\n\nQuestion: {question}\n\nProvide a comprehensive answer."""

        try:
            response = self.openai_client.responses.create(
                model=self.config.vlm_model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=self.config.max_answer_tokens if not is_migration else 8000,
                reasoning={"effort": "medium"},
            )
            answer = response.output_text or ""
            # Reasoning models (o-series / gpt-5) can return None / empty content
            # when reasoning tokens exhaust the budget.  Fall back to raw context.
            if not answer and text_context:
                logger.warning(
                    "LLM returned empty content (reasoning model / token budget) — "
                    "returning raw context chunks as fallback."
                )
                return f"[Retrieved context — generation unavailable]\n\n{text_context[:3000]}"
            return answer
        except Exception as e:
            logger.error(f"Text-only answer generation failed: {e}")
            return f"Error: {e}\n\nContext:\n{text_context[:1000]}"

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            "documents_processed": len(self.processed_docs),
            "total_text_chunks": len(self.vector_store.texts),
            "total_images_indexed": 0,
            "config": {
                "chunk_size": self.config.chunk_size,
                "enable_image_extraction": self.config.enable_image_extraction,
                "enable_clip": self.config.enable_clip_embeddings
            },
        }

        if self.clip_embedder and self.clip_embedder.collection:
            stats["total_images_indexed"] = self.clip_embedder.collection.count()

        return stats

    def query_interactive(self, question: str) -> dict:
        """Process a question using the exact same logic as interactive_mode.

        This is the single source of truth for query processing.  Both the
        CLI interactive loop and the web-app chatbot chain should call this
        method so that architecture detection, pinout routing, kit-manual
        injection, and system-prompt selection are always identical.

        Args:
            question: The natural-language question to answer.

        Returns:
            The raw result dict returned by ``self.query()`` (keys: ``answer``,
            ``text_sources``, ``image_sources``, plus any extra keys the query
            method adds) augmented with diagnostic fields:

            - ``detected_architectures`` – list of detected arch strings
            - ``is_migration``           – bool
            - ``is_pinout_query``        – bool
            - ``board_info``             – board-info dict or None
        """
        _arch_key_map = {
            "TC4xx": "AURIX TC4xx",
            "TC39x": "TC39x",
            "TC38x": "TC38x",
            "TC37x": "TC37x",
            "TC36x": "TC36x",
            "TC35x": "TC35x",
            "TC33x": "TC33x-TC32x",
            "TC32x": "TC33x-TC32x",
        }

        # Step 1: migration + architecture detection
        is_migration, detected_archs = self.folder_manager.detect_migration_query(question)
        detected_folders, detected_archs = self.folder_manager.detect_architecture_from_query(question)

        # Step 2: board / MCU detection (covers exact kit names AND bare MCU names)
        board_info = self._extract_board_info(question)

        # Step 3: pinout detection (must happen before arch-augmentation)
        is_pinout_query = self.folder_manager.detect_pinout_query(question)

        # Step 4: for non-pinout queries augment detected_archs from board_info
        if board_info and not is_pinout_query:
            raw_arch = board_info.get("architecture", "")
            folder_arch = _arch_key_map.get(raw_arch, raw_arch)
            if not detected_archs:
                detected_folders = self.folder_manager.get_folders_for_architecture(folder_arch)
                detected_archs = [folder_arch]
            elif folder_arch not in detected_archs:
                detected_archs.append(folder_arch)

        # Step 5: datasheet sources for pinout queries
        datasheet_sources: list = []
        if is_pinout_query and detected_archs:
            datasheet_sources = self.folder_manager.get_datasheet_sources_for_architecture(detected_archs)

        # Step 6: inject the board's kit_manual at the front so it wins over
        #          manuals from other boards
        if board_info and board_info.get("kit_manual"):
            kit_manual = board_info["kit_manual"]
            if kit_manual not in datasheet_sources:
                datasheet_sources = [kit_manual] + datasheet_sources

        # Step 7: run the actual query
        result = self.query(
            question,
            target_folders=detected_folders if detected_folders else None,
            detected_architectures=detected_archs if detected_archs else None,
            is_migration=is_migration,
            migration_architectures=detected_archs if is_migration else None,
            is_pinout_query=is_pinout_query,
            datasheet_sources=datasheet_sources if datasheet_sources else None,
        )

        # Attach diagnostic metadata to the result
        if isinstance(result, dict):
            result.setdefault("detected_architectures", detected_archs)
            result.setdefault("is_migration", is_migration)
            result.setdefault("is_pinout_query", is_pinout_query)
            result.setdefault("board_info", board_info)

        return result

    def interactive_mode(self):
        """Run interactive query mode."""
        print("\n" + "=" * 60)
        print("  Infineon RAG Agent - Interactive Mode")
        print("=" * 60)
        print("\nCommands:")
        print("  /ingest <folder>  - Ingest documents from folder")
        print("  /stats            - Show statistics")
        print("  /quit             - Exit")
        print("  <question>        - Ask a question")
        print("-" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "/quit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "/stats":
                    stats = self.get_stats()
                    print(f"\nStats: {json.dumps(stats, indent=2)}\n")
                    continue

                if user_input.lower().startswith("/ingest"):
                    parts = user_input.split(maxsplit=1)
                    folder = parts[1] if len(parts) > 1 else "kit_manual"
                    print(f"\nIngesting documents from {folder}...")
                    try:
                        stats = self.ingest_documents(folder)
                        print(
                            f"Done! Processed {stats['documents_processed']} documents.\n"
                        )
                    except Exception as e:
                        print(f"Error: {e}\n")
                    continue

                # Delegate all query processing to query_interactive so that
                # the interactive CLI and the web-app chatbot chain always use
                # identical architecture detection / routing logic.
                print("\nSearching...")
                result = self.query_interactive(user_input)

                detected_archs = result.get("detected_architectures", [])
                board_info     = result.get("board_info")

                print(f"\nAnswer:\n{result['answer']}\n")

                if detected_archs:
                    print(f"Architecture: {', '.join(detected_archs)}")
                if board_info:
                    board_label = board_info["board"]
                    if board_info.get("detected_mcu"):
                        board_label = f"{board_info['detected_mcu']} -> {board_label}"
                    print(f"Board/MCU:    {board_label} (kit: {board_info.get('kit_manual', 'N/A')})")
                if result.get("text_sources"):
                    print(f"Sources: {len(result['text_sources'])} text chunks")
                if result.get("image_sources"):
                    print(f"         {len(result['image_sources'])} images")
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """CLI entry point."""
    import argparse

    # Fix Windows console encoding for emojis
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass  # Fallback for older Python or non-console environments

    parser = argparse.ArgumentParser(
        description="Infineon RAG Agent - Document Processing and Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available document folders and architectures
    python infineon_rag_agent.py list-folders
    python infineon_rag_agent.py list-folders --show-files
    
    # Ingest from specific architecture folder (includes parent family docs)
    python infineon_rag_agent.py ingest --architecture TC38x
    python infineon_rag_agent.py ingest --architecture TC4xx
    
    # Ingest ALL PDFs from entire documents directory (recursively)
    python infineon_rag_agent.py ingest
    
    # Ingest from specific subfolder (recursively)
    python infineon_rag_agent.py ingest kit_manual
    
    # Ingest from absolute path (recursively)
    python infineon_rag_agent.py ingest "C:/Users/Documents/Infineon"
    
    # Ingest without GPT Vision analysis
    python infineon_rag_agent.py ingest --no-vision
    
    # Force re-processing of all documents (ignore cache)
    python infineon_rag_agent.py ingest --force
    
    # Ingest single PDF file
    python infineon_rag_agent.py ongest "/path/to/document.pdf"
    
    # Interactive mode
    python infineon_rag_agent.py interactive
    
    # Show statistics
    python infineon_rag_agent.py stats
    
    # List available sources for filtering
    python infineon_rag_agent.py sources
""",
    )

    parser.add_argument(
        "action",
        choices=[
            "ingest",
            "ongest",
            "interactive",
            "stats",
            "sources",
            "list-folders",
            "ingest-projects",
        ],
        help="Action to perform",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help="Folder to ingest (default: all PDFs in documents directory), file path for ongest, or query string",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of already processed documents",
    )

    # Architecture selection options
    parser.add_argument(
        "--architecture",
        "-a",
        type=str,
        default=None,
        help="Target architecture for ingestion (e.g., TC38x, TC37x, TC4xx).",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Show PDF files in folder listing (for list-folders action)",
    )

    args = parser.parse_args()

    # Create config
    config = RAGAgentConfig(
        force_reprocess=args.force,
    )

    # Initialize folder manager for architecture selection
    folder_manager = DocumentFolderManager(config.documents_dir)

    # Handle list-folders action first (doesn't need agent)
    if args.action == "list-folders":
        print(folder_manager.list_folder_structure(show_files=args.show_files))

        # Also show available architectures
        architectures = folder_manager.get_available_architectures()
        print("\n📋 Available Architectures for --architecture flag:")
        print("-" * 50)
        for arch in sorted(architectures, key=lambda x: (x["family"], x["name"])):
            print(f"  {arch['name']:15} ({arch['family']}, {arch['pdf_count']} PDFs)")

        print("\n💡 Usage Examples:")
        print("  python infineon_rag_agent.py ingest --architecture TC38x")
        print("  python infineon_rag_agent.py interactive")
        return

    # Initialize agent
    agent = InfineonRAGAgent(config)

    if args.action == "ingest":
        # Handle architecture-based ingestion
        if args.architecture:
            folders = folder_manager.get_folders_for_architecture(args.architecture)
            if folders:
                print(f"🎯 Target architecture: {args.architecture}")
                print(f"📂 Folders to ingest:")
                for folder in folders:
                    print(f"   - {folder}")
                print()

                # Ingest from each folder
                all_stats = {
                    "input_path": str(config.documents_dir),
                    "architecture": args.architecture,
                    "folders_processed": [],
                    "files_found": 0,
                    "documents_processed": 0,
                    "documents_skipped": 0,
                    "total_chunks": 0,
                    "total_images": 0,
                    "chunks_with_images": 0,
                }

                for folder in folders:
                    print(f"Processing: {folder.name}...")
                    try:
                        stats = agent.ingest_documents(str(folder))
                        all_stats["folders_processed"].append(str(folder))
                        all_stats["files_found"] += stats.get("files_found", 0)
                        all_stats["documents_processed"] += stats.get(
                            "documents_processed", 0
                        )
                        all_stats["documents_skipped"] += stats.get(
                            "documents_skipped", 0
                        )
                        all_stats["total_chunks"] += stats.get("total_chunks", 0)
                        all_stats["total_images"] += stats.get("total_images", 0)
                        all_stats["chunks_with_images"] += stats.get(
                            "chunks_with_images", 0
                        )
                    except FileNotFoundError:
                        print(f"  Warning: No PDFs found in {folder}")

                print(f"\n✅ Ingestion complete for {args.architecture}!")
                print(f"  Folders Processed: {len(all_stats['folders_processed'])}")
                print(f"  PDFs Found: {all_stats['files_found']}")
                print(f"  Documents Processed: {all_stats['documents_processed']}")
                print(f"  Documents Skipped: {all_stats['documents_skipped']}")
                print(f"  Chunks: {all_stats['total_chunks']}")
                print(f"  Images: {all_stats['total_images']}")
            else:
                print(f"❌ Architecture '{args.architecture}' not found.")
                print("Use 'list-folders' to see available architectures.")
        else:
            print(f"Ingesting documents from: {args.input}")
            print(f"Searching recursively for all PDF files...")
            if args.force:
                print("Force mode: Re-processing all documents\n")
            else:
                print("Smart deduplication: Skipping already processed documents\n")
            stats = agent.ingest_documents(args.input)
            print(f"\nIngestion complete!")
            print(f"  Path: {stats['input_path']}")
            print(f"  PDFs Found: {stats['files_found']}")
            print(f"  Documents Processed: {stats['documents_processed']}")
            print(f"  Documents Skipped: {stats['documents_skipped']}")
            print(f"  Chunks: {stats['total_chunks']}")
            print(f"  Images: {stats['total_images']}")
            print(f"  Chunks with Images: {stats.get('chunks_with_images', 0)}")

    elif args.action == "ongest":
        if args.input == ".":
            print("Please provide a PDF file path.")
            return

        print(f"Ingesting single file: {args.input}")
        try:
            stats = agent.ingest_single_file(args.input)
            print(f"\nIngestion complete!")
            print(f"  File: {Path(args.input).name}")
            print(f"  Chunks: {stats['total_chunks']}")
            print(f"  Images: {stats['total_images']}")
        except Exception as e:
            print(f"Error: {e}")

    elif args.action == "ingest-projects":
        print("📚 Ingesting project README files...")
        print("=" * 50)
        if args.force:
            print("Force mode: Re-ingesting all projects\n")
        else:
            print("Duplicate detection: Skipping already ingested projects\n")

        try:
            stats = agent.ingest_project_readmes(force=args.force)

            print(f"\n✅ Project ingestion complete!")
            print(f"  Folders Scanned: {len(stats['folders_scanned'])}")
            for folder in stats["folders_scanned"]:
                print(f"    - {folder}")
            print(f"  Projects Found: {stats['projects_found']}")
            print(f"  Projects Ingested: {stats['projects_ingested']}")
            print(f"  Projects Skipped (already ingested): {stats['projects_skipped']}")
            print(f"  Total Chunks Created: {stats['total_chunks']}")

            if stats["skipped_projects"]:
                print(f"\n⏭️ Skipped Projects ({len(stats['skipped_projects'])}):")
                for proj in stats["skipped_projects"][:10]:  # Show first 10
                    print(f"    - {proj}")
                if len(stats["skipped_projects"]) > 10:
                    print(f"    ... and {len(stats['skipped_projects']) - 10} more")

            if stats["ingested_projects"]:
                print(
                    f"\n📥 Newly Ingested Projects ({len(stats['ingested_projects'])}):"
                )
                for proj in stats["ingested_projects"][:10]:  # Show first 10
                    print(f"    - {proj}")
                if len(stats["ingested_projects"]) > 10:
                    print(f"    ... and {len(stats['ingested_projects']) - 10} more")

            if stats["errors"]:
                print(f"\n⚠️ Errors ({len(stats['errors'])}):")
                for err in stats["errors"][:5]:  # Show first 5
                    print(f"    - {err}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    elif args.action == "sources":
        # List available sources for filtering
        sources = agent.get_available_sources()
        print("\nAvailable Sources for Filtering:")
        print("=" * 50)

        if sources["text_sources"]:
            print("\nText Sources:")
            for source in sources["text_sources"]:
                details = sources["source_details"].get(source, {})
                page_info = f"pages {details.get('page_min', '?')}-{details.get('page_max', '?')}"
                print(f"  - {source} ({page_info})")
        else:
            print("\nNo text sources found. Run 'ingest' first.")

        if sources["image_sources"]:
            print("\nImage Sources:")
            for source in sources["image_sources"]:
                print(f"  - {source}")

    elif args.action == "interactive":
        agent.interactive_mode()

    elif args.action == "stats":
        stats = agent.get_stats()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
