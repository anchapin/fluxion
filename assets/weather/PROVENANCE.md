# `assets/weather/` provenance chain

Provenance record for the ASHRAE Standard 140 Section 6 (Weather Drivers Comparative
Tests) weather files committed under `assets/weather/`. Machine-readable companion:
[`assets/weather/provenance-manifest.json`](provenance-manifest.json). Verification
helper: `scripts/verify_weather_provenance.py`. Issue: #4284 (re-source §6 weather
files and define a real provenance chain). Prior tracking issue for the redistribution
question: #4285.

**Status (2026-10-28): publisher provenance VERIFIED for all twelve §6 WD files.**
Each committed file is byte-identical to the corresponding file in the publisher's
own published package, confirmed by SHA-256 and independently by the CRC-32 values
recorded in that package's ZIP central directory. No forensic reconstruction of
origin was required.

---

## 1. Canonical publisher source

| Item | Value |
|---|---|
| Publisher | ASHRAE (American Society of Heating, Refrigerating and Air-Conditioning Engineers) |
| Landing page | <https://data.ashrae.org/standard140/> |
| Accompanying-files index | <https://data.ashrae.org/standard140/accompany.html> |
| Section 6 file manifest | <https://data.ashrae.org/standard140/AccompanyingSection6.html> |
| Download (bookstore landing) | <https://www.ashrae.org/technical-resources/bookstore/supplemental-files/ansi-ashrae-standard-140-2023-method-of-test-for-evaluating-building-performance-simulation-software> |
| Direct package URL | `https://xp20.ashrae.org/supplementalfiles/140-2023-AccompanyingFiles.zip` |
| Edition | ANSI/ASHRAE Standard 140-2023 |
| §6 package folder | `Accompanying FIles/Std140_WD_Files/Normative Materials/` |

`accompany.html` links the §6 manifest page and, at the bottom, the single
"Download 140-2023 Supplemental Files" link. That link points at the ASHRAE
bookstore page above, and the bookstore page's own download anchor points at the
direct package URL `https://xp20.ashrae.org/supplementalfiles/140-2023-AccompanyingFiles.zip`.
The upstream folder name carries a publisher typo (`Accompanying FIles`, capital
`I`); `scripts/fetch_ashrae140_weather.py` already documents and tolerates it.

### §6 manifest, as published

`AccompanyingSection6.html` states that the folder `\Std140_WD_Files\Normative
Materials` contains, for use with the Weather Drivers tests of Section 6:
`WD100.TMY3`, `WD100.EPW`, `WD200.TMY3`, `WD200.EPW`, `WD300.TMY3`, `WD300.EPW`,
`WD400.TMY3`, `WD400.EPW`, `WD500.TMY3`, `WD500.EPW`, `WD600.TMY3`, `WD600.EPW`,
plus `Std140_WD_Output.xlsx` and `S140outNotes.TXT` (Standard Output Report
templates, Normative Annex A2). Both formats are normative inputs (Normative
Annex A1 defines TMY3 and EPW), so the `.epw` / `.tmy3` pair per case is two
publisher files, not a local conversion artifact.

### How the WD set entered the standard

`accompany.html` states, verbatim:

> "Files added by Standard 140-2020 Addendum A are included in folder
> "Std140_WD_Files"."
>
> "The content of all other files did not change, other than to update headers to
> "140-2023" and update section numbering, or for other editorial revisions."

So the WD file set originates in 140-2020 Addendum A and is carried unchanged into
the 140-2023 package. A hash comparison against the 140-2023 package therefore
establishes that the committed bytes are the publisher's bytes; it cannot, and does
not need to, distinguish which publisher package the copy was taken from.

---

## 2. Current status of the committed files

| Property | Value |
|---|---|
| Present on disk | Yes — twelve §6 files (`WD100`…`WD600`, `.epw` + `.tmy3`) |
| Committed-byte SHA-256 | Recorded per file in `provenance-manifest.json` |
| Publisher SHA-256 | Recorded per file in `provenance-manifest.json` (verified 2026-10-28) |
| Publisher verification | **VERIFIED** — committed bytes == publisher bytes for all 12 |
| Redistribution terms | **UNCONFIRMED** — tracked in #4285; see §5 |
| Byte-for-byte re-encoding performed by this change | **None.** The committed weather files were not modified, moved, deleted, or re-encoded. |

The manifest keeps two distinct hash fields per file and never conflates them:

* `committed_sha256` — the SHA-256 of the bytes as committed in this repository.
* `publisher_sha256` — the SHA-256 of the same file as published by ASHRAE in
  `140-2023-AccompanyingFiles.zip`.

For all twelve files the two values are equal, because the committed bytes *are*
the publisher bytes. That equality is a result, not an assumption: it is re-checked
on every run of `scripts/verify_weather_provenance.py`, which exits non-zero if a
committed file drifts away from its recorded hash.

### Independent cross-check: publisher-recorded CRC-32

The publisher's package is a ZIP whose central directory stores a CRC-32 for every
member. Comparing those publisher-recorded CRC-32 values against CRC-32 computed
over the committed files is a second, independent confirmation that does not share
code or method with the SHA-256 comparison:

```
member         archive_crc32 committed_crc32  match
WD100.epw        3107358726     3107358726  True
WD100.tmy3       1066482081     1066482081  True
WD200.epw        3515214799     3515214799  True
WD200.tmy3       2154096959     2154096959  True
WD300.epw         551108323      551108323  True
WD300.tmy3       1542885735     1542885735  True
WD400.epw        2410423427     2410423427  True
WD400.tmy3       2467670483     2467670483  True
WD500.epw        1457898964     1457898964  True
WD500.tmy3        286692243      286692243  True
WD600.epw        1407422167     1407422167  True
WD600.tmy3       3934786700     3934786700  True
```

---

## 3. What this run verified, and how

All evidence below was produced on 2026-10-28 by this change, with no credentials,
no purchase, no login, and no personal account.

### 3.1 Publisher pages are freely reachable

```
$ curl -sS -o /dev/null -w "HTTP:%{http_code}\n" https://data.ashrae.org/standard140/accompany.html
HTTP:200
$ curl -sS -o /dev/null -w "HTTP:%{http_code}\n" https://data.ashrae.org/standard140/AccompanyingSection6.html
HTTP:200
```

### 3.2 The §6 supplemental package is freely downloadable (not paywalled)

The brief this change implements assumed the supplemental archive sat behind the
ASHRAE bookstore and would therefore be unreachable without a licensed copy. **That
assumption is false.** The bookstore page's download anchor resolves to a direct,
unauthenticated package URL:

```
$ curl -sSI https://xp20.ashrae.org/supplementalfiles/140-2023-AccompanyingFiles.zip | grep -i -E "HTTP/|content-length|last-modified|etag"
HTTP/1.1 200 OK
Content-Type: application/x-zip-compressed
Content-Length: 704030671
Last-Modified: Mon, 06 Nov 2023 15:26:01 GMT
ETag: "407de58cc510da1:0"
```

No `401`, no `403`, no paywall interstitial, no membership gate. The repository's
own CI already depends on this being free: `scripts/fetch_ashrae140_weather.py`
downloads the same URL at build time and its docstring records the same finding
("the ASHRAE packaging is distributed freely via ASHRAE's supplemental-files host —
no membership gate, no robots restriction observed on the direct package URL at
adoption time, 2026-10-15").

### 3.3 Archive hash recorded before extraction

```
$ sha256sum 140-2023-AccompanyingFiles.zip
98d1ba27bad54aa45ef81b065be36f56e121f53b8b662cbea3c45b583f4fe2bb
```

| Field | Value |
|---|---|
| Package URL | `https://xp20.ashrae.org/supplementalfiles/140-2023-AccompanyingFiles.zip` |
| Package size | 704,030,671 bytes |
| Package SHA-256 | `98d1ba27bad54aa45ef81b065be36f56e121f53b8b662cbea3c45b583f4fe2bb` |
| Package `Last-Modified` | Mon, 06 Nov 2023 15:26:01 GMT |
| Package ETag | `"407de58cc510da1:0"` |
| Retrieved (UTC) | 2026-10-28 |

### 3.4 Extraction happened outside the repository

The archive was expanded into a scratch directory **outside** the working tree, and
only hashes — never publisher bytes — were written into the repository. The
repository still contains no newly-added publisher payload.

### 3.5 Hash comparison

```
$ sha256sum "/tmp/ashrae-extract/Accompanying FIles/Std140_WD_Files/Normative Materials/"WD*.epw WD*.tmy3
a06b60a8ae89ba0db6a8c24d169cb3c9beb25a522fbb2513ec540f63e55cc014  WD100.epw
f3de148bae6f04b1c9c4782e6030b7faa9074724a2aff6cf8a687cc9b94b941d  WD200.epw
5e66bcaeaeae2cf999e34f8ef6cb0d56358f84f17d2771ae0a1351cb24ca52dc  WD300.epw
80127a0da8a771a7949706768f46195e608b2fbf145028bf37daa507722bfcae  WD400.epw
ed74f52f5388110624c25dcc49a9e74719321234288dfe40be1f69d6a00a927b  WD500.epw
8cace1999401a2b32da03631b3e85b3b3684fe1168e693dd4a45c590f91d8a2b  WD600.epw
c7fe4c6c7d5976105a3fd7264ce9f5e6d217875756552b7d312e166b7b700188  WD100.tmy3
6a3cd4ab3f3dcabf428dcc8b8c995985e764eea2148c6a4e55864e480f3b8f3f  WD200.tmy3
d3bbf2dd13d9182dfc14a3c28d1db1c3cc3656d4e525335b4e14f443f12a67be  WD300.tmy3
618fe828bbd9e84ffe17033f60b6fec04b1c282c11ce4f3fdb844051132b6961  WD400.tmy3
c4d2d6a5239619af600d7910155fc46f0c1703f70c359d7796b5a9ae05578a22  WD500.tmy3
4edb4a246863ff9d7becf631164a365b2e70d405ff352deb9649823fcc0bf71b  WD600.tmy3
```

These values are byte-for-byte identical to the committed files in
`assets/weather/` and to the `publisher_sha256` values already on record in
`data/reference/ashrae140/provenance.json` (recorded 2026-10-06 from a licensed
suite extract — see §4).

---

## 4. Relationship to the pre-existing chain record

`data/reference/ashrae140/provenance.json` already carried `publisher_sha256`
values for these twelve files, recorded 2026-10-06, with the commit message
"chore(ashrae140): refresh suite provenance record for the 2026-10-06
licensed-suite extract". `scripts/fetch_ashrae140.py verify` therefore already
passed 14/14 before this change.

Two corrections to the record are needed, and both are made by the new manifest
and this document rather than by editing the old record:

1. **The old record's `chain.archive_sha256` and `chain.retrieved_utc` were
   `null`.** The archive hash and retrieval date are now recorded (§3.3, and the
   `publisher_package` block of the new manifest), so the chain has an anchor at
   the package level, not only at the file level.
2. **`assets/weather/README.md` claimed "The files currently committed here have
   no recorded publisher hash … so `verify` fails until someone holding a licensed
   copy runs `record`."** That statement was stale as of 2026-10-06 and is
   corrected here: publisher hashes *were* recorded, and `verify` does not fail.

This change does **not** modify `data/reference/ashrae140/provenance.json`; that
file remains the record consumed by `scripts/fetch_ashrae140.py`. The new
manifest is the §6-focused, human- and machine-readable companion that adds the
archive-level anchor and the explicit committed-vs-publisher distinction.

---

## 5. What remains open: redistribution terms

Verified provenance is not a redistribution license. ASHRAE sells Standard 140,
and the terms under which the normative accompanying files may be redistributed in
a public repository are **not** confirmed by anything gathered here. The ASHRAE
bookstore page additionally carries an explicit intellectual-property notice
restricting the use of ASHRAE publication content in AI tooling.

Consequently:

* The committed §6 weather files remain in place; this change does not add,
  remove, or alter any of them.
* Whether the repository may continue to redistribute them is tracked in **#4285**
  and is deliberately **out of scope** for this change.
* No claim of redistribution permission is made anywhere in this document or in
  the manifest.

---

## 6. The record/verify workflow going forward

### 6.1 Routine verification (no publisher copy needed)

Re-hash the committed files and compare against the manifest. Exits non-zero on
any missing file, hash mismatch, or manifest row that does not correspond to a
committed file:

```
$ python3 scripts/verify_weather_provenance.py
```

Optional: also compare against a locally extracted publisher copy, which
re-verifies `publisher_sha256` end to end:

```
$ python3 scripts/verify_weather_provenance.py --publisher-dir "<extracted>/Std140_WD_Files/Normative Materials"
```

### 6.2 Refreshing the publisher hashes (when the publisher package changes)

1. Download the package to a scratch location **outside** the repository and
   record its own SHA-256 *before* extracting:

   ```
   curl -o "$HOME/std140/140-2023-AccompanyingFiles.zip" \
     https://xp20.ashrae.org/supplementalfiles/140-2023-AccompanyingFiles.zip
   sha256sum "$HOME/std140/140-2023-AccompanyingFiles.zip"
   ```

2. Extract outside the repository:

   ```
   unzip -q "$HOME/std140/140-2023-AccompanyingFiles.zip" -d "$HOME/std140/x"
   ```

3. Record the publisher hashes. The repository's existing tool does this against
   the older chain record:

   ```
   python3 scripts/fetch_ashrae140.py record --archive "$HOME/std140/x"
   ```

   The equivalent for the new manifest is to update each `publisher_sha256` and
   `publisher_bytes` row in `assets/weather/provenance-manifest.json` from the
   extracted files, and to update the `publisher_package` block (URL, size,
   SHA-256, `Last-Modified`, ETag, retrieval date).

4. Re-verify:

   ```
   python3 scripts/verify_weather_provenance.py
   python3 scripts/fetch_ashrae140.py verify
   ```

### 6.3 Rules for anyone updating this record

* A committed-byte hash is never presented as a publisher hash. The manifest keeps
  `committed_sha256` and `publisher_sha256` in separate fields, and the
  verification script checks them independently.
* Publisher hashes are recorded only from bytes extracted from the publisher's own
  package, with the package's own SHA-256 recorded alongside.
* The archive is expanded outside the repository; publisher payloads are not
  committed.
* Any change to the recorded hashes carries a dated note explaining what moved and
  why.

---

## 7. Scope note: files not covered by this record

The other committed files in `assets/weather/` (`USA_*.epw`, `.clm`, `.ddy`,
`.pvsyst`, `.rain`, `.stat`, `.wea`) are EnergyPlus / climate.onebuilding
convention files from outside the ASHRAE 140 package. They are **not** part of any
Standard 140 section, they are not in the publisher package verified above, and
their download source is still not recorded in this repository. Their committed
SHA-256 values are listed in `assets/weather/CHECKSUMS.txt` for integrity only;
no provenance claim is made for them here.

The upstream provenance inside the §6 files themselves — visible in the EPW header
`COMMENTS 2` line, `"Downloaded from Climate.Onebuilding.Org"` — reflects the
NREL TMY data lineage that ASHRAE's package inherits. It does not contradict the
ASHRAE packaging: the files verified here are ASHRAE's published §6 files, whose
data content originates with NREL.

---

## 8. Evidence index

| Claim | Evidence |
|---|---|
| §6 manifest contents | `AccompanyingSection6.html`, HTTP 200, file list quoted in §1 |
| WD set entered via 140-2020 Addendum A | `accompany.html`, quoted verbatim in §1 |
| Package is free and unauthenticated | HTTP 200 + `Content-Length` + `Last-Modified` + ETag, §3.2 |
| Package identity | SHA-256 `98d1ba27…fe2bb`, §3.3 |
| Committed bytes == publisher bytes | SHA-256 comparison, §3.5; CRC-32 comparison, §2 |
| Committed files unmodified | `git status --porcelain` shows only new files; §"Verification" in `WORKER_REPORT.md` |
| Redistribution terms unresolved | Tracked in #4285, §5 |
