"""
Near-duplicate detection (MinHash + LSH) and n-gram overlap, in numpy/pyarrow only.

Used by scripts/make_heldout.py (held-out set D0, deduplicated against the
pretraining sample) and scripts/ngram_overlap.py (13-gram contamination check).

Pipeline for a batch of documents:
1. words: by default lowercase, replace every run of non-letter/non-digit chars with a
   space, split on whitespace (normalize=False: plain whitespace split). Done in pyarrow.
2. each distinct word -> 64-bit hash (blake2b, deterministic across processes/machines).
3. word n-grams -> 64-bit hash: polynomial combination of the word hashes (wrap-around
   uint64 arithmetic), finalized with the splitmix64 mixer. Only n-grams that lie fully
   inside one document are kept. Documents with fewer than n words have no n-grams.
4. MinHash: h_i(x) = ((a_i * x + b_i) mod 2^64) >> 32 with random odd a_i (multiply-shift
   hashing), signature[i] = min over the document's distinct shingles. Estimated Jaccard
   = fraction of equal signature entries.
5. LSH banding: `bands` bands of `rows` rows each (bands * rows <= num_perm). Two docs are
   candidates if any band matches exactly; P(candidate | Jaccard j) = 1 - (1 - j^rows)^bands.
   Candidates are then verified with the estimated Jaccard >= threshold.
"""

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Default parameters for D0 deduplication (documented in scripts/make_heldout.py and its report)
SHINGLE_N = 5          # word 5-gram shingles
NUM_PERM = 128         # MinHash permutations
LSH_BANDS = 16         # 16 bands x 8 rows = 128
LSH_ROWS = 8           # S-curve midpoint (1/16)^(1/8) = 0.707; P(candidate | J=0.8) = 0.947, | J=0.9) = 0.9999
JACCARD_THRESHOLD = 0.8
PREFILTER_MARGIN = 0.1

_U64 = np.uint64
_POLY = _U64(0x9E3779B97F4A7C15)  # odd multiplier for combining word hashes into n-gram hashes


def lsh_threshold(bands, rows):
    """Approximate Jaccard where the LSH S-curve crosses 1/2: (1/bands)^(1/rows)."""
    return (1.0 / bands) ** (1.0 / rows)


def lsh_collision_probability(jaccard, bands, rows):
    """Probability that two docs with this Jaccard similarity share at least one LSH band."""
    return 1.0 - (1.0 - jaccard ** rows) ** bands


def _splitmix64(z):
    with np.errstate(over="ignore"):
        z = z.copy()
        z ^= z >> _U64(30)
        z *= _U64(0xBF58476D1CE4E5B9)
        z ^= z >> _U64(27)
        z *= _U64(0x94D049BB133111EB)
        z ^= z >> _U64(31)
    return z


def _hash_strings(strings):
    """Deterministic 64-bit hashes of a list of str."""
    if not strings:
        return np.zeros(0, dtype=_U64)
    digests = b"".join(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest() for s in strings)
    return np.frombuffer(digests, dtype="<u8").astype(_U64)


def word_hashes(texts, normalize=True):
    """
    Tokenize documents into words and hash them.
    Returns (hashes: uint64[num_words], counts: int64[num_docs]) with the words of all
    documents concatenated in order.
    """
    if isinstance(texts, pa.ChunkedArray):
        arr = texts.combine_chunks()
    elif isinstance(texts, pa.Array):
        arr = texts
    else:
        arr = pa.array(list(texts), type=pa.large_string())
    num_docs = len(arr)
    if num_docs == 0:
        return np.zeros(0, dtype=_U64), np.zeros(0, dtype=np.int64)
    arr = pc.fill_null(arr, "")
    if normalize:
        arr = pc.utf8_lower(arr)
        arr = pc.replace_substring_regex(arr, pattern=r"[^\pL\pN]+", replacement=" ")
    arr = pc.utf8_trim_whitespace(arr)
    lists = pc.utf8_split_whitespace(arr)
    lengths = pc.list_value_length(lists).to_numpy(zero_copy_only=False).astype(np.int64)
    flat = pc.list_flatten(lists)
    doc_of = np.repeat(np.arange(num_docs, dtype=np.int64), lengths)
    keep = pc.utf8_length(flat).to_numpy(zero_copy_only=False) > 0  # empty docs split to [""]
    encoded = pc.dictionary_encode(flat)
    indices = encoded.indices.to_numpy(zero_copy_only=False)
    vocab_hashes = _hash_strings(encoded.dictionary.to_pylist())
    hashes = vocab_hashes[indices[keep]] if len(vocab_hashes) else np.zeros(0, dtype=_U64)
    counts = np.bincount(doc_of[keep], minlength=num_docs).astype(np.int64)
    return hashes, counts


def ngram_hashes(hashes, counts, n):
    """
    64-bit hashes of all word n-grams lying fully inside one document.
    Returns (ngram_hashes: uint64[k], doc_ids: int64[k]) ordered by document then position.
    """
    total = len(hashes)
    if total < n:
        return np.zeros(0, dtype=_U64), np.zeros(0, dtype=np.int64)
    doc_of = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    m = total - n + 1
    acc = np.zeros(m, dtype=_U64)
    with np.errstate(over="ignore"):
        for k in range(n):
            acc = acc * _POLY + hashes[k:k + m]
    acc = _splitmix64(acc + _U64(n))
    valid = doc_of[:m] == doc_of[n - 1:]
    return acc[valid], doc_of[:m][valid]


def unique_per_doc(hashes, doc_ids):
    """Distinct (doc_id, hash) pairs, sorted by doc_id then hash."""
    if len(hashes) == 0:
        return hashes, doc_ids
    order = np.lexsort((hashes, doc_ids))
    h, d = hashes[order], doc_ids[order]
    keep = np.ones(len(h), dtype=bool)
    keep[1:] = (h[1:] != h[:-1]) | (d[1:] != d[:-1])
    return h[keep], d[keep]


SHINGLE_BITS = 40  # shingle hashes are truncated to 40 bits so (doc_id, hash) packs into one uint64
_DOC_BITS = 64 - SHINGLE_BITS


def doc_shingles(texts, n=SHINGLE_N, normalize=True):
    """
    Distinct word-n-gram shingle hashes per document: (hashes, doc_ids, num_docs), sorted by
    doc_id then hash. Hashes are 40-bit (collisions are negligible at document scale) so that
    (doc_id, hash) packs into one uint64 and per-doc dedup is a single fast sort.
    """
    wh, counts = word_hashes(texts, normalize=normalize)
    num_docs = len(counts)
    assert num_docs < (1 << _DOC_BITS), "too many documents in one batch"
    h, d = ngram_hashes(wh, counts, n)
    keys = np.unique((d.astype(_U64) << _U64(SHINGLE_BITS)) | (h >> _U64(_DOC_BITS)))
    hashes = keys & _U64((1 << SHINGLE_BITS) - 1)
    doc_ids = (keys >> _U64(SHINGLE_BITS)).astype(np.int64)
    return hashes, doc_ids, num_docs


class MinHasher:
    """MinHash signatures with num_perm multiply-shift hash functions (deterministic per seed)."""

    def __init__(self, num_perm=NUM_PERM, seed=1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(0, 2**64, size=num_perm, dtype=np.uint64, endpoint=False) | _U64(1)
        self.b = rng.integers(0, 2**64, size=num_perm, dtype=np.uint64, endpoint=False)

    def signatures(self, hashes, doc_ids, num_docs, max_block_elems=1 << 23):
        """
        hashes/doc_ids sorted by doc_id (as returned by unique_per_doc / doc_shingles).
        Returns uint32[num_docs, num_perm]; docs without shingles are all 0xFFFFFFFF and must
        be excluded by the caller (see shingle counts).
        """
        sig = np.full((num_docs, self.num_perm), np.iinfo(np.uint32).max, dtype=np.uint32)
        if len(hashes) == 0:
            return sig
        docs_present, starts = np.unique(doc_ids, return_index=True)
        perm_block = int(max(1, min(self.num_perm, max_block_elems // len(hashes))))
        with np.errstate(over="ignore"):
            for p0 in range(0, self.num_perm, perm_block):
                p1 = min(self.num_perm, p0 + perm_block)
                values = (self.a[p0:p1, None] * hashes[None, :] + self.b[p0:p1, None]) >> _U64(32)
                mins = np.minimum.reduceat(values, starts, axis=1)
                sig[docs_present, p0:p1] = mins.T.astype(np.uint32)
        return sig


def estimated_jaccard(sig_a, sig_b):
    return float(np.mean(sig_a == sig_b))


class LSHIndex:
    """Banded LSH over MinHash signatures; maps band keys to item ids."""

    def __init__(self, bands=LSH_BANDS, rows=LSH_ROWS):
        self.bands, self.rows = bands, rows
        self.buckets = {}

    def _keys(self, sig):
        r = self.rows
        return [(b, sig[b * r:(b + 1) * r].tobytes()) for b in range(self.bands)]

    def add(self, item_id, sig):
        for key in self._keys(sig):
            self.buckets.setdefault(key, []).append(item_id)

    def query(self, sig):
        found = set()
        for key in self._keys(sig):
            found.update(self.buckets.get(key, ()))
        return found


def _sorted_member(values, sorted_set):
    """Boolean mask: values[i] in sorted_set (sorted unique uint64 array)."""
    if len(sorted_set) == 0 or len(values) == 0:
        return np.zeros(len(values), dtype=bool)
    pos = np.searchsorted(sorted_set, values)
    pos_clipped = np.minimum(pos, len(sorted_set) - 1)
    return (pos < len(sorted_set)) & (sorted_set[pos_clipped] == values)


class _BitSet:
    """Direct-addressed bitmap on the low `bits` bits of hashes: membership with false positives only."""

    def __init__(self, values, bits=28):
        self.mask = _U64((1 << bits) - 1)
        flags = np.zeros(1 << bits, dtype=bool)
        flags[(np.asarray(values, dtype=_U64) & self.mask).astype(np.int64)] = True
        self.packed = np.packbits(flags, bitorder="little")  # 2^bits / 8 bytes (32MB for 28 bits)

    def maybe_contains(self, values):
        idx = (values & self.mask).astype(np.int64)
        return ((self.packed[idx >> 3] >> (idx & 7).astype(np.uint8)) & 1).astype(bool)


class NearDupIndex:
    """
    MinHash/LSH index over a (small) reference set of documents, e.g. the held-out
    candidates. `query_texts` finds documents of a (large) stream that are near-duplicates
    (estimated Jaccard >= threshold over word n-gram shingles) of indexed documents.

    Prefilter (exact, cheap): a streamed doc A can only reach Jaccard t with an indexed doc B
    if |A ∩ B| >= t |A ∪ B| >= t |A|, so at least a fraction t of A's distinct shingles must
    occur in the union of all indexed shingles. Docs below (t - prefilter_margin) are
    skipped before computing their MinHash signature (the margin keeps docs whose MinHash
    estimate could exceed t). prefilter_margin=None disables the prefilter.
    """

    def __init__(self, texts, shingle_n=SHINGLE_N, num_perm=NUM_PERM, bands=LSH_BANDS, rows=LSH_ROWS,
                 threshold=JACCARD_THRESHOLD, seed=1, prefilter_margin=PREFILTER_MARGIN, normalize=True):
        assert bands * rows <= num_perm, "LSH needs bands * rows <= num_perm"
        self.shingle_n, self.threshold, self.normalize = shingle_n, threshold, normalize
        self.prefilter_margin = prefilter_margin
        self.hasher = MinHasher(num_perm=num_perm, seed=seed)
        h, d, n = doc_shingles(texts, n=shingle_n, normalize=normalize)
        self.num_docs = n
        self.shingle_counts = np.bincount(d, minlength=n)
        self.sigs = self.hasher.signatures(h, d, n)
        self.union = np.unique(h)
        self.union_bits = _BitSet(self.union)
        self.lsh = LSHIndex(bands, rows)
        for i in np.flatnonzero(self.shingle_counts > 0):
            self.lsh.add(int(i), self.sigs[i])

    def _verified(self, sig, exclude=None):
        out = []
        for j in sorted(self.lsh.query(sig)):
            if j == exclude:
                continue
            est = estimated_jaccard(sig, self.sigs[j])
            if est >= self.threshold:
                out.append((j, est))
        return out

    def internal_duplicates(self):
        """Pairs (i, j, est) with j < i where indexed doc i is a near-duplicate of earlier doc j."""
        pairs = []
        for i in np.flatnonzero(self.shingle_counts > 0):
            for j, est in self._verified(self.sigs[i], exclude=int(i)):
                if j < i:
                    pairs.append((int(i), int(j), est))
        return pairs

    def query_texts(self, texts):
        """
        Returns (matches, stats): matches is a list of (query_doc_idx, index_doc_idx, est_jaccard);
        stats has 'docs', 'candidates' (docs that passed the prefilter).
        """
        h, d, n = doc_shingles(texts, n=self.shingle_n, normalize=self.normalize)
        sizes = np.bincount(d, minlength=n)
        if self.prefilter_margin is None or len(h) == 0:
            cand = sizes > 0
        else:
            hit = self.union_bits.maybe_contains(h)  # cheap test, no false negatives
            hit[hit] = _sorted_member(h[hit], self.union)  # exact test on the few survivors
            hits = np.bincount(d[hit], minlength=n)
            cand = (sizes > 0) & (hits >= (self.threshold - self.prefilter_margin) * sizes)
        cand_ids = np.flatnonzero(cand)
        matches = []
        if len(cand_ids):
            sel = cand[d]
            remap = np.full(n, -1, dtype=np.int64)
            remap[cand_ids] = np.arange(len(cand_ids))
            sigs = self.hasher.signatures(h[sel], remap[d[sel]], len(cand_ids))
            for k, q in enumerate(cand_ids):
                for j, est in self._verified(sigs[k]):
                    matches.append((int(q), j, est))
        return matches, {"docs": n, "candidates": int(len(cand_ids))}


class NgramOverlap:
    """
    Set of hashed word n-grams of a reference corpus (e.g. the D1 validation split), and
    counting which of them occur in a streamed corpus (e.g. pretraining shards).
    """

    def __init__(self, texts, n=13, normalize=True):
        self.n, self.normalize = n, normalize
        wh, counts = word_hashes(texts, normalize=normalize)
        h, d = ngram_hashes(wh, counts, n)
        self.num_docs = len(counts)
        self.num_ngram_occurrences = len(h)
        self.occurrence_hashes = h           # with multiplicity, for occurrence-weighted overlap
        self.doc_hashes, self.doc_ids = unique_per_doc(h, d)
        self.unique = np.unique(h)
        self.unique_bits = _BitSet(self.unique)
        self.docs_with_ngrams = int(len(np.unique(self.doc_ids)))

    _member = staticmethod(_sorted_member)

    def scan(self, texts):
        """Return the sorted unique reference n-gram hashes that occur in `texts`."""
        wh, counts = word_hashes(texts, normalize=self.normalize)
        h, _ = ngram_hashes(wh, counts, self.n)
        h = h[self.unique_bits.maybe_contains(h)]
        return np.unique(h[_sorted_member(h, self.unique)])

    def summarize(self, matched):
        """Overlap statistics given the union of `scan` results."""
        matched = np.unique(np.asarray(matched, dtype=_U64))
        doc_hit = self._member(self.doc_hashes, matched)
        overlapping_docs = int(len(np.unique(self.doc_ids[doc_hit])))
        occ_hit = int(self._member(self.occurrence_hashes, matched).sum())
        num_unique = int(len(self.unique))
        return {
            "n": self.n,
            "normalize": self.normalize,
            "reference_docs": self.num_docs,
            "reference_docs_with_ngrams": self.docs_with_ngrams,
            "reference_unique_ngrams": num_unique,
            "reference_ngram_occurrences": int(self.num_ngram_occurrences),
            "overlapping_unique_ngrams": int(len(matched)),
            "frac_unique_ngrams_overlapping": len(matched) / num_unique if num_unique else 0.0,
            "frac_ngram_occurrences_overlapping": occ_hit / self.num_ngram_occurrences if self.num_ngram_occurrences else 0.0,
            "docs_with_overlap": overlapping_docs,
            "frac_docs_with_overlap": overlapping_docs / self.docs_with_ngrams if self.docs_with_ngrams else 0.0,
        }
