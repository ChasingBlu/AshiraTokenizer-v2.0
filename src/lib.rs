use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const VOCAB_SIZE: usize = 16_384;
pub const TOKEN_PAD: u16 = 0;
pub const TOKEN_UNK: u16 = 1;
pub const TOKEN_BOS: u16 = 2;
pub const TOKEN_EOS: u16 = 3;
pub const BYTE_TOKEN_START: u16 = 20;
pub const BPE_TOKEN_START: u16 = 276;
pub const WEIGHT_SCALE: i64 = 1000;

const SPECIAL_TOKEN_DEFS: [(&str, u16); 44] = [
    ("<PAD>", TOKEN_PAD),
    ("<UNK>", TOKEN_UNK),
    ("<BOS>", TOKEN_BOS),
    ("<EOS>", TOKEN_EOS),
    ("<KAREEM>", 4),
    ("<DYLAN_THINKING>", 5),
    ("<DYLAN_RESPONSE>", 6),
    ("<DYLAN_ADVERSARIAL>", 7),
    ("<BLU>", 8),
    ("<ECHO>", 9),
    ("<RESONANCE>", 10),
    ("<AI>", 11),
    ("<PHIL>", 12),
    ("<SYM>", 13),
    ("<REFLECTION>", 14),
    ("<CAIROS>", 15),
    ("[[ANCHOR]]", 16),
    ("[[CSA]]", 17),
    ("<science_doc>", 18),
    ("</KAREEM>", 4),
    ("</DYLAN_THINKING>", 5),
    ("</DYLAN_RESPONSE>", 6),
    ("</DYLAN_ADVERSARIAL>", 7),
    ("</BLU>", 8),
    ("</ECHO>", 9),
    ("</RESONANCE>", 10),
    ("</AI>", 11),
    ("</PHIL>", 12),
    ("</SYM>", 13),
    ("</REFLECTION>", 14),
    ("</CAIROS>", 15),
    ("[[/ANCHOR]]", 16),
    ("[[/CSA]]", 17),
    ("</science_doc>", 18),
    ("<kareem_response>", 4),
    ("</kareem_response>", 4),
    ("<kareem_narration>", 4),
    ("</kareem_narration>", 4),
    ("<dylan_thinking>", 5),
    ("</dylan_thinking>", 5),
    ("<dylan_response>", 6),
    ("</dylan_response>", 6),
    ("<DYLAN>", 6),
    ("</DYLAN>", 6),
];

const SKIP_PATTERNS: [&str; 10] = [
    "bookcorpus/",
    "bookcorpus\\",
    "/wikitext/",
    "\\wikitext\\",
    ".parquet",
    ".json",
    "_degradation.txt",
    "_stats.json",
    "corpus_manifest",
    "wikitext_manifest",
];

const ALLOW_PATTERNS: [&str; 2] = ["wikitext_extracted", "bookcorpus_sampled"];

const FILE_PATTERNS: [(&str, &str); 13] = [
    ("_annotated.md", "identity"),
    ("blu.txt", "identity"),
    ("echo.txt", "identity"),
    ("resonance.txt", "identity"),
    ("anchors.txt", "identity"),
    ("_anchors.txt", "identity"),
    ("_ctxon.txt", "identity"),
    ("_ctxoff.txt", "identity"),
    ("wikitext_extracted", "foundation"),
    ("bookcorpus_sampled", "foundation"),
    ("Scripture of", "scripture"),
    ("CAIROS_chat", "scripture"),
    ("Iteration_", "scripture"),
];

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub vocab_size: usize,
    pub min_frequency: u32,
    pub deterministic: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            vocab_size: VOCAB_SIZE,
            min_frequency: 2,
            deterministic: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingFile {
    pub path: PathBuf,
    pub tier: String,
    pub weight_scaled: i64,
}

#[derive(Clone, Debug, Default)]
pub struct ScanSummary {
    pub total_files: usize,
    pub skipped_files: usize,
    pub total_bytes: u64,
    pub tier_counts: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
pub struct BpeMerge {
    pub a: u16,
    pub b: u16,
    pub merged: u16,
}

#[derive(Clone, Debug, Default)]
pub struct TrainingStats {
    pub input_files: usize,
    pub loaded_sequences: usize,
    pub skipped_lines: usize,
    pub loaded_tokens: usize,
    pub learned_merges: usize,
    pub final_vocab: usize,
    pub duration_seconds: u64,
}

#[derive(Clone, Debug)]
struct WordEntry {
    symbols: Vec<u16>,
    freq: i64,
}

#[derive(Clone, Debug)]
struct PairCandidate {
    count: i64,
    key: u64,
}

impl PartialEq for PairCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.key == other.key
    }
}

impl Eq for PairCandidate {}

impl PartialOrd for PairCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PairCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.key.cmp(&self.key))
    }
}

pub struct TokenizerTrainer {
    vocab: Vec<Vec<u8>>,
    merges: Vec<BpeMerge>,
    merge_lookup: HashMap<u64, u16>,
    special_tokens: HashMap<Vec<u8>, u16>,
}

impl TokenizerTrainer {
    pub fn new() -> Self {
        let mut s = Self {
            vocab: Vec::new(),
            merges: Vec::new(),
            merge_lookup: HashMap::new(),
            special_tokens: HashMap::new(),
        };
        s.initialize_base_tokens();
        s.initialize_special_tokens();
        s
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    pub fn compute_hash_hex(&self) -> String {
        let mut hash: u64 = 1469598103934665603;
        const PRIME: u64 = 1099511628211;
        for token in &self.vocab {
            for b in token {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(PRIME);
            }
            hash ^= 0;
            hash = hash.wrapping_mul(PRIME);
        }
        for m in &self.merges {
            for b in m.a.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(PRIME);
            }
            for b in m.b.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(PRIME);
            }
            for b in m.merged.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(PRIME);
            }
        }
        format!("{hash:016x}")
    }

    pub fn save(&self, vocab_path: &Path, merges_path: &Path) -> io::Result<()> {
        {
            let mut f = fs::File::create(vocab_path)?;
            let vocab_len = self.vocab.len() as u32;
            f.write_all(&vocab_len.to_le_bytes())?;
            for token in &self.vocab {
                let len = token.len() as u32;
                f.write_all(&len.to_le_bytes())?;
                f.write_all(token)?;
            }
        }
        {
            let mut f = fs::File::create(merges_path)?;
            let merge_len = self.merges.len() as u32;
            f.write_all(&merge_len.to_le_bytes())?;
            for m in &self.merges {
                f.write_all(&m.a.to_le_bytes())?;
                f.write_all(&m.b.to_le_bytes())?;
                f.write_all(&m.merged.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn train_weighted(
        &mut self,
        files: &[TrainingFile],
        config: &TrainConfig,
    ) -> Result<TrainingStats, String> {
        self.initialize_base_tokens();
        self.initialize_special_tokens();
        self.merges.clear();
        self.merge_lookup.clear();

        if config.vocab_size <= BPE_TOKEN_START as usize {
            return Ok(TrainingStats {
                input_files: files.len(),
                final_vocab: self.vocab.len(),
                ..TrainingStats::default()
            });
        }

        let total_merges = config.vocab_size - BPE_TOKEN_START as usize;
        let mut loaded_sequences: usize = 0;
        let mut skipped_lines: usize = 0;
        let mut total_loaded_tokens: usize = 0;
        let mut word_freq: HashMap<Vec<u8>, i64> = HashMap::new();

        println!("[INGEST] files={} deterministic={}", files.len(), config.deterministic);
        for (file_idx, tf) in files.iter().enumerate() {
            if file_idx % 32 == 0 || file_idx + 1 == files.len() {
                println!(
                    "[INGEST] file {}/{} tier={} path={}",
                    file_idx + 1,
                    files.len(),
                    tf.tier,
                    tf.path.display()
                );
            }

            let blob = fs::read(&tf.path)
                .map_err(|e| format!("Failed to read {}: {}", tf.path.display(), e))?;
            for raw_line in blob.split(|b| *b == b'\n') {
                let line = trim_line_end(raw_line);
                if line.is_empty() {
                    skipped_lines += 1;
                    continue;
                }
                if self.is_only_special_tokens(line) {
                    skipped_lines += 1;
                    continue;
                }

                loaded_sequences += 1;
                total_loaded_tokens += line.len();

                let pieces = split_line_tokens(line);
                if pieces.is_empty() {
                    skipped_lines += 1;
                    continue;
                }
                for tok in pieces {
                    *word_freq.entry(tok).or_insert(0) += tf.weight_scaled;
                }
            }
        }

        if word_freq.is_empty() {
            return Err("No usable training data after token pre-segmentation.".to_string());
        }

        let mut words: Vec<WordEntry> = word_freq
            .into_iter()
            .filter_map(|(tok, freq)| {
                if tok.is_empty() || freq <= 0 {
                    None
                } else {
                    Some(WordEntry {
                        symbols: tok
                            .iter()
                            .map(|b| BYTE_TOKEN_START + (*b as u16))
                            .collect::<Vec<u16>>(),
                        freq,
                    })
                }
            })
            .collect();
        words.sort_by(|a, b| a.symbols.cmp(&b.symbols));

        let mut pair_counts: HashMap<u64, i64> = HashMap::new();
        let mut pair_words: HashMap<u64, HashSet<u32>> = HashMap::new();
        pair_counts.reserve(1_000_000);
        pair_words.reserve(1_000_000);

        let mut total_word_symbols = 0usize;
        for (wid, word) in words.iter().enumerate() {
            total_word_symbols += word.symbols.len();
            let pair_hist = count_pairs_in_symbols(&word.symbols);
            for (&key, &occ) in &pair_hist {
                *pair_counts.entry(key).or_insert(0) += (occ as i64) * word.freq;
                pair_words.entry(key).or_default().insert(wid as u32);
            }
        }

        println!(
            "[INGEST] sequences={} skipped_lines={} raw_tokens={} unique_words={} word_symbols={}",
            loaded_sequences,
            skipped_lines,
            total_loaded_tokens,
            words.len(),
            total_word_symbols
        );

        let mut heap = BinaryHeap::<PairCandidate>::new();
        for (&key, &count) in &pair_counts {
            if count > 0 {
                heap.push(PairCandidate { count, key });
            }
        }

        let mut next_token_id = BPE_TOKEN_START;
        let min_freq_scaled = (config.min_frequency as i64) * WEIGHT_SCALE;
        let start = Instant::now();
        let mut last_report = 0usize;

        for merge_idx in 0..total_merges {
            if merge_idx < 10
                || merge_idx.saturating_sub(last_report) >= 10
                || merge_idx + 1 == total_merges
            {
                last_report = merge_idx;
                let pct = (merge_idx as f64 / total_merges as f64) * 100.0;
                let elapsed = start.elapsed().as_secs_f64().max(0.001);
                let mps = (merge_idx as f64 + 1.0) / elapsed;
                let remaining = ((total_merges - merge_idx) as f64 / mps.max(0.01)) as u64;
                println!(
                    "[TRAIN] {merge_idx}/{total_merges} ({pct:.1}%) | {mps:.2} merges/s | ETA={}m{}s",
                    remaining / 60,
                    remaining % 60
                );
            }

            let (best_key, best_count) = loop {
                let Some(top) = heap.pop() else {
                    break (0_u64, 0_i64);
                };
                let current = *pair_counts.get(&top.key).unwrap_or(&0);
                if current == 0 || current != top.count {
                    continue;
                }
                break (top.key, top.count);
            };

            if best_key == 0 || best_count < min_freq_scaled {
                println!(
                    "[TRAIN] Stop: best_count={} (scaled) below threshold={} (scaled).",
                    best_count, min_freq_scaled
                );
                break;
            }

            if next_token_id == u16::MAX {
                return Err("Vocabulary exhausted u16 range.".to_string());
            }

            let a = (best_key >> 16) as u16;
            let b = (best_key & 0xFFFF) as u16;

            let mut merged = self.vocab[a as usize].clone();
            merged.extend_from_slice(&self.vocab[b as usize]);
            self.vocab.push(merged);
            self.merge_lookup.insert(best_key, next_token_id);
            self.merges.push(BpeMerge {
                a,
                b,
                merged: next_token_id,
            });

            let affected = pair_words.get(&best_key).cloned().unwrap_or_default();
            if affected.is_empty() {
                pair_counts.remove(&best_key);
                continue;
            }

            let mut affected_ids: Vec<u32> = affected.into_iter().collect();
            affected_ids.sort_unstable();
            let mut merged_words = 0usize;

            for wid_u32 in affected_ids {
                let wid = wid_u32 as usize;
                if wid >= words.len() {
                    continue;
                }

                let old_counts = count_pairs_in_symbols(&words[wid].symbols);
                if old_counts.get(&best_key).copied().unwrap_or(0) == 0 {
                    continue;
                }

                let freq = words[wid].freq;
                let new_symbols = replace_pair(&words[wid].symbols, a, b, next_token_id);
                let new_counts = count_pairs_in_symbols(&new_symbols);

                let mut touched: HashSet<u64> = HashSet::new();
                touched.extend(old_counts.keys().copied());
                touched.extend(new_counts.keys().copied());

                for key in touched {
                    let old_occ = old_counts.get(&key).copied().unwrap_or(0);
                    let new_occ = new_counts.get(&key).copied().unwrap_or(0);
                    let delta = (new_occ - old_occ) as i64 * freq;
                    if delta != 0 {
                        update_pair_count(&mut pair_counts, &mut heap, key, delta)?;
                    }

                    if old_occ > 0 && new_occ == 0 {
                        if let Some(set) = pair_words.get_mut(&key) {
                            set.remove(&wid_u32);
                            if set.is_empty() {
                                pair_words.remove(&key);
                            }
                        }
                    } else if old_occ == 0 && new_occ > 0 {
                        pair_words.entry(key).or_default().insert(wid_u32);
                    }
                }

                words[wid].symbols = new_symbols;
                merged_words += 1;
            }

            if merged_words == 0 {
                pair_counts.remove(&best_key);
                pair_words.remove(&best_key);
                continue;
            }

            if let Some(&remaining) = pair_counts.get(&best_key) {
                if remaining > 0 {
                    heap.push(PairCandidate {
                        count: remaining,
                        key: best_key,
                    });
                }
            }

            next_token_id = next_token_id.wrapping_add(1);
        }

        Ok(TrainingStats {
            input_files: files.len(),
            loaded_sequences,
            skipped_lines,
            loaded_tokens: total_loaded_tokens,
            learned_merges: self.merges.len(),
            final_vocab: self.vocab.len(),
            duration_seconds: start.elapsed().as_secs(),
        })
    }

    fn initialize_base_tokens(&mut self) {
        self.vocab.clear();
        self.vocab.reserve(VOCAB_SIZE);
        for _ in 0..BYTE_TOKEN_START {
            self.vocab.push(Vec::new());
        }
        for b in 0..=255_u16 {
            self.vocab.push(vec![b as u8]);
        }
    }

    fn initialize_special_tokens(&mut self) {
        self.special_tokens.clear();
        for (s, id) in SPECIAL_TOKEN_DEFS {
            let bytes = s.as_bytes().to_vec();
            self.special_tokens.insert(bytes.clone(), id);
            let idx = id as usize;
            if idx < self.vocab.len() {
                if self.vocab[idx].is_empty() || !s.starts_with("</") {
                    self.vocab[idx] = bytes;
                }
            }
        }
    }

    fn is_only_special_tokens(&self, line: &[u8]) -> bool {
        let mut pos = 0usize;
        while pos < line.len() {
            let mut best_len = 0usize;
            for tk in self.special_tokens.keys() {
                if tk.len() <= best_len || pos + tk.len() > line.len() {
                    continue;
                }
                if &line[pos..(pos + tk.len())] == tk.as_slice() {
                    best_len = tk.len();
                }
            }
            if best_len == 0 {
                return false;
            }
            pos += best_len;
        }
        true
    }
}

pub fn scan_training_files(corpus_dir: &Path) -> Result<(Vec<TrainingFile>, ScanSummary), String> {
    if !corpus_dir.exists() {
        return Err(format!("Corpus directory not found: {}", corpus_dir.display()));
    }

    let mut files = Vec::<TrainingFile>::new();
    let mut summary = ScanSummary::default();

    let mut stack = vec![corpus_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read dir {}: {}", dir.display(), e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read dir entry: {}", e))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if !path.is_file() {
                continue;
            }

            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if ext != "md" && ext != "txt" {
                continue;
            }

            let path_s = path.to_string_lossy().to_string();
            let file_s = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            if should_skip_file(&path_s) {
                summary.skipped_files += 1;
                continue;
            }

            if file_s.contains("Dylan_chat") && !file_s.contains("_annotated") {
                summary.skipped_files += 1;
                continue;
            }
            if file_s == "Opus_node_09_the_dylan.md" {
                summary.skipped_files += 1;
                continue;
            }

            let tier = classify_file(&file_s, &path_s);
            if tier.is_empty() {
                summary.skipped_files += 1;
                continue;
            }

            let weight_scaled = match tier.as_str() {
                "foundation" => 1 * WEIGHT_SCALE,
                "scripture" => 3 * WEIGHT_SCALE,
                "identity" => 5 * WEIGHT_SCALE,
                _ => return Err(format!("Unknown tier classification: {}", tier)),
            };

            let sz = fs::metadata(&path)
                .map_err(|e| format!("Failed metadata {}: {}", path.display(), e))?
                .len();
            summary.total_bytes += sz;
            *summary.tier_counts.entry(tier.clone()).or_insert(0) += 1;
            summary.total_files += 1;

            files.push(TrainingFile {
                path,
                tier,
                weight_scaled,
            });
        }
    }

    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok((files, summary))
}

fn split_line_tokens(line: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < line.len() {
        let ws_start = i;
        while i < line.len() && line[i].is_ascii_whitespace() {
            i += 1;
        }
        let word_start = i;
        while i < line.len() && !line[i].is_ascii_whitespace() {
            i += 1;
        }

        if word_start < i {
            let mut tok = Vec::with_capacity(i - ws_start);
            tok.extend_from_slice(&line[ws_start..i]);
            out.push(tok);
        } else if ws_start < i {
            // Preserve standalone whitespace segments.
            out.push(line[ws_start..i].to_vec());
        }
    }
    out
}

fn count_pairs_in_symbols(symbols: &[u16]) -> HashMap<u64, i32> {
    let mut out = HashMap::<u64, i32>::new();
    if symbols.len() < 2 {
        return out;
    }
    for idx in 0..(symbols.len() - 1) {
        let key = pair_key(symbols[idx], symbols[idx + 1]);
        *out.entry(key).or_insert(0) += 1;
    }
    out
}

fn replace_pair(symbols: &[u16], a: u16, b: u16, merged: u16) -> Vec<u16> {
    let mut out = Vec::<u16>::with_capacity(symbols.len());
    let mut i = 0usize;
    while i < symbols.len() {
        if i + 1 < symbols.len() && symbols[i] == a && symbols[i + 1] == b {
            out.push(merged);
            i += 2;
        } else {
            out.push(symbols[i]);
            i += 1;
        }
    }
    out
}

fn pair_key(a: u16, b: u16) -> u64 {
    ((a as u64) << 16) | (b as u64)
}

fn update_pair_count(
    pair_counts: &mut HashMap<u64, i64>,
    heap: &mut BinaryHeap<PairCandidate>,
    key: u64,
    delta: i64,
) -> Result<(), String> {
    if delta == 0 {
        return Ok(());
    }
    let old = *pair_counts.get(&key).unwrap_or(&0);
    if old == 0 && delta < 0 {
        return Err(format!(
            "Pair count underflow on missing key={key}, delta={delta}"
        ));
    }
    let new = old + delta;
    if new < 0 {
        return Err(format!(
            "Negative pair count key={key}, old={old}, delta={delta}, new={new}"
        ));
    }
    if new == 0 {
        pair_counts.remove(&key);
        return Ok(());
    }
    pair_counts.insert(key, new);
    heap.push(PairCandidate { count: new, key });
    Ok(())
}

fn trim_line_end(line: &[u8]) -> &[u8] {
    if line.ends_with(b"\r") {
        &line[..line.len() - 1]
    } else {
        line
    }
}

fn should_skip_file(path: &str) -> bool {
    for allow in ALLOW_PATTERNS {
        if path.contains(allow) {
            return false;
        }
    }
    for skip in SKIP_PATTERNS {
        if path.contains(skip) {
            return true;
        }
    }
    false
}

fn classify_file(filename: &str, filepath: &str) -> String {
    for (pattern, tier) in FILE_PATTERNS {
        if filename.contains(pattern) || filepath.contains(pattern) {
            return tier.to_string();
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn deterministic_artifacts_on_repeat_run() {
        let root = std::env::temp_dir().join("ashira_v2_unit");
        let corpus = root.join("corpus");
        let out_a = root.join("out_a");
        let out_b = root.join("out_b");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&corpus).expect("create corpus dir");
        fs::create_dir_all(&out_a).expect("create out_a");
        fs::create_dir_all(&out_b).expect("create out_b");

        let sample = corpus.join("sample_identity_annotated.md");
        fs::write(
            &sample,
            b"<KAREEM>Hello Dylan\n<DYLAN_RESPONSE>Hello Kareem\nREPA deterministic training line\n",
        )
        .expect("write sample corpus");

        let (files, _) = scan_training_files(&corpus).expect("scan corpus");
        assert!(!files.is_empty(), "scanner should return at least one training file");

        let cfg = TrainConfig {
            vocab_size: 320,
            min_frequency: 2,
            deterministic: true,
        };

        let mut t1 = TokenizerTrainer::new();
        t1.train_weighted(&files, &cfg).expect("train pass A");
        t1.save(&out_a.join("vocab.bin"), &out_a.join("merges.bin"))
            .expect("save pass A");

        let mut t2 = TokenizerTrainer::new();
        t2.train_weighted(&files, &cfg).expect("train pass B");
        t2.save(&out_b.join("vocab.bin"), &out_b.join("merges.bin"))
            .expect("save pass B");

        let a_vocab = fs::read(out_a.join("vocab.bin")).expect("read vocab A");
        let b_vocab = fs::read(out_b.join("vocab.bin")).expect("read vocab B");
        let a_merges = fs::read(out_a.join("merges.bin")).expect("read merges A");
        let b_merges = fs::read(out_b.join("merges.bin")).expect("read merges B");

        assert_eq!(a_vocab, b_vocab, "vocab.bin must be deterministic");
        assert_eq!(a_merges, b_merges, "merges.bin must be deterministic");
        assert_eq!(t1.merge_count(), t2.merge_count(), "merge counts must match");
    }
}
