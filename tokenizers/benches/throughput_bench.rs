//! Throughput benchmark for batch encoding with GPT-2.
//!
//! Matches the workload used by the Python benchmark (repeated text, 10M chars,
//! 1000 documents, encode_batch_fast) so that `cargo bench --bench throughput_bench`
//! measures the same performance. Set `RAYON_NUM_THREADS` (e.g. 72) to match
//! the Python run; otherwise throughput will be much lower on many-core hosts.
//!
//! Uses `Tokenizer::from_pretrained("gpt2", None)` (requires `--features http`) so it
//! runs without local data files; HuggingFace Hub or cache is used.

#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use std::hint::black_box;
use std::time::{Duration, Instant};
use tokenizers::tokenizer::{EncodeInput, Tokenizer};

const SAMPLE_TEXT: &str = "\
The quick brown fox jumps over the lazy dog. Machine learning models \
require tokenization to process text input. HuggingFace Tokenizers \
library provides fast, production-ready tokenizers implemented in Rust. \
Byte-Pair Encoding (BPE), WordPiece, and Unigram are popular algorithms. \
Natural language processing has evolved significantly with transformers. \
Large language models like GPT and BERT have revolutionized AI applications. \
Tokenization is a critical step that affects both training and inference.";

/// Builds a repeated-text corpus of `total_chars` and splits it into `num_docs` equal-sized documents.
fn repeated_corpus_docs(total_chars: usize, num_docs: usize) -> Vec<String> {
    let text = SAMPLE_TEXT.trim();
    let rep = (total_chars / text.len()) + 1;
    let corpus: String = text.repeat(rep).chars().take(total_chars).collect();
    let doc_len = total_chars / num_docs;
    (0..num_docs)
        .map(|i| corpus.chars().skip(i * doc_len).take(doc_len).collect())
        .collect()
}

fn iter_throughput_batch(
    iters: u64,
    tokenizer: &Tokenizer,
    batch: &[EncodeInput],
) -> Duration {
    let mut duration = Duration::ZERO;
    for _ in 0..iters {
        let batch = batch.to_vec();
        let start = Instant::now();
        let _ = black_box(tokenizer.encode_batch_fast(batch, false));
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_throughput(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None)
        .expect("from_pretrained(\"gpt2\") (requires --features http and network or HF cache)");

    // Match Python benchmark that reports ~311 MB/s at 72 threads: 10M chars, 1000 documents.
    // Fewer docs (e.g. 100) under-utilizes many cores and yields much lower MiB/s.
    const CORPUS_CHARS: usize = 10_000_000;
    const NUM_DOCS: usize = 1_000;
    let docs = repeated_corpus_docs(CORPUS_CHARS, NUM_DOCS);
    let total_bytes: usize = docs.iter().map(|s| s.len()).sum();
    let batch: Vec<EncodeInput> = docs.into_iter().map(Into::into).collect();

    let mut group = c.benchmark_group("batch-throughput");
    group.throughput(Throughput::Bytes(total_bytes as u64));
    group.sample_size(20);
    // Allow enough time to complete 20 samples (each ~26–47 ms on 1–72 cores).
    group.measurement_time(Duration::from_secs(6));
    group.bench_function("gpt2_encode_batch_10M_1000docs", |b| {
        b.iter_custom(|iters| iter_throughput_batch(iters, &tokenizer, &batch))
    });
}

fn bench_decode(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_pretrained("gpt2", None)
        .expect("from_pretrained(\"gpt2\") (requires --features http and network or HF cache)");

    const CORPUS_CHARS: usize = 10_000_000;
    const NUM_DOCS: usize = 1_000;
    let docs = repeated_corpus_docs(CORPUS_CHARS, NUM_DOCS);
    let batch: Vec<EncodeInput> = docs.into_iter().map(Into::into).collect();

    let encodings = tokenizer.encode_batch_fast(batch, false).unwrap();
    let token_id_seqs: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| e.get_ids().to_vec())
        .collect();
    let total_decoded_bytes: usize = token_id_seqs
        .iter()
        .map(|ids| tokenizer.decode(ids, false).unwrap().len())
        .sum();
    let decode_input: Vec<&[u32]> = token_id_seqs.iter().map(|v| v.as_slice()).collect();

    eprintln!(
        "Decode: {} seqs, {} total decoded bytes",
        decode_input.len(),
        total_decoded_bytes
    );

    let mut group = c.benchmark_group("decode-throughput");
    group.throughput(Throughput::Bytes(total_decoded_bytes as u64));
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(6));
    group.bench_function("gpt2_decode_batch_10M_1000docs", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                let _ = black_box(tokenizer.decode_batch(&decode_input, false));
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

criterion_group!(benches, bench_throughput, bench_decode);
criterion_main!(benches);
