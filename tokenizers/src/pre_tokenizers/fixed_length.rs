use crate::normalizer::Range;
use crate::tokenizer::{NormalizedString, PreTokenizedString, PreTokenizer, Result};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

use crate::utils::macro_rules_attribute;

#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct FixedLength {
    #[serde(default = "default_length")]
    pub length: usize,
}

impl FixedLength {
    pub fn new(length: usize) -> Self {
        Self { length }
    }
}

fn default_length() -> usize {
    5
}

// Thread-local pool of NormalizedString buffers to reuse in slice_into (avoids per-slice allocations).
thread_local! {
    static SLICE_POOL: RefCell<Vec<NormalizedString>> = RefCell::new(Vec::new());
}

impl PreTokenizer for FixedLength {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            let text = normalized.get();
            if text.is_empty() {
                return Ok(vec![]);
            }

            let char_positions: Vec<_> = text.char_indices().collect();
            let segments: Vec<(usize, usize)> = char_positions
                .chunks(self.length)
                .map(|chunk| {
                    let start = chunk.first().map(|(i, _)| *i).unwrap_or(0);
                    let end = chunk
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(text.len());
                    (start, end)
                })
                .collect();

            let result = SLICE_POOL.with(|cell| {
                let mut pool = cell.borrow_mut();
                let mut buf0 = pool.pop().unwrap_or_default();
                let mut buf1 = pool.pop().unwrap_or_default();
                let mut out = Vec::with_capacity(segments.len());
                for (start, end) in segments {
                    normalized
                        .slice_into(Range::Normalized(start..end), &mut buf0)
                        .ok_or("Failed to slice normalized text")?;
                    out.push(std::mem::replace(&mut buf0, std::mem::take(&mut buf1)));
                }
                pool.push(buf0);
                pool.push(buf1);
                Ok(out)
            });
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};

    #[test]
    fn basic() {
        let tests = vec![
            (
                "Hello world",
                vec![("Hello", (0, 5)), (" worl", (5, 10)), ("d", (10, 11))],
            ),
            ("Short", vec![("Short", (0, 5))]),
            ("", vec![]),
        ];
        let pretok = FixedLength { length: 5 };
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }

    #[test]
    fn custom_length() {
        let pretok = FixedLength { length: 3 };
        let mut pretokenized = PreTokenizedString::from("Hello world");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("wor", (6, 9)),
                ("ld", (9, 11)),
            ]
        );
    }

    #[test]
    fn utf8_characters() {
        let pretok = FixedLength { length: 3 };
        let mut pretokenized = PreTokenizedString::from("Hello 👋 world");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hel", (0, 3)),
                ("lo ", (3, 6)),
                ("👋 w", (6, 12)),
                ("orl", (12, 15)),
                ("d", (15, 16)),
            ]
        );
    }
}
