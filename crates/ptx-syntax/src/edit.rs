/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fmt;
use std::ops::Range;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EditScript {
    edits: Vec<Edit>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Edit {
    range: Range<usize>,
    replacement: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditError {
    InvalidRange {
        range: Range<usize>,
    },
    Overlap {
        first: Range<usize>,
        second: Range<usize>,
    },
    OutOfBounds {
        range: Range<usize>,
        source_len: usize,
    },
    NonCharacterBoundary {
        offset: usize,
    },
}

impl fmt::Display for EditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange { range } => write!(
                formatter,
                "PTX edit range {}..{} is reversed",
                range.start, range.end
            ),
            Self::Overlap { first, second } => write!(
                formatter,
                "PTX edits {}..{} and {}..{} conflict",
                first.start, first.end, second.start, second.end
            ),
            Self::OutOfBounds { range, source_len } => write!(
                formatter,
                "PTX edit range {}..{} exceeds source length {source_len}",
                range.start, range.end
            ),
            Self::NonCharacterBoundary { offset } => {
                write!(
                    formatter,
                    "PTX edit offset {offset} is not a UTF-8 boundary"
                )
            }
        }
    }
}

impl std::error::Error for EditError {}

impl EditScript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty()
    }

    pub fn insert(&mut self, offset: usize, text: impl Into<String>) -> Result<(), EditError> {
        self.replace(offset..offset, text)
    }

    pub fn delete(&mut self, range: Range<usize>) -> Result<(), EditError> {
        self.replace(range, "")
    }

    pub fn replace(
        &mut self,
        range: Range<usize>,
        replacement: impl Into<String>,
    ) -> Result<(), EditError> {
        if range.start > range.end {
            return Err(EditError::InvalidRange { range });
        }
        if let Some(existing) = self
            .edits
            .iter()
            .find(|edit| ranges_conflict(&edit.range, &range))
        {
            return Err(EditError::Overlap {
                first: existing.range.clone(),
                second: range,
            });
        }
        self.edits.push(Edit {
            range,
            replacement: replacement.into(),
        });
        Ok(())
    }

    pub fn apply(&self, source: &str) -> Result<String, EditError> {
        let mut edits: Vec<&Edit> = self.edits.iter().collect();
        edits.sort_by_key(|edit| (edit.range.start, edit.range.end));

        for edit in &edits {
            if edit.range.end > source.len() {
                return Err(EditError::OutOfBounds {
                    range: edit.range.clone(),
                    source_len: source.len(),
                });
            }
            for offset in [edit.range.start, edit.range.end] {
                if !source.is_char_boundary(offset) {
                    return Err(EditError::NonCharacterBoundary { offset });
                }
            }
        }

        let replacement_bytes = edits
            .iter()
            .map(|edit| edit.replacement.len())
            .sum::<usize>();
        let removed_bytes = edits
            .iter()
            .map(|edit| edit.range.end - edit.range.start)
            .sum::<usize>();
        let mut output = String::with_capacity(source.len() + replacement_bytes - removed_bytes);
        let mut cursor = 0usize;
        for edit in edits {
            output.push_str(&source[cursor..edit.range.start]);
            output.push_str(&edit.replacement);
            cursor = edit.range.end;
        }
        output.push_str(&source[cursor..]);
        Ok(output)
    }
}

fn ranges_conflict(first: &Range<usize>, second: &Range<usize>) -> bool {
    match (first.is_empty(), second.is_empty()) {
        (false, false) => first.start < second.end && second.start < first.end,
        (true, true) => first.start == second.start,
        (true, false) => first.start >= second.start && first.start <= second.end,
        (false, true) => second.start >= first.start && second.start <= first.end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_non_overlapping_edits_in_source_order() {
        let mut edits = EditScript::new();
        edits.replace(6..12, "PTX").unwrap();
        edits.insert(0, "lossless ").unwrap();
        edits.delete(12..13).unwrap();
        assert_eq!(edits.apply("hello source!").unwrap(), "lossless hello PTX");
    }

    #[test]
    fn rejects_ambiguous_or_invalid_edits() {
        let mut edits = EditScript::new();
        edits.delete(2..5).unwrap();
        assert!(matches!(
            edits.insert(5, "x"),
            Err(EditError::Overlap { .. })
        ));
        assert!(matches!(
            edits.replace(Range { start: 8, end: 7 }, "x"),
            Err(EditError::InvalidRange { .. })
        ));
        let mut duplicate_insert = EditScript::new();
        duplicate_insert.insert(1, "x").unwrap();
        assert!(matches!(
            duplicate_insert.insert(1, "y"),
            Err(EditError::Overlap { .. })
        ));
    }

    #[test]
    fn validates_source_boundaries_when_applying() {
        let mut out_of_bounds = EditScript::new();
        out_of_bounds.delete(2..20).unwrap();
        assert!(matches!(
            out_of_bounds.apply("short"),
            Err(EditError::OutOfBounds { .. })
        ));

        let mut inside_utf8 = EditScript::new();
        inside_utf8.insert(1, "x").unwrap();
        assert!(matches!(
            inside_utf8.apply("λ"),
            Err(EditError::NonCharacterBoundary { offset: 1 })
        ));
    }
}
