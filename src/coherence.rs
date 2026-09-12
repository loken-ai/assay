//! Whether a response is a repeating pattern rather than an answer.
//!
//! This is the check that needs no expected answer, which is the only reason it will
//! actually be run: a gate that has to be told what the model should say is a gate every
//! sweep leaves off, and one did - `coherence_pass` was null in every cell of a whole
//! campaign while gemma4:31b emitted "--- --- --- ---" for 128 tokens and the table
//! recorded it as a throughput result against ollama's prose.
//!
//! A degenerate answer repeats a handful of units over and over, so the share of DISTINCT
//! units collapses. Real prose and real code both sit far above the threshold - the
//! separation is not delicate, which matters because a false accusation of garbage is
//! worse than the gap it closes.
//!
//! The verdict is taken where the answer is still whole. It used to be recomputed from
//! the stored preview, which is the first 120 characters, and a 128-token answer runs to
//! five hundred or more: a loop that begins after the opening sentence was invisible, and
//! a campaign reported one engine degenerate and ours sound when both were looping.

/// A literal `<0xHH>` byte-fallback piece left in the generated text.
pub(crate) fn has_literal_byte_piece(text: &str) -> bool {
    let b = text.as_bytes();
    b.windows(3).enumerate().any(|(i, w)| {
        w == b"<0x"
            && b.get(i + 3).is_some_and(u8::is_ascii_hexdigit)
            && b.get(i + 4).is_some_and(u8::is_ascii_hexdigit)
            && b.get(i + 5) == Some(&b'>')
    })
}

/// A `0x` run of at least four whole bytes that spell printable ASCII.
///
/// Four bytes rather than two: `0x4142` is short enough to appear in a real answer about
/// encodings, and the observed damage was never shorter than four.
pub(crate) fn spells_ascii_in_hex(text: &str) -> bool {
    let b = text.as_bytes();
    for i in 0..b.len().saturating_sub(2) {
        if &b[i..i + 2] != b"0x" {
            continue;
        }
        let mut j = i + 2;
        while j < b.len() && b[j].is_ascii_hexdigit() {
            j += 1;
        }
        let digits = &b[i + 2..j];
        if digits.len() < 8 || !digits.len().is_multiple_of(2) {
            continue;
        }
        let spells_text = digits.chunks(2).all(|pair| {
            std::str::from_utf8(pair)
                .ok()
                .and_then(|h| u8::from_str_radix(h, 16).ok())
                .is_some_and(|v| (0x20..=0x7e).contains(&v))
        });
        if spells_text {
            return true;
        }
    }
    false
}

pub(crate) fn looks_degenerate(text: &str) -> bool {
    // Undecodable bytes. gemma4:31b answered "kingdom" then 50 replacement characters and
    // the first version of this check passed it: they are all DISTINCT as units, so a
    // distinct-ratio sees variety where there is only damage.
    let chars = text.chars().count();
    if chars >= 20 && text.chars().filter(|c| *c == '\u{FFFD}').count() * 10 > chars {
        return true;
    }
    // Byte-fallback tokens that reached the text instead of being assembled into the
    // bytes they stand for. ernie4-5 was published at +124.5% decode and +240.3% energy
    // on 2026-09-02 while writing "a man named 0x7465653b" where ollama wrote prose -
    // those four bytes are "tee;". Every check in this function saw an ordinary answer,
    // because ordinary prose is exactly what surrounds the damage.
    //
    // Two shapes, and they need different rules. A literal `<0xHH>` piece is never
    // something a model means to say, so it is damage on sight. A bare `0x` run is not -
    // `0xDEADBEEF` is a constant any code answer may carry - so that one counts only when
    // the bytes it spells are printable text, which a real constant's are not.
    if has_literal_byte_piece(text) || spells_ascii_in_hex(text) {
        return true;
    }
    let units: Vec<&str> = text.split_whitespace().collect();
    if units.len() < 12 {
        return false; // too short to tell repetition from brevity
    }
    // A single unit repeated: "--- --- ---", "olde olde olde". Needs a longer run than the
    // phrase check below to be conclusive, so it is gated separately rather than by an early
    // return - the first version returned at 15 units and never reached the phrase check,
    // which is how a 12-unit "time-olde olde olde ..." passed.
    let distinct: std::collections::HashSet<&str> = units.iter().copied().collect();
    if units.len() >= 15 && (distinct.len() as f64) / (units.len() as f64) < 0.15 {
        return true;
    }
    // A PHRASE repeated, which the ratio above cannot see: "the end of a person, the end of
    // a person, ..." has a perfectly ordinary share of distinct words. Four of the five
    // gemma4 and lfm2 answers that were visibly looping passed the first check for exactly
    // this reason. Count how much of the text one three-word window covers.
    // Counting DISTINCT windows rather than the commonest one: a cycle of k words gives every
    // window a share of 1/k, so a "commonest window" threshold has to be tuned per cycle
    // length and misses the long ones - "the end of a person, ..." repeats five words and no
    // single window covers more than 22%. Distinct windows over total lands near 1.0 for
    // prose and collapses toward k/n for anything looping, whatever k is.
    let windows: Vec<[&str; 3]> = units.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
    let distinct_windows: std::collections::HashSet<&[&str; 3]> = windows.iter().collect();
    if (distinct_windows.len() as f64) / (windows.len() as f64) < 0.5 {
        return true;
    }
    // A sane opening followed by a looping tail - "king who was able to see a time, that,
    // that, that, ..." - keeps a high share of distinct windows because the head supplies
    // them. What gives it away is one unit owning most of the answer.
    let mut unit_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for u in &units {
        *unit_counts.entry(*u).or_insert(0) += 1;
    }
    unit_counts
        .values()
        .max()
        .map(|m| (*m as f64) / (units.len() as f64) > 0.40)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{has_literal_byte_piece, looks_degenerate, spells_ascii_in_hex};

    /// The gate must fire on a byte run that was rendered instead of assembled, and stay
    /// silent on the hex a code answer legitimately carries. Both halves are the point:
    /// the first is the ernie4-5 cell this was written for, the second is what stops the
    /// rule from accusing the `long` prompt, which is a code review.
    #[test]
    fn it_recognises_rendered_byte_tokens_and_leaves_real_hex_alone() {
        // ernie4-5, verbatim from the campaign's recorded preview.
        assert!(looks_degenerate(
            "man named \n0x7465653b, who was a knight in shining armor. He had a son named \
             0x7465653c, and they lived together in the kingdom of 0x7465653d. The king loved \
             him very much and gave him a lot of land for his inheritance."
        ));
        // The other shape: the piece itself, unassembled.
        assert!(has_literal_byte_piece("a line<0x0A>and the next"));
        // A constant spells bytes that are not text, so it is not damage.
        assert!(!spells_ascii_in_hex(
            "the sentinel is 0xDEADBEEF and the mask 0xFFFFFFFF"
        ));
        assert!(!looks_degenerate(
            "The function masks the header with 0xDEADBEEF before hashing it, which keeps the \
             low bits stable across runs and makes the test reproducible on both machines."
        ));
        // Too short to tell an encoding example from damage.
        assert!(!spells_ascii_in_hex("the byte pair 0x4142 spells AB"));
    }

    /// The check has to fire on what was actually observed, and stay silent on what a
    /// working model writes. A gate that cannot do both is worse than none: it either
    /// misses the garbage it was written for, or it accuses good cells and gets removed.
    #[test]
    fn it_recognises_a_repeating_answer_and_leaves_real_text_alone() {
        // gemma4:31b, verbatim from the campaign's recorded preview.
        assert!(looks_degenerate(&"--- ".repeat(40)));
        // The four that slipped through the first version, verbatim from the sweep. Each
        // has an ordinary share of distinct words and is plainly not an answer.
        assert!(looks_degenerate(
            "time-olde olde olde olde olde olde olde olde olde olde olde ol"
        ));
        assert!(looks_degenerate(
            "person, the end of a person, the end of a person, the end of a person, the end of a"
        ));
        assert!(looks_degenerate(
            "king who was able to see a time, that, that, that, that, that, that, that, that,"
        ));
        assert!(looks_degenerate(
            "time, in a kingdom far far away, to a time, in a kingdom far far away, to a time, \
             in a kingdom far far away, to a"
        ));
        assert!(looks_degenerate(&format!(
            "kingdom{}",
            "\u{FFFD}".repeat(55)
        )));
        // Single-token loops - the other shape this takes.
        assert!(looks_degenerate(&"three ".repeat(30)));
        assert!(looks_degenerate(&"the the the ".repeat(12)));

        assert!(!looks_degenerate(
            "To understand how a modern CPU pipeline works, we must first divide the \
             instruction into stages: fetch, decode, execute, memory access and write \
             back, each handled by a different part of the chip while the next \
             instruction is already entering the one before it."
        ));
        // Code repeats structure without repeating units, and must not be accused.
        assert!(!looks_degenerate(
            "fn main() { let mut total = 0; for i in 0..10 { total += i * 2; } \
             println!(\"{}\", total); let names = vec![\"ana\", \"bo\", \"cy\"]; \
             for n in names { println!(\"hello {}\", n); } }"
        ));
        // Brevity is not degeneracy - a short correct answer must pass.
        assert!(!looks_degenerate("Paris."));
        assert!(!looks_degenerate(
            "The capital of France is Paris, on the Seine."
        ));
    }
}
