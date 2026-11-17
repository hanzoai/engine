pub mod xgrammar;

pub use xgrammar::XGrammarLogitProcessor;

/// Enum to select the constrained decoding strategy.
pub enum ConstrainedStrategy {
    None,
    Regex(String),
    XGrammar(String), // JSON Schema or Tool Call Grammar
}
