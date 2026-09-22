//! The scorer against a committed run: every figure on the board is a pure function of the raw
//! sample files, so scoring them again yields the figures that were published.

use std::path::Path;

use hanzo_bench::board;
use serde_json::Value;

fn fixture() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/evo-rocm-qwen3-1p7b-20260721-174808"
    ))
}

/// The run's samples and manifest alone, in a directory of the run's name.
fn samples_only() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join(fixture().file_name().unwrap());
    std::fs::create_dir(&dir).unwrap();
    for entry in std::fs::read_dir(fixture()).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.ends_with(".json") && name != "board.json" {
            std::fs::copy(&path, dir.join(name)).unwrap();
        }
    }
    (tmp, dir)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn sorted_lines(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines: Vec<String> = text
        .lines()
        .filter(|l| !l.starts_with('%'))
        .map(Into::into)
        .collect();
    lines.sort();
    lines
}

#[test]
fn scoring_the_samples_reproduces_the_published_board() {
    let (_tmp, dir) = samples_only();
    let run = board::score(&dir).unwrap();
    assert_eq!((run.backend.as_str(), run.cells.len()), ("ROCm", 5));

    let published: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("board.json")).unwrap())
            .unwrap();
    for cell in &run.cells {
        let was = published
            .iter()
            .find(|c| c["phase"] == cell.shape.phase.name() && c["n"] == cell.shape.n)
            .unwrap();
        let rival = cell.rival.as_ref().unwrap();
        for (now, was) in [(&cell.hanzo, &was["hanzo"]), (&rival.llama, &was["llama"])] {
            let spread = now.spread.as_ref().unwrap();
            assert!(close(now.mean, was["mean"].as_f64().unwrap()));
            assert!(close(spread.ci, was["ci"].as_f64().unwrap()));
            assert!(close(spread.cv, was["cv"].as_f64().unwrap()));
            assert!(close(spread.std, was["std"].as_f64().unwrap()));
            assert!(close(now.best, was["best"].as_f64().unwrap()));
            assert_eq!(now.n as u64, was["n"].as_u64().unwrap());
        }
        let was = &was["ratio"];
        assert!(close(rival.ratio.ratio, was["ratio"].as_f64().unwrap()));
        assert!(close(rival.ratio.lo, was["lo"].as_f64().unwrap()));
        assert!(close(rival.ratio.hi, was["hi"].as_f64().unwrap()));
        assert!(close(rival.ratio.best, was["best"].as_f64().unwrap()));
        assert_eq!(rival.ratio.verdict.name(), was["verdict"].as_str().unwrap());
    }
    // The same rows and macros as were published; a board now reads prefill by length.
    for name in ["board.md", "board.tex", "results-data.tex"] {
        assert_eq!(
            sorted_lines(&dir.join(name)),
            sorted_lines(&fixture().join(name)),
            "{name}"
        );
    }
    let lengths: Vec<usize> = run.cells.iter().map(|c| c.shape.n).collect();
    assert_eq!(lengths, vec![500, 512, 2048, 4096, 128]);
}

#[test]
fn a_cell_is_filed_once_per_engine() {
    let (_tmp, dir) = samples_only();
    let filed =
        serde_json::to_value(board::evidence(&[board::read(&dir).unwrap()]).unwrap()).unwrap();
    let filed = filed.as_array().unwrap();
    assert_eq!(filed.len(), 10);
    let id = |engine: &str| format!("kernel-perf:{engine}/rocm/evo/Qwen3-1.7B-Q4_K_M:prefill-2048");
    let ours = filed
        .iter()
        .find(|e| e["id"] == id("hanzo-engine").as_str())
        .unwrap();
    assert!(close(ours["value"].as_f64().unwrap(), 3506.1271431747396));
    assert_eq!(ours["meta"]["versus"]["verdict"], "LOSS");
    assert_eq!(
        ours["meta"]["versus"]["subject"],
        "llama.cpp/rocm/evo/Qwen3-1.7B-Q4_K_M"
    );
    assert_eq!(
        (
            ours["meta"]["backend"].as_str(),
            ours["meta"]["model"].as_str()
        ),
        (Some("ROCm"), Some("Qwen3-1.7B-Q4_K_M"))
    );
    assert_eq!(
        (ours["meta"]["run"].as_str(), ours["meta"]["host"].as_str()),
        (Some("evo-rocm-qwen3-1p7b-20260721-174808"), Some("evo"))
    );
    assert_eq!(ours["lib_versions"]["hanzo-engine"], "1.7.87");
    let theirs = filed
        .iter()
        .find(|e| e["id"] == id("llama.cpp").as_str())
        .unwrap();
    assert!(theirs["meta"].get("versus").is_none() && theirs["meta"].get("args").is_none());
}
