# BPR (Bayesian Personalized Ranking)

Implementation of a BPR model using pure Python and pure Rust. Python model also loaded in Rust to 
see the differences in inference times. Development of a concurrent scenario in Rust to test the 
model exported. ml-1m-csv and ml-latest-small dataset on MovieLens used for training and testing.

## Project structure
- `/mf_bpr`: contains the pure Rust model, the test of the exported Python model and the benchmark used
             for testing the concurrent scenario.
- `/mf_bpr`: contains the pure Python model.
- `M1`: powerpoint about the project.

## Main crates and library
- Candle
- ort
- axum
- tokio
- onnx
- torch