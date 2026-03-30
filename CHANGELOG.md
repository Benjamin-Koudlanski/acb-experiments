# Changelog

## v0.1.0 (2026-03-30)

Initial release accompanying the paper submission to TechRxiv.

### Included
- Core ACB library: I(n), ΔI(n), n*, P(harm|n), ρ_crit, topology crossover
- Coordination Bottleneck Index (CBI) calculator with diagnostics
- Greedy heterogeneous fleet selection algorithm
- Monte Carlo validation of P(harm|n) (reproduces paper Table 3)
- Experiment runners for P1 (fleet sizing), P2 (topology crossover), P3 (RAG diversity)
- Unified LLM backend supporting OpenAI, Ollama, and vLLM
- CLI tool: `acb cbi`, `acb nstar`, `acb pharm`, `acb crossover`
- Publication-quality plotting and statistical analysis
- Google Colab notebook for free replication with open models
- GitHub Actions CI workflow
- Unit tests for core model, CBI, and Monte Carlo validation

### Not Yet Included
- Experimental results (P1/P2/P3 have not been executed)
- Pre-built Docker container
- Support for sequential pipeline topologies
