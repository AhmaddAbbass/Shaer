# Ashaar Runtime Layout

This runtime is now split into two self-contained packages:

- `bilstm_only/`: BiLSTM classifier assets (`bilstm_model/`) plus `reward.py` providing `classifier_meter_scores` and `bilstm_reward`.
- `ashaar_only/`: Ashaar structural analysis assets (`Ashaar/`, `test.yml`, etc.) plus `reward.py` providing `ashaar_meter_pattern_score` and `ashaar_reward`.

Import from the package you need, e.g.:

```python
from Models.Ashaar_runtime.bilstm_only import bilstm_reward
from Models.Ashaar_runtime.ashaar_only import ashaar_reward
```
