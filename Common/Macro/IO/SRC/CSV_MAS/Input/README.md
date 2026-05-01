Raw MAS source CSVs for the Consolidated SORA parser live here.

Expected files:
- `DomesticInterestRates_Idx14_SORA.csv`
- `DomesticInterestRates_idx17_SORA3MthCompounded.csv`

Parser:
- `scripts/Consolidated/Pipeline_MODEL/4_Postprocessing/parse_mas_sora.py`

Outputs written by the parser:
- `scripts/Consolidated/IO/SRC/stage5_model_inputs/rates/sora_daily.csv`
- `scripts/Consolidated/IO/SRC/stage5_model_inputs/rates/sora_3m_daily.csv`
