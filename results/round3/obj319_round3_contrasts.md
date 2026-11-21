items 319, word-reading components 210

| backbone | no edit | plain kana | tag (ours) | tag - kana [95% CI], p | tag - none [95% CI], p |
|---|---|---|---|---|---|
| Sarashina | 0.511 | 0.843 | 0.875 | +0.031 [-0.016, +0.080], 0.2 | +0.364 [+0.291, +0.436], 1.6e-28 |
| CosyVoice 2 | 0.282 | 0.850 | 0.784 | -0.066 [-0.113, -0.019], 0.0086 | +0.502 [+0.424, +0.576], 9.4e-35 |
| Irodori | 0.586 | 0.909 | 0.868 | -0.041 [-0.075, -0.009], 0.019 | +0.282 [+0.215, +0.351], 8e-20 |
| T5Gemma | 0.436 | 0.906 | 0.793 | -0.113 [-0.154, -0.073], 1e-07 | +0.357 [+0.277, +0.436], 6e-21 |

CosyVoice 2 released UtterTune: 0.856; tag - UT -0.072 [-0.122, -0.022], p=0.0038 (discordant 18 vs 41)

Voice disagreement over 150 same-word pairs under different voices (share of pairs whose hits differ):
  Sarashina: tag 0.147  none 0.160  kana 0.180
  CosyVoice 2: tag 0.260  none 0.147  kana 0.220
  Irodori: tag 0.153  none 0.133  kana 0.100
  T5Gemma: tag 0.327  none 0.187  kana 0.127

Inaudible outputs (peak < 0.05):
  sara42g noedit: 0/319
  sara42g kana: 0/319
  sara42g ours: 0/319
  cv242g noedit: 0/319
  cv242g kana: 0/319
  cv242g ours: 0/319
  iro42g noedit: 0/319
  iro42g kana: 0/319
  iro42g ours: 0/319
  t5g42g noedit: 0/319
  t5g42g kana: 0/319
  t5g42g ours: 7/319
