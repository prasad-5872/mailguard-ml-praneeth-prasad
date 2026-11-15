# MailGuard ML

An end-to-end, explainable email spam detection system: raw-email parsing, feature
engineering, leakage-safe model comparison, threshold tuning, calibration, a
multi-page Streamlit app and in-app monitoring. Built as an AI/ML practical project
and deployable for free on Streamlit Community Cloud. It needs no API keys, databases
or containers.

| Page | What it does |
|---|---|
| **Email Analyzer** | Paste raw email or a body, upload an `.eml`, or try a sample. Shows spam/ham verdict, probability, an adjustable threshold, extracted features, parsed headers and URLs, and per-email explanations. |
| **Model Evaluation** | Compares the 4 candidate models, with confusion matrix, ROC and PR curves, calibration, threshold analysis and error analysis. |
| **Monitoring** | Live request counts, error rate, average and p95 latency, the spam/ham split, request history and structured logs. Real inference requests update all of it. |
| **Model Information** | Model card: version, training date, dataset and licence notes, test metrics, artifact checksum, environment and limitations. |

---

## Dataset

**[Apache SpamAssassin Public Mail Corpus](https://spamassassin.apache.org/old/publiccorpus/)**,
curated by Justin Mason / the Apache SpamAssassin project. Archives used:
`20030228_easy_ham`, `20030228_easy_ham_2`, `20030228_hard_ham`, `20030228_spam`,
`20050311_spam_2`. The 2002 snapshots are subsets of these; the 2005 `spam_2` drops one
mislabelled ham.

**Why this corpus:** it contains complete raw RFC 822 messages with original headers,
MIME structure and HTML. The app parses exactly that kind of input, so the training data
matches it. Most other public "email spam" datasets are pre-tokenised body text with no
headers.

**Licence:** the corpus has no formal open-source licence. Its readme says messages
were posted to public fora or sent knowing they might be made public, and that
*copyright of the message text remains with the original senders*. For that reason this
repo **does not redistribute the corpus**. `training/train.py` downloads it at training
time into the git-ignored `data/` folder. The readme also asks that these mails never be
sent into live mail systems.

**Compatibility check: why the dataset is not combined with others.** Enron-Spam and
Ling-Spam ship as body text with no headers, from different years and organisations.
Mixing them in would make "has headers / HTML / which era" a near-perfect proxy for the
*source* of an email rather than whether it is spam. The model would learn the dataset
instead of the task, and the header-based features would be undefined for half the data.
A single raw-format corpus avoids that confound.

**Preparation (measured on the actual run):** 6,046 files parsed, 0 parse errors and
0 label errors. 182 exact duplicates were removed (normalised subject + body), and
0 conflicting-label duplicates were found. That leaves 5,864 unique emails, about 30.2%
spam. They fall into 5,360 near-duplicate groups.

## ML pipeline

```
raw .eml / pasted text
  └─ parsing (email stdlib, compat32, charset fallbacks, HTML→text, URL & link extraction)
      ├─ text: subject + body → normalise (URLs→urltoken, emails→emailtoken, digits→0)
      │     ├─ word TF-IDF (1–2 grams)
      │     └─ char_wb TF-IDF (3–5 grams)
      └─ 25 structured features (URL/IP/shortener counts, link-text mismatch, Reply-To ≠ From,
         HTML/forms/images, caps & '!' ratios, spam-phrase counts, missing Message-ID …)
  └─ candidate models → Platt calibration (NB, SVM) → tuned threshold → SpamModel artifact
```

**Leakage controls**

1. **Deduplication before splitting**, plus **group-aware splitting**: near-duplicates
   (the same campaign with different numbers, URLs or names) share a group key, and
   `StratifiedGroupKFold` keeps every group in a single split. This gives about 71/14/14
   train/validation/test (4,188 / 838 / 838).
2. **Header text is excluded from TF-IDF.** Ham and spam in this corpus arrived through
   different channels, so tokens like list names or relay hosts would leak the collection
   source. Only general header *properties* are used, such as a Reply-To/From domain
   mismatch.
3. TF-IDF vocabularies, IDF weights and scalers are **fitted on the training split only**.
4. Hyper-parameters, Platt calibration, the decision threshold and model selection all
   use the **validation** split. The **test split is only used for the final report.**

**Candidates:** Multinomial Naive Bayes, Logistic Regression, Linear SVM and LightGBM. The
LightGBM candidate uses a reduced vocabulary of 10k word and 10k char features. NB and
SVM scores are Platt-scaled because NB is over-confident and SVM margins are not
probabilities. Structured counts are `log1p`-transformed before scaling. Diagnosis on
validation data showed heavy-tailed counts (|z| up to 34) hurting the linear models, and
this change raised logistic regression's validation PR-AUC from 0.9944 to 0.9970.

**Threshold:** chosen on validation to maximise **F0.5**, which weights precision above
recall. A legitimate email lost to the spam folder usually costs more than a missed spam.

**Selection rule:** the highest validation F0.5 at the tuned threshold, with validation
PR-AUC as tie-break.

## Results

These come from the committed artifacts (`models/evaluation.json` and `models/model_card.json`),
produced by `python -m training.train` on 2026-11-13. The test split has 838 emails
(253 spam, 585 ham), and each model is scored at its own tuned threshold.

| Model | Thr. | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN | Val F0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Multinomial NB (Platt) | 0.79 | 0.9857 | 0.9959 | 0.9565 | 0.9758 | 0.9998 | 0.9994 | 1 | 11 | 0.9880 |
| Logistic Regression | 0.51 | 0.9881 | 0.9919 | 0.9684 | 0.9800 | 0.9989 | 0.9978 | 2 | 8 | 0.9839 |
| Linear SVM (Platt) | 0.75 | 0.9845 | 0.9959 | 0.9526 | 0.9737 | 0.9995 | 0.9988 | 1 | 12 | 0.9871 |
| **LightGBM (deployed)** | 0.46 | 0.9881 | 0.9880 | 0.9723 | 0.9801 | 0.9989 | 0.9978 | 3 | 7 | **0.9890** |

LightGBM won on validation F0.5 by a very small margin over Naive Bayes (0.9890 vs 0.9880).
On the test split the models are statistically very close. Read the ranking as
"all four are strong baselines", not as a decisive win.

Test errors for the deployed model, by corpus subset: `easy_ham` 0/352, `easy_ham_2` 2/206,
`hard_ham` 1/27, `spam` 1/80, `spam_2` 6/173.

## Explainability

* **Linear models:** each contribution is the weight multiplied by the feature value over
  the features present in the email. Together they form an exact additive decomposition
  of the score (log-likelihood ratios for NB).
* **LightGBM:** contributions are exact **TreeSHAP** values from LightGBM's `pred_contrib`.
  A test cross-checks them against the `shap` library. TreeSHAP can credit *absent*
  text, such as the lack of quoted-reply `>` lines. The UI marks these "(absent)".
* **Global view:** top coefficients for linear models, or LightGBM gain.

## Monitoring and logging

* Every request gets a 12-character request ID. It is timed with `perf_counter` and
  recorded as a success or an error (validation, model-unavailable or internal).
* Logs are timestamped, single-line JSON written to stdout and also kept in a bounded
  in-memory ring buffer (500 entries). **Raw email content is never logged.** Only
  metadata is: sizes, URL count, label, probability, latency and error type. A test
  enforces this.
* Metrics live in a thread-safe, bounded store. Counters run since process start; p95
  latency and history cover the last 1,000 requests. **The metrics are in-memory, so they
  reset whenever the app restarts.** That includes redeploys, crashes, and Streamlit
  Community Cloud putting the app to sleep. They are shared by all sessions of one app
  instance.

## Project structure

```
app.py                  Streamlit entry point (st.navigation, 4 pages)
views/                  analyzer.py · evaluation.py · monitoring.py · model_info.py
src/
  config.py             paths, limits, constants
  parsing.py            MIME/RFC 822 parsing, HTML→text, URL extraction
  features.py           text normalisation + 25 structured features
  validation.py         input validation
  model.py              preprocessors, candidates, SpamModel (predict/explain), artifact I/O
  trainer.py            leakage-safe fitting, calibration, threshold tuning, selection
  metrics.py            metrics, curves, threshold optimisation, percentiles
  data.py               corpus download/extract, label validation, dedup, group split
  inference.py          request pipeline: IDs, latency, logging, metrics
  monitoring.py         bounded in-memory metrics store
  logging_utils.py      JSON logging + ring buffer
  charts.py             Plotly figures (validated colour-blind-safe palette)
  app_resources.py      cached model / metrics singletons for Streamlit
training/train.py       end-to-end training CLI → models/
models/                 mailguard_model.joblib · model_card.json · evaluation.json (committed)
samples/                synthetic demo emails (fictional senders, example.* domains)
tests/                  pytest suite (parsing, features, validation, inference, thresholds,
                        metrics, model loading, monitoring, data prep, headless UI)
```

## Run locally

Use Python 3.12. The pinned versions match the environment that produced the model.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

## Train (regenerates `models/`)

```bash
python -m training.train                 # downloads the corpus (~7 MB) on first run
python -m training.train --skip-download # reuse data/raw/
```

The full pipeline took about 6 minutes on a laptop CPU. `--small` runs a quick smoke
test with tiny vocabularies. If you write it to another `--output-dir`, the committed
model stays untouched.

## Test

```bash
python -m pytest
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub, including `models/`.
2. Go to <https://share.streamlit.io>, choose **Create app**, then **Deploy a public app from GitHub**.
3. Select the repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings** and choose **Python 3.12**. No secrets are needed.
5. Deploy. Cloud installs `requirements.txt`, plus `packages.txt` (`libgomp1`, the OpenMP
   runtime LightGBM needs).

Because the trained model is committed, the cloud app never trains or downloads the
corpus. If you retrain, commit all three files in `models/` together. The app checks the
artifact's SHA-256 against `model_card.json` when it loads.

## Limitations

* The training mail dates from 2002–2005. Modern phishing, languages and obfuscation are
  under-represented, so expect lower real-world performance than the test metrics.
* Ham comes mostly from technical mailing lists and newsletters. Business, personal and
  non-English mail is under-represented, which raises the risk of false positives on it.
* Test metrics come from a random group-aware split of the same corpus, so they are
  optimistic. There is no temporal or cross-organisation test.
* `missing_date` never varies in the training data, so the model cannot use it even
  though the app extracts and displays it. Several other signals are very rare in the
  training emails, so the model has little evidence for them: missing Message-ID (0.02%),
  URL shorteners (0.1%), display-name domain mismatch (0.1%) and link-text mismatch (0.7%).
* Pasting a body without headers disables the header-based signals.
* There are no reputation checks: no SPF/DKIM/DMARC, URL reputation or attachment scanning.
* Model artifacts are joblib pickles. Only load ones you produced yourself.

## Licence

The code is MIT-licensed (see `LICENSE`). The SpamAssassin corpus is not included and is
not covered by this licence.
