# What others did: model fits to resting EEG, and evoked responses (2026-09-27)

Two questions: (1) how have others fitted neural-mass / whole-brain models to resting
EEG, and what did we miss; (2) would evoked (task) responses work better, has it been
done, and is there data. Short answers first, details and references below.

**Short answers.**
1. Our fitting method is at least as careful as the published work: likelihood,
   observation model, out-of-sample tests, reliability and batch checks. Most clinical
   papers have none of these. What we missed is in the *design*, not the fitting:
   - they fit **eyes-open and eyes-closed together**, with only one or two parameters
     allowed to change between them;
   - they include a **thalamic loop**, which makes alpha *and* beta;
   - they check parameters against **known answers** (age, drugs, pathology);
   - they test **one hypothesis across several paradigms**.

   Nobody has a replicated, model-based resting-EEG marker of depression. Our null result
   agrees with the field.
2. Evoked responses give better-posed parameters for two reasons. First, a known input
   time gives the phase of the response, not just its power. Second, a within-subject
   condition difference cancels skull, amplifier and batch effects. This has been done in
   depression (Kerr 2011, Pinotsis 2022, Gilbert & Zarate), and properly across paradigms
   in schizophrenia (Adams 2022). But depression effects on ERPs are also small, so it
   will not create an effect by itself.

   The best open dataset is **OpenNeuro ds003474**: the *same 122 people and IDs* as
   ds003478, which we already fit, doing a reward-learning task. **EMBARC** (rest + LDAEP
   + treatment outcome) is the best larger one, but it needs an access application.

---

## 1. Resting EEG: what others did

| Line of work | What they fit | Scale / result |
|---|---|---|
| **Corticothalamic neural field** (Robinson, Rennie) | Analytic linear spectrum of cortex + thalamic relay + reticular nucleus, noise-driven | Alpha *and* beta come from resonances of the corticothalamic loop [1]. EO vs EC fitted in 100 subjects [2]. Age trends in 1498 subjects aged 6–86 [3] |
| **Intracortical population model** (Liley) | EO and EC spectra together, with the fewest parameter changes between them | 82 subjects: **one** parameter (input to cortical inhibitory cells) explains alpha blocking and scales with it [4] |
| **Model comparison** (Bastiaens 2025) | JR, Moran-David-Friston, Liley, Robinson models side by side | Alpha comes from non-unique parameter sets. JR alpha is a limit cycle, and JR alone cannot produce alpha blocking. All models make beta harmonics [5] |
| **DCM for cross-spectra** (canonical microcircuit, few sources) | CSD of 2–6 sources, Bayesian model comparison, PEB | Resting MEG detects ketamine's NMDA/AMPA effect [6]. Test–retest over 2 weeks: 4/156 parameters changed [7]. One MDD study: 57 subjects, 3 frontal channels, no replication [8] |
| **Linear spectral graph model** (Raj) | Few global parameters, analytic network spectrum | Excitatory/inhibitory parameters from resting MEG track tau and amyloid in Alzheimer's [9]. Simulation-based inference on developmental EEG [10] |
| **TVB / whole brain** | Simulation-based inference toolkits [11] | In MDD: only TMS-evoked-response simulations (next section) and DBS planning. **No resting-EEG TVB fit in MDD found** |
| **Linearity** | Linear vs nonlinear models on rest data | Linear models describe macroscale resting dynamics best [12]. This supports our linear regime |
| **Descriptive EEG markers of MDD** | Meta-analyses | Treatment-prediction markers not clinically reliable [13]. Frontal alpha asymmetry is not diagnostic [14] |

Where the modelling has worked: **age, sleep and anaesthesia, Alzheimer's/FTD, and drug
challenges**. In all of these the EEG change is large. It has not worked for depression
in any replicated study.

### What we missed (ranked by value for effort)

1. **Joint EO + EC fit (within-subject contrast).** We fit eyes-closed only. Rowe [2] and
   Hartoyo [4] share all parameters between the two conditions and let 1–2 of them change.
   - The EO–EC change is measured inside one person on one amplifier, so skull, lead
     field and **batch** mostly cancel. Batch confounding is what sank TDBRAIN's
     Healthy-vs-MDD contrast.
   - Available: TDBRAIN (EO and EC for everyone), ds003478 (alternating EO/EC minutes),
     Mumtaz (separate EO and EC files). Not MODMA (eyes-closed only).
2. **A thalamic loop.** In the corticothalamic model, beta near twice the alpha frequency
   comes from the loop delay [1]. Our round-3 result showed that a cortical fast JR
   generator cannot do this. The Robinson model is linear with an analytic spectrum, so it
   drops into the M5.4 Whittle pipeline as a new transfer function.
3. **Positive controls with known answers.** We have parameter recovery and split-half
   reliability, but no test that a parameter *means* what its name says. Others use age
   [3], drug [6] or pathology [9]. **Age is available for free in our data:** if the
   parameters don't track age the way van Albada found, they won't track depression either.
4. **One hypothesis tested across paradigms.** Adams 2022 [18] tests one mechanism
   (pyramidal-cell gain) across rest, MMN, 40 Hz ASSR and fMRI in schizophrenia. We
   screened many parameters in one paradigm.
5. *(Optional)* **Simulation-based inference** [10, 11], to see non-Gaussian posteriors.
   Our Laplace approximation plus recovery tests covers most of this.

**What we did not miss** (and most published clinical fits lack): a proper likelihood,
out-of-sample and external tests, split-half calibration, and the batch check. For
example, the MDD DCM studies [8, 16] had no out-of-sample test.

---

## 2. Evoked responses

### 2a. Will it work?

**Why it should help.** In our linear model, resting EEG is the network driven by
unknown noise. The spectrum is |H(f)|² times an unknown input spectrum, so:
- phase is lost;
- "input colour" and "network resonance" are confused (hence the 62% population
  background we needed).

An evoked response with a known onset is the network's impulse response H(t):
- it gives latencies and phase, and where the input enters;
- ongoing activity is averaged away;
- a **condition difference** (deviant−standard, reward−no-reward, loud−soft) inside one
  person cancels skull, amplifier and batch effects.

The same transfer function we already compute gives the evoked response via an inverse
FFT of H(f)·U(f). The code change is small. Jansen & Rit built their model for visual
evoked potentials in the first place [15].

**Why it may still not give a depression result.**
- An ERP has few spatial patterns (2–4 topographies), so only a handful of sources are
  identifiable. You learn about a sensory or frontal pathway, not the whole brain.
- Components that differ in depression (P300, reward positivity) are cognitive. They
  depend on attention, motivation and medication, and have no clean external input.
- **Effects are small:**
  - P300 amplitude vs depression: r ≈ −0.15 after bias correction [17];
  - reward positivity: reduced, but a small effect [19].
- Per-subject signal-to-noise needs about 50–100 clean trials per condition.

**Verdict.** Better-posed parameters, yes. A depression marker from evoked data alone,
probably not. The strongest design is **rest + evoked in the same people**, with one
mechanistic hypothesis tested on both (the Adams 2022 [18] design).
- Easiest to fit: steady-state responses (SSVEP, 40 Hz ASSR; DCM for steady-state
  responses [22]) and early sensory ERPs (N1/P2, MMN).
- Hardest: P300 and the reward positivity.

### 2b. Has it been done?

| Study | Data | Model | Result |
|---|---|---|---|
| Jansen & Rit 1995 [15] | Visual evoked potentials | JR (original purpose) | Alpha and VEP from one model |
| Rennie, Robinson, Wright 2002 [20] | Spectra + ERPs | Corticothalamic | One model for both; ERP peaks from thalamic feedback |
| David et al. 2006; Garrido et al. 2007 [21] | ERPs, MMN | DCM (JR-based) | Standard method; MMN explained by feedback loops |
| **Kerr, Kemp, Rennie, Robinson 2011** [16a] | Auditory oddball ERPs: 49 melancholic, 34 non-melancholic, 111 subclinical, 98 HC (Brain Resource, not public) | Corticothalamic | Slower thalamocortical conduction in melancholia; less cortical and thalamocortical excitation, more reticular inhibition. No replication known |
| Pinotsis et al. 2022 [16] | MSIT task ERPs, 15 depressed / 34 HC | DCM, 6 sources | DCM features classify better than ERP features; small sample, no out-of-sample test |
| Gilbert, Zarate et al. 2018; 2021 [23] | MEG somatosensory / emotional faces, MDD on ketamine | DCM | Ketamine response linked to AMPA/NMDA parameters (within-subject drug contrast) |
| **Adams et al. 2022** [18] | Schizophrenia 108, relatives 57, HC 107: rest + MMN + 40 Hz ASSR + fMRI | DCM (canonical microcircuit) | **One** parameter change (pyramidal self-inhibition) explains all paradigms. The template for Q2 |
| Momi, Wang, Griffiths 2023 [24] | TMS-EEG, healthy | Whole-brain JR (PyTepFit) | TMS-evoked responses from recurrent network activity |
| Hofsähs, …, Ritter 2026 [25] | TMS-EEG, 20 healthy (Biabani & Rogasch) | TVB, JR | MDD **simulated** only (GABA deficit gives larger TEPs); no MDD data |

### 2c. Datasets (MDD vs controls with evoked or task EEG)

| Dataset | Paradigm | n | Access | Verdict |
|---|---|---|---|---|
| **OpenNeuro ds003474** (Cavanagh) | Probabilistic selection: stimulus-locked visual responses; feedback (reward) responses, where Cavanagh 2019 found a smaller reward feature with depression [26] | 122 students, high/low BDI, **same people and IDs as ds003478** | Open, CC0, 17.9 GB, 64-ch Neuroscan, events with feedback codes (checked) | **Best first step:** joint rest + evoked fit in the same subjects |
| ds003478 run-2 | Rest 1 h later, after the task | Same 122 | Same bucket | Free within-subject rest–rest check |
| EMBARC (NDA #2199) | Rest EO/EC + **LDAEP** (tones at 5 loudness levels; serotonin-linked) [27] | MDD (~300) + HC; sertraline vs placebo | Needs NDA account and institutional Data Use Certification | Best large dataset (rest + evoked + outcome); weeks to get |
| MODMA ERP set | Dot-probe with emotional faces | 24 MDD / 29 HC, 128-ch | Separate MODMA download (licence form). **Only the resting set is on disk** | Small; worth adding if it is easy to get |
| OpenNeuro ds005356 | **MEG** reward-learning task | 52 MDD / 38 HC (SCID) | Open, 174 GB; rest "to be uploaded" | MEG, not EEG; large |
| Mumtaz TASK files (**on disk**, 61) | "P300" | 30 / 28 | Downloaded | **No event markers in the files (checked), so epochs can't be cut. Not usable as is** |
| TDBRAIN oddball (**on disk**) | Auditory oddball, 500/1000 Hz | 129 subjects, **all healthy, all early batch** (checked) | Downloaded | Like PyTepFit: method development only |
| HBN-EEG | Surround suppression (flicker: steady-state responses), other tasks | 3000+ aged 5–21, some with depressive disorders | Open (OpenNeuro) | Large, but youth with heavy comorbidity |
| iSPOT-D / Brain Resource (Kerr 2011's data), MDD TMS-EEG | — | — | Not open | — |

## 3. What I would do, in order

1. **Joint EO+EC fit with an age positive control**, on data already on disk (ds003478,
   Mumtaz, TDBRAIN). Shared parameters plus 1–2 condition-specific ones. Needs no new
   data; addresses the batch problem and whether the parameters mean anything.
2. **Thalamic relay + reticular populations in the linear transfer function**, the
   Robinson-style fix for beta.
3. **Download ds003474** and fit feedback-locked and stimulus-locked responses together
   with ds003478 rest, in the same ~120 people. Test one hypothesis: high BDI lowers the
   gain of the frontal response to reward.
4. **EMBARC application** (rest + LDAEP + sertraline outcome) as the confirmatory,
   larger dataset.

## References

1. Robinson PA, Rennie CJ, Wright JJ, et al. *Prediction of electroencephalographic spectra from neurophysiology.* Phys Rev E 2001;63:021903. https://journals.aps.org/pre/abstract/10.1103/PhysRevE.63.021903
2. Rowe DL, Robinson PA, Rennie CJ. *Estimation of neurophysiological parameters from the waking EEG using a biophysical model of brain dynamics.* J Theor Biol 2004;231:413–433. https://www.sciencedirect.com/science/article/abs/pii/S002251930400325X
3. van Albada SJ, Kerr CC, Chiang AKI, Rennie CJ, Robinson PA. *Neurophysiological changes with age probed by inverse modeling of EEG spectra.* Clin Neurophysiol 2010;121:21–38. https://pubmed.ncbi.nlm.nih.gov/19854102/
4. Hartoyo A, Cadusch PJ, Liley DTJ, Hicks DG. *Inferring a simple mechanism for alpha-blocking by fitting a neural population model to EEG spectra.* PLoS Comput Biol 2020;16:e1007662. https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007662
5. Bastiaens SP, et al. *A comprehensive investigation of intracortical and corticothalamic models of the alpha rhythm.* PLoS Comput Biol 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12064047/
6. Muthukumaraswamy SD, Shaw AD, Jackson LE, Hall J, Moran R, Saxena N. *Evidence that subanesthetic doses of ketamine cause sustained disruptions of NMDA and AMPA-mediated frontoparietal connectivity in humans.* J Neurosci 2015;35:11694–11706. https://www.jneurosci.org/content/35/33/11694
7. Jafarian A, et al. *Reliability of dynamic causal modelling of resting-state magnetoencephalography.* Hum Brain Mapp 2024;45:e26782. https://pmc.ncbi.nlm.nih.gov/articles/PMC11237883/
8. *A feasibility study on inferring connectivity changes in frontal lobes of MDD patients via spectral DCM.* Brain Informatics 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13280083/
9. Ranasinghe KG, et al. *Altered excitatory and inhibitory neuronal subpopulation parameters are distinctly associated with tau and amyloid in Alzheimer's disease.* eLife 2022;11:e77850. https://elifesciences.org/articles/77850
10. *Simulation-based inference of developmental EEG maturation with the spectral graph model.* Commun Phys 2024. https://www.nature.com/articles/s42005-024-01748-w
11. *Virtual Brain Inference (VBI), a flexible and integrative toolkit for efficient probabilistic inference on whole-brain models.* eLife 2025. https://elifesciences.org/articles/106194
12. Nozari E, et al. *Macroscopic resting-state brain dynamics are best described by linear models.* Nat Biomed Eng 2024. https://www.nature.com/articles/s41551-023-01117-y
13. Widge AS, Bilge MT, Montana R, et al. *Electroencephalographic biomarkers for treatment response prediction in major depressive illness: a meta-analysis.* Am J Psychiatry 2019;176:44–56. https://pubmed.ncbi.nlm.nih.gov/30278789/
14. van der Vinne N, Vollebregt MA, van Putten MJAM, Arns M. *Frontal alpha asymmetry as a diagnostic marker in depression: fact or fiction? A meta-analysis.* NeuroImage Clin 2017;16:79–87.
15. Jansen BH, Rit VG. *Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns.* Biol Cybern 1995;73:357–366.
16. Pinotsis DA, Fitzgerald S, See C, Sementsova A, Widge AS. *Toward biophysical markers of depression vulnerability.* Front Psychiatry 2022;13:938694. https://pmc.ncbi.nlm.nih.gov/articles/PMC9622949/
16a. Kerr CC, Kemp AH, Rennie CJ, Robinson PA. *Thalamocortical changes in major depression probed by deconvolution and physiology-based modeling.* NeuroImage 2011;54:2672–2682. https://pubmed.ncbi.nlm.nih.gov/21073966/
17. *Systematic review and meta-analysis: Impact of unipolar depression on P300 amplitude and latency.* Neurosci Biobehav Rev 2025. https://pubmed.ncbi.nlm.nih.gov/40412458
18. Adams RA, Pinotsis D, Tsirlis K, et al. *Computational modeling of electroencephalography and functional magnetic resonance imaging paradigms indicates a consistent loss of pyramidal cell synaptic gain in schizophrenia.* Biol Psychiatry 2022;91:202–215. https://pmc.ncbi.nlm.nih.gov/articles/PMC8654393/
19. Keren H, O'Callaghan G, Vidal-Ribas P, et al. *Reward processing in depression: a conceptual and meta-analytic review across fMRI and EEG studies.* Am J Psychiatry 2018;175:1111–1120. https://psychiatryonline.org/doi/10.1176/appi.ajp.2018.17101124
20. Rennie CJ, Robinson PA, Wright JJ. *Unified neurophysical model of EEG spectra and evoked potentials.* Biol Cybern 2002;86:457–471. https://link.springer.com/article/10.1007/s00422-002-0310-9
21. David O, Kiebel SJ, Harrison LM, et al. *Dynamic causal modeling of evoked responses in EEG and MEG.* NeuroImage 2006;30:1255–1272. Garrido MI, Kilner JM, Kiebel SJ, Friston KJ. *Evoked brain responses are generated by feedback loops.* PNAS 2007;104:20961–20966.
22. Moran RJ, Stephan KE, Seidenbecher T, et al. *Dynamic causal models of steady-state responses.* NeuroImage 2009;44:796–811.
23. Gilbert JR, …, Zarate CA. *Glutamatergic signaling drives ketamine-mediated response in depression: evidence from dynamic causal modeling.* Int J Neuropsychopharmacol 2018;21:740–747. https://academic.oup.com/ijnp/article/21/8/740/4969995 ; *Ketamine and attentional bias toward emotional faces: DCM of MEG connectivity in treatment-resistant depression.* 2021. https://pubmed.ncbi.nlm.nih.gov/34220581/
24. Momi D, Wang Z, Griffiths JD. *TMS-evoked responses are driven by recurrent large-scale network dynamics.* eLife 2023;12:e83232. https://elifesciences.org/articles/83232
25. Hofsähs T, Pille M, Kern L, Negi A, Meier JM, Ritter P. *The Virtual Brain links transcranial magnetic stimulation evoked potentials and inhibitory neurotransmitter changes in major depressive disorder.* Imaging Neurosci 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC12961309/
26. Cavanagh JF, Bismark AW, Frank MJ, Allen JJB. *Multiple dissociations between comorbid depression and anxiety on reward and punishment processing: evidence from computationally informed EEG.* Comput Psychiatry 2019;3:1–17. https://pmc.ncbi.nlm.nih.gov/articles/PMC6515849/ — data: https://openneuro.org/datasets/ds003474/versions/1.1.0
27. Tenke CE, et al. *Demonstrating test-retest reliability of electrophysiological measures for healthy adults in a multisite study of biomarkers of antidepressant treatment response* (EMBARC). Psychophysiology 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5181116/
28. Other datasets: MODMA (Cai et al., Sci Data 2022) https://www.nature.com/articles/s41597-022-01211-x ; ds005356 https://github.com/OpenNeuroDatasets/ds005356 ; HBN-EEG https://openneuro.org/datasets/ds005507/versions/1.0.1 ; Mumtaz figshare 4244171.
