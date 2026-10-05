#!/usr/bin/env python3
"""Extension experiments for the RD-TabXAI / PhenoFuse-LTR study (all leakage-safe, real HPO).

Adds, on the rigorous leave-target-out benchmark (disease holdout):
  E1  Stronger/fairer baselines on the SAME features:
        - rank-aggregation (unsupervised mean of z-scored similarities)
        - pointwise logistic regression
        - pointwise random forest
      to show the win comes from learned list-wise fusion, not merely "using ML".
  E2  Clinical-realism robustness: term dropout (incompleteness), spurious terms
      (mis-phenotyping), generalisation to parents (imprecision). Model trained on clean
      annotations, evaluated on progressively noisier test queries.
  E3  Rigorous multi-method statistics: Friedman omnibus + Nemenyi critical difference.
  E4  Explainability: SHAP attributions for the ranker (global mean|SHAP|).

    PYTHONPATH=src python scripts/run_extensions.py
"""
import sys, json, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdtabxai.io_utils import load_config, set_seed, ensure_dir, LOG
from rdtabxai.data.hpo import ancestor_map
from rdtabxai.evaluation import ltr
from rdtabxai.evaluation.splits import make_split
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import friedmanchisquare, rankdata, studentized_range

SIM_COLS = ["jaccard","cosine_bin","phrank","sum_ic_shared","max_ic_shared","resnik_bma","lin_bma"]

def rank_eval_from_scores(df_te, score):
    d = df_te.copy(); d["score"] = score
    return ltr._rank_eval_df(d)

def per_disease_rr(m):
    d = pd.DataFrame(m["per_query"]); return d.groupby("ORPHA_ID")["rr"].mean()

def main():
    cfg = load_config(ROOT/"configs/real_hpo.yaml"); set_seed(cfg["seed"]); seed=cfg["seed"]
    pdir = ROOT/cfg["paths"]["data_processed"]; tdir=ensure_dir(ROOT/cfg["paths"]["results_tables"])
    profiles=pd.read_csv(pdir/"profiles.csv"); pairs=pd.read_csv(pdir/"ranking_pairs.csv")
    disease_terms=json.loads((pdir/"disease_terms.json").read_text())
    anc=ancestor_map(pd.read_csv(pdir/"hpo_terms.csv"))
    g2p=pd.read_csv(pdir/"genes_to_phenotype.csv")
    gene_to_terms=g2p.groupby("gene_symbol")["hpo_id"].apply(set).to_dict()
    gdt={}
    for (g,d),sub in g2p.groupby(["gene_symbol","disease_id"]): gdt.setdefault(g,{})[d]=set(sub["hpo_id"])

    # ---- clean LTO features (disease holdout), computed once ----
    df,tr,te = ltr.compute_features(pairs,profiles,gene_to_terms,disease_terms,anc,cfg,
                                    strategy="disease_holdout",gene_disease_terms=gdt,leave_target_out=True)
    feats = ltr.ALL_FEATURES
    Xtr,ytr = df.loc[tr,feats].values, df.loc[tr,"causal"].values
    Xte = df.loc[te,feats].values
    df_te = df.loc[te]

    # proposed LTR
    prop = ltr._fit_lambdarank(df,tr,te,feats,seed)

    # ===== E1 stronger baselines on same features =====
    LOG.info("E1 stronger baselines")
    # rank-aggregation: mean of z-scored similarity columns (fit z on train only)
    mu=df.loc[tr,SIM_COLS].mean(); sd=df.loc[tr,SIM_COLS].std().replace(0,1)
    ragg=((df_te[SIM_COLS]-mu)/sd).mean(axis=1).values
    m_ragg=rank_eval_from_scores(df_te,ragg)
    # pointwise logistic regression
    sc=StandardScaler().fit(Xtr)
    lr=LogisticRegression(max_iter=1000,class_weight="balanced").fit(sc.transform(Xtr),ytr)
    m_lr=rank_eval_from_scores(df_te, lr.predict_proba(sc.transform(Xte))[:,1])
    # pointwise random forest
    rf=RandomForestClassifier(n_estimators=400,min_samples_leaf=5,random_state=seed,
                              class_weight="balanced",n_jobs=-1).fit(Xtr,ytr)
    m_rf=rank_eval_from_scores(df_te, rf.predict_proba(Xte)[:,1])
    # single best similarity baselines (from columns)
    base={b:ltr._rank_by_column(df_te,b,cfg) for b in ["resnik_bma","phrank","lin_bma","jaccard","cosine_bin"]}

    methods={"PhenoFuse-LTR":prop,"Rank-aggregation":m_ragg if isinstance(m_ragg,dict) else m_ragg,
             "Logistic-regression":m_lr,"Random-forest":m_rf,
             "Resnik-BMA":base["resnik_bma"],"Phrank-style":base["phrank"],"Lin-BMA":base["lin_bma"],
             "Jaccard":base["jaccard"],"Cosine":base["cosine_bin"]}
    rowsE1=[]
    for n,m in methods.items():
        rowsE1.append({"method":n,"MRR":round(m["mrr"],3),"Hit@1":round(m["hit@1"],3),
                       "Hit@10":round(m["hit@10"],3),"NDCG@10":round(m["ndcg@10"],3),"n_queries":m["n_queries"]})
    pd.DataFrame(rowsE1).sort_values("MRR",ascending=False).to_csv(tdir/"E1_stronger_baselines.csv",index=False)

    # ===== E3 Friedman + Nemenyi across methods (per-disorder MRR) =====
    LOG.info("E3 Friedman/Nemenyi")
    prr={n:per_disease_rr(m) for n,m in methods.items()}
    common=set.intersection(*[set(s.index) for s in prr.values()])
    common=sorted(common)
    M=pd.DataFrame({n:prr[n].loc[common].values for n in methods})  # rows=disorders, cols=methods
    # Friedman
    stat,p=friedmanchisquare(*[M[n].values for n in methods])
    # average ranks (rank 1 = best => rank by -MRR)
    ranks=np.array([rankdata(-row) for row in M.values])  # per disorder, 1=best
    avg_ranks=ranks.mean(axis=0)
    k=len(methods); N=len(common)
    q=studentized_range.ppf(0.95,k,np.inf)/np.sqrt(2)
    CD=q*np.sqrt(k*(k+1)/(6*N))
    cd_rows=[{"method":n,"avg_rank":round(r,3)} for n,r in zip(methods,avg_ranks)]
    pd.DataFrame(cd_rows).sort_values("avg_rank").to_csv(tdir/"E3_avg_ranks.csv",index=False)
    json.dump({"friedman_stat":float(stat),"friedman_p":float(p),"k":k,"N":N,
               "critical_difference_0.05":float(CD),
               "avg_ranks":{n:float(r) for n,r in zip(methods,avg_ranks)}},
              open(tdir/"E3_friedman_nemenyi.json","w"),indent=2)

    # ===== E4 SHAP global attribution for the ranker =====
    LOG.info("E4 SHAP")
    ranker=lgb.LGBMRanker(objective="lambdarank",n_estimators=500,num_leaves=31,learning_rate=0.05,
                          min_child_samples=20,subsample=0.9,colsample_bytree=0.9,random_state=seed,verbose=-1,
                          deterministic=True,force_row_wise=True,num_threads=1)
    g=df.loc[tr].sort_values("profile_id").groupby("profile_id").size().values
    tt=df.loc[tr].sort_values("profile_id")
    ranker.fit(tt[feats].values,tt["causal"].values,group=g)
    import shap
    expl=shap.TreeExplainer(ranker)
    sv=expl.shap_values(df_te[feats].values)
    mean_abs=np.abs(sv).mean(axis=0)
    pd.DataFrame({"feature":feats,"mean_abs_shap":np.round(mean_abs,4)}).sort_values(
        "mean_abs_shap",ascending=False).to_csv(tdir/"E4_shap_global.csv",index=False)

    # ===== E2 noise robustness =====
    LOG.info("E2 noise robustness")
    # rebuild ic/mica + gene LTO terms to recompute features for perturbed test profiles
    train_orpha=set(df.loc[tr,"ORPHA_ID"]); ic,mica=ltr._build_ic_and_cache(disease_terms,train_orpha,anc)
    ltr._ANC={t:anc.get(t,{t}) for o in disease_terms for t in disease_terms[o]}
    vocab=sorted({t for o in disease_terms for t in disease_terms[o]})
    def cap(ts):
        ts=list(ts)
        # same deterministic tie-break as ltr.compute_features (IC desc, then term ID)
        return set(sorted(ts,key=lambda t:(-ic.get(t,0.0),t))[:60]) if len(ts)>60 else set(ts)
    def gene_lto(g,src):
        dd=gdt.get(g,{}); acc=set()
        for d,ts in dd.items():
            if str(d)!=str(src): acc|=ts
        return cap(acc)
    pterms=profiles.set_index("profile_id")["terms"].apply(lambda s:set(s.split("|"))).to_dict()
    te_pairs=df.loc[te,["profile_id","ORPHA_ID","candidate_gene","causal"]].copy()
    # train proposed once on clean (already have ranker); best baseline = resnik column recomputed under noise
    rng=np.random.RandomState(seed)
    def perturb(P,kind,level):
        P=set(P)
        if kind=="dropout":
            return {t for t in sorted(P) if rng.random()>level}  # sorted: hash-order independent
        if kind=="spurious":
            add=set(rng.choice(vocab,int(level),replace=False)) if level>0 else set()
            return P|add
        if kind=="generalize":
            out=set()
            for t in sorted(P):
                if rng.random()<level:
                    anc_t=sorted(a for a in anc.get(t,{t}) if a!=t)
                    out.add(rng.choice(anc_t) if anc_t else t)
                else: out.add(t)
            return out
        return P
    def eval_noise(kind,level):
        # recompute features for perturbed test profiles
        prof_pert={pid:perturb(pterms.get(pid,set()),kind,level) for pid in te_pairs["profile_id"].unique()}
        rows=[]
        for r in te_pairs.itertuples():
            P=prof_pert[r.profile_id]; G=gene_lto(r.candidate_gene,r.ORPHA_ID)
            rows.append(ltr._pair_features(P,G,ic,mica))
        F=pd.DataFrame(rows,index=te_pairs.index)
        d=pd.concat([te_pairs.reset_index(drop=True),F.reset_index(drop=True)],axis=1)
        prop_s=ranker.predict(d[feats].values)
        mp=rank_eval_from_scores(d,prop_s)
        mb=ltr._rank_by_column(d,"resnik_bma",cfg)
        # rank-agg under noise
        ra=((d[SIM_COLS]-mu.values)/sd.values).mean(axis=1).values
        mr=rank_eval_from_scores(d,ra)
        return mp["mrr"],mb["mrr"],mr["mrr"]
    noise_rows=[]
    for kind,levels in [("dropout",[0.0,0.2,0.4,0.6]),("spurious",[0,2,4,6]),("generalize",[0.0,0.3,0.6])]:
        for lv in levels:
            p_,b_,r_=eval_noise(kind,lv)
            noise_rows.append({"noise":kind,"level":lv,"PhenoFuse-LTR":round(p_,3),
                               "Rank-aggregation":round(r_,3),"Resnik-BMA":round(b_,3)})
            LOG.info("  %s=%s proposed=%.3f ragg=%.3f resnik=%.3f",kind,lv,p_,r_,b_)
    pd.DataFrame(noise_rows).to_csv(tdir/"E2_noise_robustness.csv",index=False)

    # ---- print summary ----
    print("\n===== E1 stronger baselines (rigorous, disease holdout) =====")
    print(pd.read_csv(tdir/"E1_stronger_baselines.csv").to_string(index=False))
    print("\n===== E3 Friedman/Nemenyi =====")
    j=json.load(open(tdir/"E3_friedman_nemenyi.json"))
    print(f"Friedman chi2={j['friedman_stat']:.1f} p={j['friedman_p']:.2e}  k={j['k']} N={j['N']}  CD(0.05)={j['critical_difference_0.05']:.3f}")
    print(pd.read_csv(tdir/"E3_avg_ranks.csv").to_string(index=False))
    print("\n===== E4 SHAP global =====")
    print(pd.read_csv(tdir/"E4_shap_global.csv").to_string(index=False))
    print("\n===== E2 noise robustness =====")
    print(pd.read_csv(tdir/"E2_noise_robustness.csv").to_string(index=False))

if __name__=="__main__":
    main()
