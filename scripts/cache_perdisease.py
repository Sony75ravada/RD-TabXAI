import sys,json; from pathlib import Path; import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from rdtabxai.io_utils import load_config,set_seed,ensure_dir
from rdtabxai.data.hpo import ancestor_map
from rdtabxai.evaluation import ltr
cfg=load_config(ROOT/"configs/real_hpo.yaml"); set_seed(cfg["seed"]); seed=cfg["seed"]
pdir=ROOT/cfg["paths"]["data_processed"]; tdir=ensure_dir(ROOT/cfg["paths"]["results_tables"])
profiles=pd.read_csv(pdir/"profiles.csv"); pairs=pd.read_csv(pdir/"ranking_pairs.csv")
disease_terms=json.loads((pdir/"disease_terms.json").read_text())
anc=ancestor_map(pd.read_csv(pdir/"hpo_terms.csv"))
g2p=pd.read_csv(pdir/"genes_to_phenotype.csv")
gene_to_terms=g2p.groupby("gene_symbol")["hpo_id"].apply(set).to_dict()
gdt={}
for (g,d),sub in g2p.groupby(["gene_symbol","disease_id"]): gdt.setdefault(g,{})[d]=set(sub["hpo_id"])
df,tr,te=ltr.compute_features(pairs,profiles,gene_to_terms,disease_terms,anc,cfg,
    strategy="disease_holdout",gene_disease_terms=gdt,leave_target_out=True)
feats=ltr.ALL_FEATURES
prop=ltr._fit_lambdarank(df,tr,te,feats,seed)
res=ltr._rank_by_column(df.loc[te],"resnik_bma",cfg)
pp=pd.DataFrame(prop["per_query"]).groupby("ORPHA_ID")["rr"].mean()
rr=pd.DataFrame(res["per_query"]).groupby("ORPHA_ID")["rr"].mean()
out=pd.DataFrame({"ORPHA_ID":pp.index,"proposed":pp.values})
out=out.merge(pd.DataFrame({"ORPHA_ID":rr.index,"resnik":rr.values}),on="ORPHA_ID")
out["delta"]=out["proposed"]-out["resnik"]
out.sort_values("delta").to_csv(tdir/"E5_per_disease_rr.csv",index=False)
print("saved E5_per_disease_rr.csv  n=",len(out),"mean delta=",round(out.delta.mean(),3),
      "frac proposed>=resnik=",round((out.delta>=0).mean(),3))
