#!/usr/bin/env python3
"""Publication figure system v2 for RD-TabXAI / PhenoFuse-LTR.
One consistent visual language across all figures; vector PDF + PNG at 300 dpi.
Reads only from results_real/tables/*.csv (and E1-E5 extensions). No fabricated values.
    python scripts/make_figures_v2.py
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
T = ROOT/"results_real/tables"
OUT = ROOT/"results_real/figures"; OUT.mkdir(parents=True, exist_ok=True)
PAPER = ROOT/"results_real/figures"  # single output folder

# ---------------------------------------------------------------- design system
INK   = "#1B2430"      # text / axes
MUTE  = "#5C6B7A"      # secondary text
GRID  = "#E4E8EC"
HERO  = "#1F4E79"      # PhenoFuse-LTR (learned list-wise fusion)
LEARN = ["#3D7CB8", "#6FA8DC"]        # other learned (RF, LR)
CLASS = ["#E07B39", "#EBA45E", "#F2C888"]  # classical IC (Resnik, Phrank, Lin)
NEUT  = ["#8C9AA6", "#AFB9C2", "#C9D0D6"]   # overlap / averaging
ACCENT_POS = "#2E7D6B"; ACCENT_NEG = "#B5546B"

METHOD_COLOR = {
    "PhenoFuse-LTR": HERO,
    "Random-forest": LEARN[0], "Logistic-regression": LEARN[1],
    "Resnik-BMA": CLASS[0], "Phrank-style": CLASS[1], "Lin-BMA": CLASS[2],
    "Rank-aggregation": NEUT[0], "Cosine": NEUT[1], "Jaccard": NEUT[2],
}
SHORT = {"PhenoFuse-LTR":"PhenoFuse-LTR","Random-forest":"Random forest",
         "Logistic-regression":"Logistic reg.","Resnik-BMA":"Resnik-BMA",
         "Phrank-style":"Phrank","Lin-BMA":"Lin-BMA","Rank-aggregation":"Rank-agg.",
         "Cosine":"Cosine","Jaccard":"Jaccard"}

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.linewidth": 0.9, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 13.5, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "legend.fontsize": 12, "legend.frameon": False, "figure.facecolor": "white",
    "axes.facecolor": "white",
})
def style(ax, grid="y"):
    ax.tick_params(length=3, width=0.8)
    if grid in ("y","both"): ax.yaxis.grid(True, color=GRID, lw=0.9, zorder=0)
    if grid in ("x","both"): ax.xaxis.grid(True, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(OUT/f"{name}.{ext}")
        if ext=="pdf": fig.savefig(PAPER/f"{name}.pdf")
    plt.close(fig); print("  ->", name)

def mrr_ci(s):
    m=re.match(r"([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", s); 
    return (float(m.group(1)),float(m.group(2)),float(m.group(3))) if m else (float(s.split()[0]),None,None)

# ---------------------------------------------------------------- load tables
naive={(r["split"],r["method"]):r for _,r in pd.read_csv(T/"T4_naive_benchmark.csv").iterrows()}
rig  ={(r["split"],r["method"]):r for _,r in pd.read_csv(T/"T5_rigorous_benchmark.csv").iterrows()}
ABL=pd.read_csv(T/"T7_ablation.csv"); IMP=pd.read_csv(T/"T8_feature_importance.csv")
PERM=pd.read_csv(T/"T11_permutation_validity.csv"); STAB=pd.read_csv(T/"T12_test_subset_stability.csv")
E1=pd.read_csv(T/"E1_stronger_baselines.csv"); E2=pd.read_csv(T/"E2_noise_robustness.csv")
E3=json.load(open(T/"E3_friedman_nemenyi.json")); E4=pd.read_csv(T/"E4_shap_global.csv")
E5=pd.read_csv(T/"E5_per_disease_rr.csv")
BMETH=["PhenoFuse-LTR","Resnik-BMA","Phrank-style","Lin-BMA","Cosine","Jaccard"]

# ================================================================ F1 pipeline
def f1():
    fig,ax=plt.subplots(figsize=(11.4,3.0)); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,30)
    stages=[("Human Phenotype\nOntology (2025-01)","#EAF0F6"),
            ("Leakage-safe cohort\n400 OMIM disorders","#EAF0F6"),
            ("Simulated profiles\n+ candidate pools","#EAF0F6"),
            ("Multi-view features\noverlap · IC · BMA","#DCE7F1"),
            ("PhenoFuse-LTR\n(LightGBM ranker)","#C9DCEC"),
            ("Ranked genes\n+ XAI + audit","#EAF0F6")]
    w,h,gap=15.0,12.5,1.8; x=0.5; y=8.5
    for i,(txt,fc) in enumerate(stages):
        ec=HERO if i in (3,4) else "#9FB3C6"
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",
                     lw=1.7 if i in (3,4) else 1.1, ec=ec, fc=fc, zorder=3))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=11,
                color=INK,weight="bold" if i in (3,4) else "normal",zorder=4)
        if i<len(stages)-1:
            ax.add_patch(FancyArrowPatch((x+w,y+h/2),(x+w+gap,y+h/2),
                         arrowstyle="-|>",mutation_scale=13,lw=1.4,color=MUTE,zorder=2))
        x+=w+gap
    ax.text(50,26.0,"Leave-target-out evaluation   ·   train-only information content   ·   permutation-audited",
            ha="center",va="center",fontsize=11.5,style="italic",color=MUTE)
    save(fig,"F1_pipeline")

# ================================================================ F2 naive vs LTO
def f2():
    meth=BMETH; x=np.arange(len(meth)); bw=0.38
    nv=[mrr_ci(naive[("Disease holdout",m)]["MRR (95% CI)"])[0] for m in meth]
    lt=[mrr_ci(rig[("Disease holdout",m)]["MRR (95% CI)"])[0] for m in meth]
    fig,ax=plt.subplots(figsize=(8.4,4.7)); style(ax)
    b1=ax.bar(x-bw/2,nv,bw,label="Naive (annotation recovery)",color="#C9D0D6",ec="white",lw=.6,zorder=3)
    b2=ax.bar(x+bw/2,lt,bw,label="Rigorous (leave-target-out)",color=HERO,ec="white",lw=.6,zorder=3)
    for xi,v in zip(x-bw/2,nv): ax.text(xi,v+.012,f"{v:.2f}",ha="center",fontsize=9.8,color=MUTE)
    for xi,v in zip(x+bw/2,lt): ax.text(xi,v+.012,f"{v:.2f}",ha="center",fontsize=9.8,color=INK,weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in meth],rotation=18,ha="right")
    ax.set_ylabel("Mean reciprocal rank"); ax.set_ylim(0,1.08)
    ax.set_title("Naive saturation collapses under rigorous evaluation",loc="left")
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.26),ncol=2,frameon=False,
              handlelength=1.3,columnspacing=2.4,fontsize=11.2)
    save(fig,"F2_headline_naive_vs_lto")

# ================================================================ F3 MRR CI two-panel
def f3():
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.4),sharex=True)
    for ax,(split,lab) in zip(axes,[("Disease holdout","Disease holdout"),("Gene holdout","Gene holdout")]):
        style(ax,grid="x")
        rows=[(m,)+mrr_ci(rig[(split,m)]["MRR (95% CI)"]) for m in BMETH]
        rows=sorted(rows,key=lambda r:r[1])
        y=np.arange(len(rows))
        for yi,(m,v,lo,hi) in zip(y,rows):
            c=METHOD_COLOR[m]
            ax.barh(yi,v,color=c,ec="white",height=.62,zorder=3,
                    alpha=1 if m=="PhenoFuse-LTR" else .92)
            if lo is not None:
                ax.plot([lo,hi],[yi,yi],color=INK,lw=1.3,zorder=4)
                # end caps for clarity
                for xx in (lo,hi): ax.plot([xx,xx],[yi-.13,yi+.13],color=INK,lw=1.1,zorder=4)
            # place the label clear of the whisker end so they never overlap
            lx=(hi if lo is not None else v)+.022
            ax.text(lx,yi,f"{v:.3f}",va="center",fontsize=10.8,
                    weight="bold" if m=="PhenoFuse-LTR" else "normal")
        ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in [r[0] for r in rows]])
        ax.set_xlim(0,1.12); ax.set_title(lab,loc="left",fontsize=13.7)
        ax.set_xlabel("Mean reciprocal rank (95% CI)")
    fig.suptitle("Rigorous leave-target-out: PhenoFuse-LTR separates from every baseline",
                 x=.012,ha="left",fontsize=14.7,weight="bold")
    fig.tight_layout(rect=[0,0,1,.95])
    save(fig,"F3_mrr_ci")

# ================================================================ F4 Hit@k curves
def f4():
    ks=[1,5,10]; colkey={1:"Hit@1",5:"Hit@5",10:"Hit@10"}
    fig,ax=plt.subplots(figsize=(7.8,4.6)); style(ax)
    for m in BMETH:
        ys=[float(rig[("Disease holdout",m)][colkey[k]]) for k in ks]
        ax.plot(ks,ys,marker="o",ms=6,lw=2.4 if m=="PhenoFuse-LTR" else 1.6,
                color=METHOD_COLOR[m],label=SHORT[m],zorder=5 if m=="PhenoFuse-LTR" else 3,
                alpha=1 if m=="PhenoFuse-LTR" else .9)
    ax.set_xticks(ks); ax.set_xlabel("k"); ax.set_ylabel("Hit@k  (fraction with causal gene in top-k)")
    ax.set_ylim(.3,1.02); ax.set_title("Hit@k, disease holdout",loc="left")
    ax.legend(loc="lower right",ncol=2)
    save(fig,"F4_hitk_curves")

# ================================================================ F5 ablation
def f5():
    cum=ABL[ABL["Feature groups"].isin(["overlap","overlap+ic_shared","overlap+ic_shared+bma"])]
    labels=["overlap","+ ic_shared","+ bma"]; vals=cum["MRR"].tolist()
    logo=ABL[ABL["Feature groups"].str.startswith("all ")]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,4.3),gridspec_kw={"width_ratios":[1.25,1]})
    style(a1)
    cols=["#C9D0D6","#6FA8DC",HERO]
    bars=a1.bar(labels,vals,color=cols,ec="white",zorder=3,width=.62)
    base=0
    for i,(l,v) in enumerate(zip(labels,vals)):
        # value label sits clearly above the bar top
        a1.text(i,v+.030,f"{v:.3f}",ha="center",weight="bold",fontsize=11.2,zorder=6)
        if i>0:
            inc=vals[i]-vals[i-1]
            # increment shown to the LEFT of the bar centre, below the value label,
            # so it can never sit on top of the bar's value
            # increment printed inside the bar (white), so it can never collide with labels
            a1.text(i,v-0.07,f"+{inc:.3f}",ha="center",va="center",
                    fontsize=10.6,color="white",weight="bold",zorder=6)
    a1.set_ylim(0,1.02); a1.set_ylabel("Mean reciprocal rank")
    a1.set_title("Cumulative feature views",loc="left")
    style(a2,grid="x")
    full=float(cum["MRR"].iloc[-1]); lg=logo.set_index("Feature groups")  # full model, from T7
    order=["all − overlap","all − ic_shared","all − bma"]; nice=["– overlap","– ic_shared","– bma"]
    costs=[full-lg.loc[k,"MRR"] for k in order]
    y=np.arange(len(order))
    a2.barh(y,costs,color=["#B5546B","#E0A0AE","#E0A0AE"],ec="white",height=.6,zorder=3)
    for yi,c in zip(y,costs): a2.text(c+.0008,yi,f"−{c:.3f}",va="center",fontsize=10.8)
    a2.set_yticks(y); a2.set_yticklabels(nice); a2.invert_yaxis()
    a2.set_xlabel("MRR cost when view removed"); a2.set_title("Leave-one-view-out",loc="left")
    a2.set_xlim(0,max(costs)*1.35)
    fig.tight_layout(); save(fig,"F5_ablation")

# ================================================================ F6 importance (split counts)
def f6():
    d=IMP.copy(); d.columns=["feature","imp"]; d=d.sort_values("imp")
    fig,ax=plt.subplots(figsize=(7.6,4.8)); style(ax,grid="x")
    grp={"resnik_bma":CLASS[0],"lin_bma":CLASS[2],"phrank":CLASS[1],"sum_ic_shared":CLASS[1],
         "max_ic_shared":CLASS[1],"mean_ic_profile":CLASS[1],"jaccard":NEUT[1],"cosine_bin":NEUT[1],
         "n_gene":HERO,"n_profile":NEUT[2],"n_shared":NEUT[2]}
    ax.barh(d["feature"],d["imp"],color=[grp.get(f,MUTE) for f in d["feature"]],ec="white",height=.66,zorder=3)
    for i,(f,v) in enumerate(zip(d["feature"],d["imp"])): ax.text(v+18,i,str(int(v)),va="center",fontsize=10.6)
    ax.set_xlabel("LightGBM split count"); ax.set_title("Feature importance (model-internal)",loc="left")
    ax.set_xlim(0,d["imp"].max()*1.12)
    save(fig,"F6_feature_importance")

# ================================================================ F7 per-disease delta
def f7():
    d=E5.sort_values("delta").reset_index(drop=True); x=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(9.2,4.2)); style(ax)
    ax.bar(x,d["delta"],color=[ACCENT_POS if v>=0 else ACCENT_NEG for v in d["delta"]],width=1.0,zorder=3)
    ax.axhline(0,color=INK,lw=.9)
    mu=d["delta"].mean(); ax.axhline(mu,color=HERO,lw=1.6,ls="--",zorder=4,
                                     label=f"mean Δ = {mu:.3f}")
    frac=(d["delta"]>=0).mean()
    ax.set_xlabel("Held-out disorders (sorted by Δ)"); ax.set_ylabel("Δ MRR  (PhenoFuse − Resnik)")
    ax.set_title(f"Per-disorder improvement is broadly distributed  ({frac*100:.0f}% of disorders ≥ 0)",loc="left")
    ax.set_xlim(-.5,len(d)-.5); ax.legend(loc="upper left")
    save(fig,"F7_per_disease_delta")

# ================================================================ F8 overall ranking
def f8():
    rows=[]
    for m in BMETH:
        d=mrr_ci(rig[("Disease holdout",m)]["MRR (95% CI)"]); g=mrr_ci(rig[("Gene holdout",m)]["MRR (95% CI)"])
        rows.append((m,(d[0]+g[0])/2,min(d[1],g[1]),max(d[2],g[2])))
    rows=sorted(rows,key=lambda r:r[1])
    fig,ax=plt.subplots(figsize=(8.0,4.3)); style(ax,grid="x")
    y=np.arange(len(rows))
    for yi,(m,v,lo,hi) in zip(y,rows):
        ax.barh(yi,v,color=METHOD_COLOR[m],ec="white",height=.62,zorder=3)
        ax.plot([lo,hi],[yi,yi],color=INK,lw=1.3,zorder=4)
        for xx in (lo,hi): ax.plot([xx,xx],[yi-.13,yi+.13],color=INK,lw=1.1,zorder=4)
        ax.text(hi+.022,yi,f"{v:.3f}",va="center",fontsize=10.8,
                weight="bold" if m=="PhenoFuse-LTR" else "normal")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[r[0]] for r in rows])
    ax.set_xlim(0,1.14); ax.set_xlabel("Mean reciprocal rank (mean of disease & gene holdout)")
    ax.set_title("Overall ranking across both splits",loc="left")
    save(fig,"F8_overall_ranking")

# ================================================================ F9 permutation
def f9():
    labs=["Real labels","Permuted\ntrain labels","Theoretical\nrandom"]
    vals=[float(PERM.iloc[0]["MRR"]),float(PERM.iloc[1]["MRR"]),float(PERM.iloc[2]["MRR"])]
    fig,ax=plt.subplots(figsize=(6.4,4.4)); style(ax)
    bars=ax.bar(labs,vals,color=[HERO,"#B5546B","#C9D0D6"],ec="white",width=.6,zorder=3)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.012,f"{v:.3f}",ha="center",weight="bold",fontsize=11.7)
    ax.axhspan(vals[2]-.02,vals[2]+.02,color="#C9D0D6",alpha=.35,zorder=1)
    ax.annotate("collapses to chance\nwhen labels permuted →",xy=(1,vals[1]+.03),xytext=(1.5,.55),
                fontsize=11,color=MUTE,ha="center",
                arrowprops=dict(arrowstyle="-|>",color=MUTE,lw=1.1))
    ax.set_ylabel("Mean reciprocal rank"); ax.set_ylim(0,.95)
    ax.set_title("Label-permutation leakage audit",loc="left")
    save(fig,"F9_permutation_validity")

# ================================================================ E1 stronger baselines
def e1():
    d=E1.sort_values("MRR").reset_index(drop=True); y=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(8.4,4.7)); style(ax,grid="x")
    for yi,r in zip(y,d.itertuples()):
        c=METHOD_COLOR.get(r.method,MUTE)
        ax.barh(yi,r.MRR,color=c,ec="white",height=.66,zorder=3)
        ax.text(r.MRR+.012,yi,f"{r.MRR:.3f}",va="center",fontsize=10.8,
                weight="bold" if r.method=="PhenoFuse-LTR" else "normal")
    ax.set_yticks(y); ax.set_yticklabels([SHORT.get(m,m) for m in d["method"]])
    ax.set_xlim(0,1.0); ax.set_xlabel("Mean reciprocal rank (rigorous, disease holdout)")
    ax.set_title("Learned list-wise fusion vs. pointwise ML, averaging, and single similarities",loc="left",fontsize=13.2)
    handles=[Line2D([0],[0],marker="s",ls="",mec="white",mfc=HERO,ms=10,label="Learned list-wise (proposed)"),
             Line2D([0],[0],marker="s",ls="",mec="white",mfc=LEARN[0],ms=10,label="Learned pointwise"),
             Line2D([0],[0],marker="s",ls="",mec="white",mfc=CLASS[0],ms=10,label="Classical similarity"),
             Line2D([0],[0],marker="s",ls="",mec="white",mfc=NEUT[0],ms=10,label="Overlap / averaging")]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.16),
              ncol=2,frameon=False,handlelength=1.2,columnspacing=1.8,fontsize=11)
    fig.tight_layout()
    save(fig,"E1_stronger_baselines")

# ================================================================ E2 noise robustness
def e2():
    fig,axes=plt.subplots(1,3,figsize=(12.2,3.9))
    panels=[("dropout","Term dropout (incompleteness)","fraction terms removed"),
            ("spurious","Spurious terms (mis-phenotyping)","# random terms added"),
            ("generalize","Generalisation (imprecision)","fraction terms generalised")]
    series=[("PhenoFuse-LTR",HERO,"o"),("Rank-aggregation",NEUT[0],"s"),("Resnik-BMA",CLASS[0],"^")]
    for ax,(kind,title,xl) in zip(axes,panels):
        style(ax); sub=E2[E2["noise"]==kind]
        for nm,c,mk in series:
            ax.plot(sub["level"],sub[nm],marker=mk,ms=5.5,lw=2.2 if nm=="PhenoFuse-LTR" else 1.5,
                    color=c,label=SHORT[nm],zorder=5 if nm=="PhenoFuse-LTR" else 3)
        ax.set_title(title,loc="left",fontsize=12.5); ax.set_xlabel(xl)
        ax.set_ylim(.45,.88)
    axes[0].set_ylabel("Mean reciprocal rank")
    h,l=axes[2].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.02))
    fig.suptitle("Robustness to clinical-realism noise (model trained on clean annotations)",
                 x=.012,ha="left",fontsize=14.2,weight="bold")
    fig.tight_layout(rect=[0,0.08,1,.94]); save(fig,"E2_noise_robustness")

# ================================================================ E3 Nemenyi CD diagram
def e3():
    ranks=E3["avg_ranks"]; CD=E3["critical_difference_0.05"]
    items=sorted(ranks.items(),key=lambda kv:kv[1])  # best (low) first
    names=[k for k,_ in items]; rv=[v for _,v in items]
    k=len(names); lo=int(np.floor(min(rv))); hi=int(np.ceil(max(rv)))
    fig,ax=plt.subplots(figsize=(10.2,4.6)); ax.axis("off")
    ax.set_xlim(lo-0.4,hi+0.4); ax.set_ylim(0,12)
    axis_y=9.0
    # title (top, clear of everything)
    ax.text(lo-0.4,11.5,"Friedman–Nemenyi critical-difference diagram",
            fontsize=14.7,weight="bold",color=INK,ha="left")
    # rank axis with ticks above
    ax.plot([lo,hi],[axis_y,axis_y],color=INK,lw=1.4)
    for t in range(lo,hi+1):
        ax.plot([t,t],[axis_y,axis_y+0.16],color=INK,lw=1.1); ax.text(t,axis_y+0.42,str(t),ha="center",fontsize=11.2)
    ax.text((lo+hi)/2,axis_y+0.92,"average rank   (1 = best)",ha="center",fontsize=11.5,color=MUTE)
    # CD bar, top-left, below title
    cdy=axis_y+1.7
    ax.plot([lo,lo+CD],[cdy,cdy],color=INK,lw=2.4,solid_capstyle="butt")
    for xx in (lo,lo+CD): ax.plot([xx,xx],[cdy-0.12,cdy+0.12],color=INK,lw=1.3)
    ax.text(lo+CD/2,cdy+0.32,f"CD = {CD:.2f}",ha="center",fontsize=11.7,weight="bold")
    # method leaders: left half labelled left, right half labelled right
    mid=(lo+hi)/2
    left_idx=[i for i in range(k) if rv[i]<=mid]; right_idx=[i for i in range(k) if rv[i]>mid]
    for slot,i in enumerate(left_idx):
        yy=axis_y-1.0-slot*0.95; col=METHOD_COLOR.get(names[i],INK)
        ax.plot([rv[i],rv[i]],[axis_y,yy],color=col,lw=1.7)
        ax.plot([rv[i],lo-0.2],[yy,yy],color=col,lw=1.7)
        ax.text(lo-0.3,yy,f"{SHORT.get(names[i],names[i])}  ({rv[i]:.2f})",ha="right",va="center",
                fontsize=11.5,weight="bold" if names[i]=="PhenoFuse-LTR" else "normal")
    for slot,i in enumerate(right_idx):
        yy=axis_y-1.0-slot*0.95; col=METHOD_COLOR.get(names[i],INK)
        ax.plot([rv[i],rv[i]],[axis_y,yy],color=col,lw=1.7)
        ax.plot([rv[i],hi+0.2],[yy,yy],color=col,lw=1.7)
        ax.text(hi+0.3,yy,f"{SHORT.get(names[i],names[i])}  ({rv[i]:.2f})",ha="left",va="center",
                fontsize=11.5)
    # maximal cliques of methods within CD; drop contained ones; stack
    cliques=[]
    for i in range(k):
        j=i
        while j+1<k and (rv[j+1]-rv[i])<=CD: j+=1
        if j>i: cliques.append((i,j))
    maximal=[c for c in cliques if not any(c!=d and d[0]<=c[0] and c[1]<=d[1] for d in cliques)]
    for s,(i,j) in enumerate(maximal):
        yy=axis_y-0.30-s*0.34
        ax.plot([rv[i]-0.05,rv[j]+0.05],[yy,yy],color="#B5546B",lw=3.4,solid_capstyle="round")
    ax.text(lo-0.4,0.5,f"Friedman χ² = {E3['friedman_stat']:.1f},  p = {E3['friedman_p']:.1e},  "
            f"k = {E3['k']},  N = {E3['N']}   ·   bars join methods with no significant difference",
            fontsize=10.9,color=MUTE,ha="left")
    save(fig,"E3_cd_diagram")

# ================================================================ E4 SHAP
def e4():
    d=E4.sort_values("mean_abs_shap"); y=np.arange(len(d))
    grp={"resnik_bma":CLASS[0],"lin_bma":CLASS[2],"phrank":CLASS[1],"sum_ic_shared":CLASS[1],
         "max_ic_shared":CLASS[1],"mean_ic_profile":CLASS[1],"jaccard":NEUT[1],"cosine_bin":NEUT[1],
         "n_gene":HERO,"n_profile":NEUT[2],"n_shared":NEUT[2]}
    fig,ax=plt.subplots(figsize=(7.6,4.8)); style(ax,grid="x")
    ax.barh(d["feature"],d["mean_abs_shap"],color=[grp.get(f,MUTE) for f in d["feature"]],
            ec="white",height=.66,zorder=3)
    for i,(f,v) in enumerate(zip(d["feature"],d["mean_abs_shap"])): ax.text(v+.015,i,f"{v:.2f}",va="center",fontsize=10.6)
    ax.set_xlabel("mean |SHAP| (impact on ranking score)")
    ax.set_title("SHAP global attribution for the ranker",loc="left")
    ax.set_xlim(0,d["mean_abs_shap"].max()*1.14)
    save(fig,"E4_shap_global")

print("Generating figure system v2:")
for fn in (f1,f2,f3,f4,f5,f6,f7,f8,f9,e1,e2,e3,e4): fn()
print("done.")
