#!/usr/bin/env python3
"""Redesigned schematic figures (Figure 1 architecture, Figure 2 leave-target-out concept).
Consistent with the make_figures_v2 visual system. Vector PDF + PNG.
    python scripts/make_schematics.py
"""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results_real/figures"; PAPER=OUT
for d in (OUT,PAPER): d.mkdir(parents=True,exist_ok=True)

INK="#1B2430"; MUTE="#5C6B7A"; HERO="#1F4E79"; HEROFILL="#E7EFF7"
ORANGE="#E07B39"; ORANGEFILL="#FBEDE0"; GREEN="#2E7D6B"; GREENFILL="#E4F1ED"
ROSE="#B5546B"; ROSEFILL="#F6E5EA"; NEUTFILL="#EEF1F4"; NEUTEDGE="#9FB3C6"; CARD="#F6F8FA"

mpl.rcParams.update({
    "savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.04,
    "font.family":"DejaVu Sans","text.color":INK,"figure.facecolor":"white"})

def save(fig,name):
    for ext in ("pdf","png"):
        fig.savefig(OUT/f"{name}.{ext}")
        if ext=="pdf": fig.savefig(PAPER/f"{name}.pdf")
    plt.close(fig); print("  ->",name)

def box(ax,x,y,w,h,fc,ec,txt,fs=9.0,bold=False,lw=1.2,round=0.9,tc=None,va="center"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.18,rounding_size={round}",
                 fc=fc,ec=ec,lw=lw,zorder=3))
    if txt:
        ax.text(x+w/2,y+h/2,txt,ha="center",va=va,fontsize=fs,color=tc or INK,
                weight="bold" if bold else "normal",zorder=4)

def arrow(ax,x1,y1,x2,y2,color=MUTE,lw=1.5,ms=12,style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=ms,
                 lw=lw,color=color,zorder=2,shrinkA=0,shrinkB=0))

def chip(ax,x,y,w,h,fc,ec,txt,fs=7.6,tc=None,strike=False,lw=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.35",
                 fc=fc,ec=ec,lw=lw,zorder=5))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc or INK,zorder=6)
    if strike:
        ax.plot([x+0.06,x+w-0.06],[y+h/2,y+h/2],color=ROSE,lw=1.4,zorder=7)

# ============================================================ FIGURE 1: ARCHITECTURE
def architecture():
    fig,ax=plt.subplots(figsize=(11.6,4.6)); ax.axis("off")
    ax.set_xlim(0,116); ax.set_ylim(0,46)

    # ---- col 1: data sources ----
    ax.text(11,43.2,"Real HPO release 2025-01-16",ha="center",fontsize=8.6,style="italic",color=MUTE)
    for i,t in enumerate(["Ontology graph\n(hp.obo)","Disorder–phenotype\n(phenotype.hpoa)","Gene–phenotype\n(+ disease id)"]):
        box(ax,1,30.5-i*8.6,20,7.0,CARD,NEUTEDGE,t,fs=8.2,round=0.7)
    # ---- col 2: cohort ----
    box(ax,27,12,20,20,NEUTFILL,NEUTEDGE,
        "Leakage-safe\ncohort &\nprofile simulation\n\n(group-disjoint\ndisease / gene\nsplits)",fs=8.4,round=0.9)
    for i in range(3):
        arrow(ax,21,27.0-i*8.6,27,24.5-i*2.0)
    # ---- col 3: PROPOSED MODEL container ----
    mx,mw=53,30
    ax.add_patch(FancyBboxPatch((mx,7.5),mw,33,boxstyle="round,pad=0.3,rounding_size=1.2",
                 fc="white",ec=HERO,lw=2.0,zorder=3))
    ax.text(mx+mw/2,37.6,"PhenoFuse-LTR  (proposed)",ha="center",fontsize=9.4,weight="bold",color=HERO,zorder=4)
    views=[("Overlap view",NEUTFILL,NEUTEDGE),
           ("IC-shared view  ·  train-only IC",ORANGEFILL,ORANGE),
           ("Best-match-average view",ORANGEFILL,ORANGE)]
    for i,(t,fc,ec) in enumerate(views):
        box(ax,mx+2.0,30.0-i*5.4,mw-4,4.2,fc,ec,t,fs=7.9,round=0.5,lw=1.1)
    box(ax,mx+5.0,9.6,mw-10,4.4,HEROFILL,HERO,"LambdaRank fusion\n(LightGBM)",fs=8.4,bold=True,round=0.6,lw=1.4)
    # single clean converging arrow from the stacked views into the fusion box
    arrow(ax,mx+mw/2,19.0,mx+mw/2,14.2,color=HERO,lw=1.4,ms=11)
    ax.text(mx+mw/2,8.0,"all views use train-only IC  ·  leakage-safe",ha="center",
            fontsize=7.2,style="italic",color=ORANGE,zorder=4)
    arrow(ax,47,22,mx,22)  # cohort -> model
    # ---- col 4: evaluation regimes ----
    ex=89
    arrow(ax,mx+mw,22,ex,22)
    box(ax,ex,25.5,18,7.0,ROSEFILL,ROSE,"Naive regime\n(annotation recovery)",fs=7.8,round=0.6)
    box(ax,ex,11.5,18,7.0,GREENFILL,GREEN,"Leave-target-out\n(rigorous)",fs=7.8,round=0.6)
    arrow(ax,ex+9,25.5,ex+9,18.5,color=NEUTEDGE,lw=1.0,ms=8)
    # ---- col 5: outputs ----
    arrow(ax,ex+18,22,109.5,22)
    box(ax,109.5,12,6.0,20,CARD,NEUTEDGE,"Ranked\ngenes\n+\nclustered\nstats\n+\nSHAP",fs=7.6,round=0.7)
    save(fig,"S1_architecture")

# ============================================================ FIGURE 2: LEAVE-TARGET-OUT
def leave_target_out():
    fig,ax=plt.subplots(figsize=(11.4,5.0)); ax.axis("off")
    ax.set_xlim(0,114); ax.set_ylim(0,50)

    # ---- shared query at top ----
    ax.text(57,48.2,"Query: held-out disorder $D^\\star$ and its simulated phenotype profile $P$",
            ha="center",fontsize=9.2,weight="bold")
    qx=37
    for i in range(4):
        chip(ax,qx+i*9.6,43.0,8.8,3.2,HEROFILL,HERO,f"$t_{{{i+1}}}$",fs=8.2)
    ax.text(34.5,44.6,"$P=$",ha="right",fontsize=9.2)

    def panel(x0,title,fill,edge,circular):
        # panel frame
        ax.add_patch(FancyBboxPatch((x0,2.5),52,36,boxstyle="round,pad=0.3,rounding_size=1.0",
                     fc="white",ec=edge,lw=1.6,zorder=1))
        ax.text(x0+26,36.0,title,ha="center",fontsize=9.8,weight="bold",color=edge,zorder=4)
        ax.text(x0+26,33.2,"candidate causal gene $g$ represented by the terms of its associated disorders",
                ha="center",fontsize=7.5,color=MUTE,zorder=4)
        # group D* (the target disorder)
        gy=26.5
        ax.text(x0+4,gy+1.6,"from $D^\\star$",fontsize=7.6,color=ROSE if circular else MUTE,ha="left")
        for i in range(4):
            if circular:
                chip(ax,x0+15+i*8.0,gy,7.4,3.0,ROSEFILL,ROSE,f"$t_{{{i+1}}}$",fs=7.8)
            else:
                chip(ax,x0+15+i*8.0,gy,7.4,3.0,"#F3F4F6","#C9D0D6",f"$t_{{{i+1}}}$",fs=7.8,tc="#AEB6BE",strike=True)
        # groups D1, D2
        for j,(lbl,base) in enumerate([("from $D_1$",5),("from $D_2$",7)]):
            yy=20.5-j*5.2
            ax.text(x0+4,yy+1.6,lbl,fontsize=7.6,color=MUTE,ha="left")
            for i in range(2):
                chip(ax,x0+15+i*8.0,yy,7.4,3.0,NEUTFILL,NEUTEDGE,f"$t_{{{base+i}}}$",fs=7.8)
        # annotation band
        if circular:
            ax.add_patch(Rectangle((x0+14.4,gy-0.5),32.6,4.0,fc="none",ec=ROSE,lw=1.4,ls=(0,(4,2)),zorder=8))
            ax.annotate("query terms recovered from the\ngene's own target annotations",
                        xy=(x0+15,gy+1.5),xytext=(x0+26,9.5),fontsize=7.5,color=ROSE,ha="center",
                        arrowprops=dict(arrowstyle="-|>",color=ROSE,lw=1.0,connectionstyle="arc3,rad=0.15"))
            ax.text(x0+26,5.4,"circular  →  scores saturate",ha="center",fontsize=8.2,weight="bold",color=ROSE)
        else:
            ax.text(x0+31,gy+1.5,"terms($D^\\star$) excluded",fontsize=7.6,color=ROSE,style="italic",ha="left")
            ax.annotate("match must come from phenotypic\nconsistency across the gene's other disorders",
                        xy=(x0+19,20.5+1.5),xytext=(x0+26,9.5),fontsize=7.5,color=GREEN,ha="center",
                        arrowprops=dict(arrowstyle="-|>",color=GREEN,lw=1.0,connectionstyle="arc3,rad=-0.15"))
            ax.text(x0+26,5.4,"non-circular  →  measures true inference",ha="center",fontsize=8.2,weight="bold",color=GREEN)

    panel(2,"Naive regime",ROSEFILL,ROSE,circular=True)
    panel(60,"Leave-target-out regime",GREENFILL,GREEN,circular=False)
    # equation linking
    ax.text(57,0.6,"Leave-target-out representation:   $\\mathrm{terms}(g\\,;D^\\star)=\\bigcup_{D\\,\\in\\,\\mathcal{D}(g),\\;D\\neq D^\\star}\\mathrm{terms}(D)$",
            ha="center",fontsize=8.6,color=INK)
    save(fig,"S2_leave_target_out")

print("Generating schematics:")
architecture(); leave_target_out()
print("done.")
