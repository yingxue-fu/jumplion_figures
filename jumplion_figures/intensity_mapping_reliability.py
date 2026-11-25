#!/usr/bin/env python3
"""Unified reliability analysis for JUMPlion precursor-specific intensity mapping.

Each leave-one-out (LOO) fit is computed once and reused for:
  * mapping robustness / gate sensitivity;
  * slope and MAE threshold sensitivity;
  * held-out prediction / extrapolation validation;
  * actual-recovery extrapolation-cutoff retention.

This module does not alter production mapping outputs.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ID = "Precursor.Id"
SPECIES = "Species"
DISTANCE_LABELS = [
    "Within fitted range", "0-0.5 log2 outside", "0.5-1 log2 outside",
    "1-2 log2 outside", ">2 log2 outside",
]
CUTOFFS = [("No limit", np.inf), ("<=2.0", 2.0), ("<=1.0", 1.0),
           ("<=0.5", 0.5), ("Interpolation only", 0.0)]
# "<=2.0" is the production applicability safeguard (max_extrapolation_log2 passed to
# run_reliability / run_intensity_mapping) -- a guardrail against extreme extrapolation, not a
# claim that error is uniformly small anywhere within 2 log2 of the observed range.
# Equal-width (1.0) slope windows shifted around slope=1 (centers 0.8..1.2), NOT progressively
# narrowing ranges -- the point is to test whether sliding the window away from the expected
# 1:1 DIA-NN~JUMPlion relationship changes retention/robustness/accuracy, not to narrow in on a
# single "best" width. Production = 0.5-1.5 (center 1.0).
SLOPE_RANGES = [(0.3,1.3),(0.4,1.4),(0.5,1.5),(0.6,1.6),(0.7,1.7)]
# Candidate MAE cutoffs to test retention/robustness tradeoffs around the production value
# (passed in as `max_mae`; currently 0.50 for Fig. S4D -- see intensity_mapping_mechanism.py).
MAE_CUTOFFS = [0.30,0.35,0.40,0.50,0.60]


def _bool(s):
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(False).astype(str).str.lower().isin(["true","1","yes","y"])


def _import_production(module_name="jumplion.intensity_mapping", module_dir=None):
    """The production mapping module: by default the one shipped with the installed `jumplion` package
    (pip install jumplion), so the reliability analysis tests exactly the released code."""
    if module_dir:
        sys.path.insert(0, str(Path(module_dir).resolve()))
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name and module_name.startswith(e.name):
            raise ModuleNotFoundError(f"cannot import '{module_name}': install JUMPlion into this environment "
                                      "(pip install jumplion, Python 3.12+)") from e
        raise


def _distance(x, train):
    lo, hi = float(np.min(train)), float(np.max(train))
    return lo-x if x < lo else x-hi if x > hi else 0.0


def _distance_bin(d):
    if d <= 0: return DISTANCE_LABELS[0]
    if d <= .5: return DISTANCE_LABELS[1]
    if d <= 1: return DISTANCE_LABELS[2]
    if d <= 2: return DISTANCE_LABELS[3]
    return DISTANCE_LABELS[4]


def _fit(prod, x, y):
    metrics, model_data = prod.fast_linear_fit(x, y)
    return metrics, model_data


def _pass(n, slope, mae, min_overlap, min_slope=.5, max_slope=1.5, max_mae=.4):
    return bool(n >= min_overlap and np.isfinite(slope) and min_slope <= slope <= max_slope
                and np.isfinite(mae) and mae <= max_mae)


def _species_table(path):
    if path is None: return None
    p = Path(path)
    if not p.exists(): return None
    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    s = pd.read_csv(p, sep=sep, usecols=[ID, SPECIES]).drop_duplicates(ID)
    s[ID] = s[ID].astype(str); s[SPECIES] = s[SPECIES].astype(str).str.upper().str.strip()
    return s.set_index(ID)


def _summarize_extrapolation(loo):
    rows=[]
    for lab in DISTANCE_LABELS:
        z=loo[loo["Extrapolation.Bin"].eq(lab)]
        if z.empty:
            rows.append({"Extrapolation.Bin":lab,"N":0}); continue
        a=z["Abs.Prediction.Error"].dropna(); e=z["Prediction.Error"].dropna()
        rows.append({"Extrapolation.Bin":lab,"N":len(z),
                     "Median.Abs.Error":a.median(),"Q25.Abs.Error":a.quantile(.25),
                     "Q75.Abs.Error":a.quantile(.75),"MAE":a.mean(),
                     "P90.Abs.Error":a.quantile(.90),"P95.Abs.Error":a.quantile(.95),
                     "Median.Signed.Error":e.median(),"Mean.Signed.Error":e.mean()})
    return pd.DataFrame(rows)


def _actual_recoveries(diann, jump, metrics, runs, species, quantification_parquet=None, max_extrapolation_log2=2.0):
    rows=[]; smap={} if species is None else species[SPECIES].to_dict()
    good=metrics.index[metrics["Model.Good"]].intersection(diann.index).intersection(jump.index)
    for pid in good:
        y=diann.loc[pid,runs].to_numpy(float); x=jump.loc[pid,runs].to_numpy(float)
        paired=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
        recover=~np.isfinite(y)&np.isfinite(x)&(x>0)
        if not paired.any() or not recover.any(): continue
        lx=np.log2(x[paired]); lo,hi=float(lx.min()),float(lx.max())
        ly=np.log2(y[np.isfinite(y)&(y>0)]); mean_y=float(ly.mean()) if len(ly) else np.nan
        for run,val in zip(np.asarray(runs)[recover],np.log2(x[recover])):
            d=lo-val if val<lo else val-hi if val>hi else 0.0
            rows.append({ID:str(pid),"Run":str(run),SPECIES:smap.get(str(pid),np.nan),
                         "Mean.DIANN.Log2":mean_y,"Extrapolation.Distance":float(d),
                         "Extrapolation.Bin":_distance_bin(float(d))})
    rec=pd.DataFrame(rows)
    if rec.empty: return rec, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    prec=rec[[ID,"Mean.DIANN.Log2"]].drop_duplicates(ID).dropna()
    try:
        _,edges=pd.qcut(prec["Mean.DIANN.Log2"],q=5,retbins=True,duplicates="drop"); edges=np.unique(edges)
        labs=[f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges)-1)]
        rec["Intensity.Bin"]=pd.cut(rec["Mean.DIANN.Log2"],bins=edges,labels=labs,include_lowest=True)
    except ValueError:
        rec["Intensity.Bin"]="all"
    if quantification_parquet and Path(quantification_parquet).exists():
        q=pd.read_parquet(quantification_parquet)
        tier="evidence_tier_label"
        if {ID,"Run",tier}.issubset(q.columns):
            ann=q[[ID,"Run",tier]].drop_duplicates([ID,"Run"]); ann[ID]=ann[ID].astype(str); ann["Run"]=ann["Run"].astype(str)
            rec=rec.merge(ann,on=[ID,"Run"],how="left")
    def retention(group=None):
        source=[(None,rec)] if group is None else list(rec.dropna(subset=[group]).groupby(group,observed=True))
        out=[]
        for g,z in source:
            n=len(z); total_prec=z[ID].nunique()
            for name,c in CUTOFFS:
                keep=np.ones(n,dtype=bool) if np.isinf(c) else z["Extrapolation.Distance"].to_numpy(float)<=c
                r={"Cutoff":name,"Max.Extrapolation.Log2":c,
                   "Is.Production":bool(np.isfinite(c) and np.isclose(c,max_extrapolation_log2)),
                   "N.Current":n,"N.Retained":int(keep.sum()),
                   "N.Excluded":int((~keep).sum()),"Retained.Fraction":float(keep.mean()) if n else np.nan}
                if group is None:
                    kept=z.loc[keep,ID].nunique(); r.update({"N.Current.Recovered.Precursors":total_prec,
                        "N.Precursors.Retaining.Recovery":kept,"N.Precursors.Losing.All.Recovery":total_prec-kept})
                else: r[group]=g
                out.append(r)
        return pd.DataFrame(out)
    tier="evidence_tier_label"
    return rec, retention(), retention(SPECIES), retention("Intensity.Bin"), retention(tier) if tier in rec.columns else pd.DataFrame()


def run_reliability(quantity_diann, quantity_jumplion, mapping_metrics, output_dir,
                    species_annotation=None, quantification_parquet=None,
                    production_module="jumplion.intensity_mapping", production_module_dir=None,
                    min_slope=.5, max_slope=1.5, max_mae=.4, max_extrapolation_log2=2.0,
                    fit_tolerance=1e-8, require_production_match=True):
    """Run unified reliability analysis and return all result DataFrames."""
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    prod=_import_production(production_module,production_module_dir)
    diann=prod.load_quantity_matrix(quantity_diann,id_col=ID)
    jump=prod.load_quantity_matrix(quantity_jumplion,id_col=ID)
    runs=diann.columns.intersection(jump.columns); precs=diann.index.intersection(jump.index)
    diann=diann.loc[precs,runs]; jump=jump.loc[precs,runs]
    m=pd.read_csv(mapping_metrics,sep="\t").drop_duplicates(ID).set_index(ID); m.index=m.index.astype(str)
    m["Model.Good"]=_bool(m["Model.Good"])
    eff=pd.to_numeric(m["Effective.Min.Overlap"],errors="coerce").dropna().unique()
    if len(eff)!=1: raise ValueError(f"Expected one Effective.Min.Overlap, found {eff}")
    min_overlap=int(eff[0]); species=_species_table(species_annotation)

    verify=[]; loo_rows=[]; prec_rows=[]
    for pid,pm in m.iterrows():
        if pid not in diann.index or pid not in jump.index: continue
        y0=diann.loc[pid].to_numpy(float); x0=jump.loc[pid].to_numpy(float); rr=np.asarray(runs,dtype=str)
        valid=np.isfinite(x0)&np.isfinite(y0)&(x0>0)&(y0>0)
        x=np.log2(x0[valid]); y=np.log2(y0[valid]); vr=rr[valid]
        full,model=_fit(prod,x,y); n=int(full["N.Overlap"])
        # Production (fit_linear_models) never fits below Effective.Min.Overlap -- it leaves
        # Slope/MAE as NaN for those rows (Model.Fail.Reason="INSUFFICIENT_OVERLAP"). Recomputing
        # a fit here regardless of n would never match that NaN, so apply the same gate.
        below_threshold = n < min_overlap
        full_slope = np.nan if below_threshold else full["Slope"]
        full_mae = np.nan if below_threshold else full["MAE"]
        sd=abs(full_slope-pm["Slope"]) if np.isfinite(full_slope) and np.isfinite(pm["Slope"]) else np.nan
        md=abs(full_mae-pm["MAE"]) if np.isfinite(full_mae) and np.isfinite(pm["MAE"]) else np.nan
        slope_match = ((not np.isfinite(full_slope) and not np.isfinite(pm["Slope"])) or
                       (np.isfinite(sd) and sd <= fit_tolerance))
        mae_match = ((not np.isfinite(full_mae) and not np.isfinite(pm["MAE"])) or
                     (np.isfinite(md) and md <= fit_tolerance))
        n_match = n == int(pm["N.Overlap"])
        verify.append({ID:pid,"N.Overlap.Match":n_match,"Slope.Match":slope_match,"MAE.Match":mae_match,
                       "Production.N.Overlap":int(pm["N.Overlap"]),"Recomputed.N.Overlap":n,
                       "Production.Slope":pm["Slope"],"Recomputed.Slope":full_slope,
                       "Production.MAE":pm["MAE"],"Recomputed.MAE":full_mae,
                       "Abs.Slope.Difference":sd,"Abs.MAE.Difference":md,
                       "Fit.Match":bool(n_match and slope_match and mae_match)})
        if below_threshold:
            # No valid production model for this precursor -- nothing to LOO-analyze either.
            continue
        full_pass=_pass(n,full["Slope"],full["MAE"],min_overlap,min_slope,max_slope,max_mae)
        dx=x-x.mean() if n else np.array([]); ssx=float(np.dot(dx,dx)) if n else 0
        leverage=(1/n)+(dx**2)/ssx if n and ssx>0 else np.full(n,np.nan)
        residual=y-(full["Intercept"]+full["Slope"]*x) if n else np.array([])
        for j in range(n):
            keep=np.ones(n,dtype=bool); keep[j]=False
            lf,_=_fit(prod,x[keep],y[keep]); ln=int(lf["N.Overlap"])
            eligible=ln>=min_overlap
            slope_ok=eligible and np.isfinite(lf["Slope"]) and min_slope<=lf["Slope"]<=max_slope
            mae_ok=eligible and np.isfinite(lf["MAE"]) and lf["MAE"]<=max_mae
            lp=bool(slope_ok and mae_ok)
            pred=lf["Intercept"]+lf["Slope"]*x[j] if np.isfinite(lf["Intercept"]) and np.isfinite(lf["Slope"]) else np.nan
            err=pred-y[j] if np.isfinite(pred) else np.nan; dist=_distance(x[j],x[keep]) if keep.sum() else np.nan
            loo_rows.append({ID:pid,"Run":vr[j],"Full.N.Overlap":n,"Full.Slope":full["Slope"],"Full.MAE":full["MAE"],"Full.Pass":full_pass,
                "LOO.N.Overlap":ln,"LOO.Slope":lf["Slope"],"LOO.MAE":lf["MAE"],"LOO.Overlap.Eligible":eligible,
                "LOO.Slope.Criterion.Pass":slope_ok,"LOO.MAE.Criterion.Pass":mae_ok,"LOO.Pass":lp,
                "Delta.Slope":lf["Slope"]-full["Slope"] if np.isfinite(lf["Slope"]) and np.isfinite(full["Slope"]) else np.nan,
                "Abs.Delta.Slope":abs(lf["Slope"]-full["Slope"]) if np.isfinite(lf["Slope"]) and np.isfinite(full["Slope"]) else np.nan,
                "Residual":residual[j] if j<len(residual) else np.nan,"Leverage":leverage[j] if j<len(leverage) else np.nan,
                "Heldout.JUMPlion.Log2":x[j],"Heldout.DIANN.Log2":y[j],"Predicted.DIANN.Log2":pred,
                "Prediction.Error":err,"Abs.Prediction.Error":abs(err) if np.isfinite(err) else np.nan,
                "Extrapolation.Distance":dist,"Extrapolation.Bin":_distance_bin(dist) if np.isfinite(dist) else np.nan})
    loo=pd.DataFrame(loo_rows); verification=pd.DataFrame(verify)
    verification.to_csv(out / "reliability_production_fit_verification.tsv", sep="\t", index=False)
    if len(verification):
        bad = verification.loc[~verification["Fit.Match"]]
        if len(bad):
            msg = (f"Unified reliability analysis does not reproduce {len(bad)}/{len(verification)} production fits. "
                   "This usually means the wrong quantity matrices or production module were supplied. "
                   f"See {out / 'reliability_production_fit_verification.tsv'}.")
            if require_production_match:
                raise RuntimeError(msg)
            print("WARNING:", msg)
    if species is not None and not loo.empty: loo=loo.merge(species.reset_index(),on=ID,how="left")

    # Per-precursor stability summaries.
    rows=[]
    for pid,z in loo.groupby(ID,sort=False):
        ze=z[z["LOO.Overlap.Eligible"]]; ds=ze["Abs.Delta.Slope"].dropna(); full_pass=bool(z["Full.Pass"].iloc[0])
        slope_fail=~ze["LOO.Slope.Criterion.Pass"] if len(ze) else pd.Series(dtype=bool)
        mae_fail=~ze["LOO.MAE.Criterion.Pass"] if len(ze) else pd.Series(dtype=bool)
        nfail=int((~ze["LOO.Pass"]).sum()) if len(ze) else 0
        if full_pass:
            cls="Pass; LOO structurally ineligible" if len(ze)==0 else "Stable pass" if nfail==0 else "Model-fragile pass"
        else:
            cls="Fail with eligible LOO rescue" if len(ze) and ze["LOO.Pass"].any() else "Stable fail"
        rows.append({ID:pid,"Full.N.Overlap":int(z["Full.N.Overlap"].iloc[0]),"Full.Slope":z["Full.Slope"].iloc[0],"Full.MAE":z["Full.MAE"].iloc[0],"Full.Pass":full_pass,
                     "N.LOO.Eligible":len(ze),"N.LOO.Eligible.Fail":nfail,"N.LOO.Any.Slope.Fail":int(slope_fail.sum()) if len(ze) else 0,
                     "N.LOO.Any.MAE.Fail":int(mae_fail.sum()) if len(ze) else 0,"Median.Abs.Delta.Slope.Eligible":ds.median() if len(ds) else np.nan,
                     "P95.Abs.Delta.Slope.Eligible":ds.quantile(.95) if len(ds) else np.nan,"Posthoc.Stability.Class":cls})
    post=pd.DataFrame(rows).set_index(ID)
    if species is not None: post=post.join(species,how="left")
    post["Slope.Distance.To.Boundary"]=np.minimum(post["Full.Slope"]-min_slope,max_slope-post["Full.Slope"])
    post["MAE.Distance.To.Boundary"]=max_mae-post["Full.MAE"]
    passed=post[post["Full.Pass"]].copy()
    def ic(r):
        s=r["N.LOO.Any.Slope.Fail"]>0; a=r["N.LOO.Any.MAE.Fail"]>0
        return "Slope + MAE" if s and a else "Slope only" if s else "MAE only" if a else "No eligible downstream failure"
    passed["Eligible.LOO.Instability.Type"]=passed.apply(ic,axis=1)
    instability=passed["Eligible.LOO.Instability.Type"].value_counts().rename_axis("Instability.Type").reset_index(name="N.Precursors")
    overlap=post.groupby("Full.N.Overlap").agg(N_Precursors=("Full.Pass","size"),Median_Abs_Delta_Slope=("Median.Abs.Delta.Slope.Eligible","median"),P95_Abs_Delta_Slope=("P95.Abs.Delta.Slope.Eligible","median")).reset_index()

    # Threshold sensitivity: report both retention and reliability.
    # Full-data acceptance is evaluated once per precursor; LOO fragility is
    # evaluated only among accepted models with at least one overlap-eligible LOO fit.
    n_models = loo[ID].nunique()
    slope_rows=[]
    for lo,hi in SLOPE_RANGES:
        full_by_prec = loo.groupby(ID, sort=False).first()[["Full.N.Overlap","Full.Slope","Full.MAE"]]
        accepted = ((full_by_prec["Full.N.Overlap"] >= min_overlap) &
                    full_by_prec["Full.Slope"].between(lo,hi) &
                    (full_by_prec["Full.MAE"] <= max_mae))
        accepted_ids = accepted.index[accepted]
        z = loo[loo[ID].isin(accepted_ids) & loo["LOO.Overlap.Eligible"]].copy()
        z["Candidate.LOO.Pass"] = z["LOO.Slope"].between(lo,hi) & (z["LOO.MAE"] <= max_mae)
        per = z.groupby(ID)["Candidate.LOO.Pass"].agg(N="size", AllPass="all") if len(z) else pd.DataFrame()
        stable_fraction = float(per["AllPass"].mean()) if len(per) else np.nan
        fragile_fraction = 1.0-stable_fraction if np.isfinite(stable_fraction) else np.nan
        n_fragile = int((~per["AllPass"]).sum()) if len(per) else 0
        heldout = loo[loo[ID].isin(accepted_ids)]["Abs.Prediction.Error"].dropna()
        slope_rows.append({"Slope.Min":lo,"Slope.Max":hi,"Center":round((lo+hi)/2,2),"Label":f"{lo:.1f}-{hi:.1f}",
                           "Is.Production":bool(np.isclose(lo,min_slope) and np.isclose(hi,max_slope)),
                           "N.Accepted":int(accepted.sum()),
                           "Accepted.Fraction":float(accepted.mean()) if len(accepted) else np.nan,
                           "N.With.Eligible.LOO":int(len(per)),
                           "N.LOO.Fragile":n_fragile,
                           "LOO.Stable.Fraction":stable_fraction,
                           "LOO.Fragile.Fraction":fragile_fraction,
                           "Heldout.MAE":float(heldout.mean()) if len(heldout) else np.nan})
    slope_sens=pd.DataFrame(slope_rows)

    mae_rows=[]
    full_by_prec = loo.groupby(ID, sort=False).first()[["Full.N.Overlap","Full.Slope","Full.MAE"]]
    for cut in MAE_CUTOFFS:
        accepted = ((full_by_prec["Full.N.Overlap"] >= min_overlap) &
                    full_by_prec["Full.Slope"].between(min_slope,max_slope) &
                    (full_by_prec["Full.MAE"] <= cut))
        accepted_ids = accepted.index[accepted]
        z = loo[loo[ID].isin(accepted_ids) & loo["LOO.Overlap.Eligible"]].copy()
        z["Candidate.LOO.Pass"] = z["LOO.Slope"].between(min_slope,max_slope) & (z["LOO.MAE"] <= cut)
        per = z.groupby(ID)["Candidate.LOO.Pass"].agg(N="size", AllPass="all") if len(z) else pd.DataFrame()
        stable_fraction = float(per["AllPass"].mean()) if len(per) else np.nan
        n_fragile = int((~per["AllPass"]).sum()) if len(per) else 0
        heldout = loo[loo[ID].isin(accepted_ids)]["Abs.Prediction.Error"].dropna()
        mae_rows.append({"MAE.Cutoff":cut,"Is.Production":bool(np.isclose(cut,max_mae)),
                         "N.Accepted":int(accepted.sum()),
                         "Accepted.Fraction":float(accepted.mean()) if len(accepted) else np.nan,
                         "N.With.Eligible.LOO":int(len(per)),
                         "N.LOO.Fragile":n_fragile,
                         "LOO.Stable.Fraction":stable_fraction,
                         "LOO.Fragile.Fraction":1.0-stable_fraction if np.isfinite(stable_fraction) else np.nan,
                         "Heldout.MAE":float(heldout.mean()) if len(heldout) else np.nan})
    mae_sens=pd.DataFrame(mae_rows)

    # Huber comparison.
    huber_rows=[]
    try:
        from sklearn.linear_model import HuberRegressor
        for pid in m.index.intersection(diann.index):
            y0=diann.loc[pid].to_numpy(float); x0=jump.loc[pid].to_numpy(float); v=np.isfinite(x0)&np.isfinite(y0)&(x0>0)&(y0>0)
            x=np.log2(x0[v]); y=np.log2(y0[v])
            if len(x)<min_overlap: continue
            ols,_=_fit(prod,x,y); h=HuberRegressor(fit_intercept=True,epsilon=1.35).fit(x.reshape(-1,1),y); hp=h.predict(x.reshape(-1,1)); hs=float(h.coef_[0]); hm=float(np.mean(np.abs(y-hp)))
            op=_pass(len(x),ols["Slope"],ols["MAE"],min_overlap,min_slope,max_slope,max_mae); hpass=_pass(len(x),hs,hm,min_overlap,min_slope,max_slope,max_mae)
            grp="Both pass" if op and hpass else "OLS only" if op else "Huber only" if hpass else "Both fail"
            huber_rows.append({ID:pid,"OLS.Pass":op,"Huber.Pass":hpass,"Decision.Group":grp})
    except Exception:
        pass
    huber=pd.DataFrame(huber_rows); order=["Both pass","OLS only","Huber only","Both fail"]
    huber_summary=huber["Decision.Group"].value_counts().reindex(order,fill_value=0).rename_axis("Decision.Group").reset_index(name="N.Precursors") if len(huber) else pd.DataFrame({"Decision.Group":order,"N.Precursors":0})
    concord=float((huber["OLS.Pass"]==huber["Huber.Pass"]).mean()) if len(huber) else np.nan
    concordance=pd.DataFrame({"Metric":["N.Precursors","Decision.Concordance.Fraction"],"Value":[len(huber),concord]})

    ex_summary=_summarize_extrapolation(loo)
    rec,ret,ret_sp,ret_int,ret_tier=_actual_recoveries(diann,jump,m,runs,species,quantification_parquet,
                                                       max_extrapolation_log2=max_extrapolation_log2)

    validation_summary = pd.DataFrame({"Metric":["N.Production.Fits","N.Fit.Match","Fit.Match.Fraction"],
        "Value":[len(verification), int(verification["Fit.Match"].sum()) if len(verification) else 0,
                 float(verification["Fit.Match"].mean()) if len(verification) else np.nan]})
    results={"verification":verification,"validation_summary":validation_summary,"loo":loo,"precursor_stability":post.reset_index(),"gate_instability":instability,
             "stability_by_overlap":overlap,"slope_threshold_sensitivity":slope_sens,"mae_threshold_sensitivity":mae_sens,
             "huber_groups":huber_summary,"huber_concordance":concordance,"extrapolation_summary":ex_summary,
             "actual_recoveries":rec,"cutoff_retention":ret,"cutoff_by_species":ret_sp,"cutoff_by_intensity":ret_int,"cutoff_by_localization_tier":ret_tier}
    for name,df in results.items():
        if isinstance(df,pd.DataFrame): df.to_csv(out/f"reliability_{name}.tsv",sep="\t",index=False)
    return results


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--quantity-diann",required=True); ap.add_argument("--quantity-jumplion",required=True); ap.add_argument("--mapping-metrics",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--species-annotation"); ap.add_argument("--quantification-parquet"); ap.add_argument("--production-module",default="jumplion.intensity_mapping"); ap.add_argument("--production-module-dir")
    args=ap.parse_args(); run_reliability(args.quantity_diann,args.quantity_jumplion,args.mapping_metrics,args.output_dir,args.species_annotation,args.quantification_parquet,args.production_module,args.production_module_dir)

if __name__=="__main__": main()
