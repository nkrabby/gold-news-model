"""Pre-release probabilistic forecasts using only information known before the event.

Two stages: (1) a conservative ridge model estimates the distribution of the data
surprise; (2) existing reaction models integrate gold predictions over that
surprise distribution. Analyst research tilt is an explicit user assumption.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from market_context import FEATURES as MARKET_FEATURES
from train import MODEL_EVENTS

NUM=["forecast_previous_z","previous_surprise_z","recent_surprise_z","pre_return_24h","pre_volatility_24h"]
CAT=["event_type"]
TEST_START=pd.Timestamp("2025-01-01",tz="UTC")

def add_features(data):
    x=data.sort_values("release_time").copy()
    x["forecast_previous_z"]=(x.forecast-x.previous)/x.surprise_sigma
    x["previous_surprise_z"]=np.nan;x["recent_surprise_z"]=np.nan
    for _,idx in x.groupby("event_type").groups.items():
        idx=list(idx); s=x.loc[idx,"surprise_z"]
        x.loc[idx,"previous_surprise_z"]=s.shift(1)
        x.loc[idx,"recent_surprise_z"]=s.shift(1).rolling(2,min_periods=1).mean()
    return x

def model():
    pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUM),
                           ("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])
    return Pipeline([("features",pre),("model",Ridge(alpha=25))])

def base_row(event,forecast,previous,pre_trend,history):
    h=history[history.event_type.eq(event)].sort_values("release_time")
    last=h.iloc[-1]
    return {"event_type":event,"forecast_previous_z":(forecast-previous)/last.surprise_sigma,
            "previous_surprise_z":last.surprise_z,"recent_surprise_z":h.surprise_z.tail(2).mean(),
            "pre_return_24h":pre_trend,"pre_volatility_24h":np.nan}

def surprise_samples(bundle,features,event,tilt=0,n=161):
    mu=float(bundle["final_model"].predict(features)[0])+tilt
    residuals=bundle["surprise_residuals"].get(event,bundle["surprise_residuals"]["ALL"])
    # Deterministic residual quantiles make dashboard results reproducible.
    q=np.linspace(.01,.99,n)
    return mu+np.quantile(residuals,q),mu

def forecast(bundle,reaction_bundle,history,event,forecast_value,previous,pre_trend=0,hour=13.5,tilt=0,horizon="1h",seed=42,context=None):
    context=context or {}
    features=pd.DataFrame([base_row(event,forecast_value,previous,pre_trend,history)])
    raw_z,mu=surprise_samples(bundle,features,event,tilt)
    sign=float(history[history.event_type.eq(event)].gold_surprise_sign.median())
    f=pd.DataFrame({"signed_surprise_z":sign*raw_z,"abs_surprise_z":np.abs(raw_z),
                    "pre_return_24h":pre_trend,"pre_volatility_24h":np.nan,
                    "release_hour_sin":np.sin(2*np.pi*hour/24),"release_hour_cos":np.cos(2*np.pi*hour/24),"fomc_rate_change":0.0,
                    "ch_ret_1h":context.get("ch_ret_1h",np.nan),"ch_rsi_14":context.get("ch_rsi_14",np.nan),
                    "ch_range_pos_24h":context.get("ch_range_pos_24h",np.nan),"ch_vol_ratio":context.get("ch_vol_ratio",np.nan),
                    "event_type":event})
    for feature in MARKET_FEATURES:
        f[feature]=context.get(feature,np.nan)
    pair=reaction_bundle["models"][horizon]
    p=pair["direction"].predict_proba(f)[:,1]
    expected=pair["return"].predict(f)
    rng=np.random.default_rng(seed)
    sigma=bundle["reaction_residuals"].get(horizon,{}).get(event,bundle["reaction_residuals"][horizon]["ALL"])
    simulated=rng.choice(expected,size=3000)+rng.normal(0,sigma,size=3000)
    p_up=float(p.mean());mean_return=float(expected.mean())
    sigma_data=float(history[history.event_type.eq(event)].surprise_sigma.iloc[-1])
    return {"event_type":event,"horizon":horizon,"probability_up":p_up,
            "probability_down":1-p_up,"expected_return":mean_return,
            "lower_80":float(np.quantile(simulated,.1)),"upper_80":float(np.quantile(simulated,.9)),
            "expected_surprise_z":mu,"expected_actual":forecast_value+mu*sigma_data,
            "probability_beat":float((raw_z>0).mean()),"consensus":forecast_value,
            "label":"Bullish" if p_up>=.55 else "Bearish" if p_up<=.45 else "Neutral / low conviction"}

def train(data_path,output_path,reaction_path):
    data=add_features(pd.read_csv(data_path,parse_dates=["release_time"]))
    fit=data.dropna(subset=["surprise_z","forecast_previous_z"])
    train=fit[fit.release_time<TEST_START];test=fit[fit.release_time>=TEST_START].copy()
    validation=model();validation.fit(train[NUM+CAT],train.surprise_z)
    test["predicted_surprise_z"]=validation.predict(test[NUM+CAT])
    final=model();final.fit(fit[NUM+CAT],fit.surprise_z)
    residuals={}
    for event,g in test.groupby("event_type"):
        if len(g)>=8:residuals[event]=(g.surprise_z-g.predicted_surprise_z).to_numpy()
    residuals["ALL"]=(test.surprise_z-test.predicted_surprise_z).to_numpy()
    train_residuals={e:(g.surprise_z-validation.predict(g[NUM+CAT])).to_numpy() for e,g in train.groupby("event_type")}
    train_residuals["ALL"]=(train.surprise_z-validation.predict(train[NUM+CAT])).to_numpy()
    reactions=joblib.load(reaction_path);reaction_residuals={}
    for h,pair in reactions["models"].items():
        tr=data[data.release_time<TEST_START].dropna(subset=[f"return_{h}"]).copy()
        cols=pair.get("numeric",reactions["numeric_features"])
        pred=pair["return"].predict(tr[cols+reactions["categorical_features"]])
        tr["residual"]=tr[f"return_{h}"].to_numpy()-pred
        reaction_residuals[h]={**{e:g.residual.std() for e,g in tr.groupby("event_type")},"ALL":tr.residual.std()}
    # Honest out-of-time check of the integrated pre-release gold probability.
    pre_metrics=[]
    for h,pair in reactions["models"].items():
        probs=[]; expected=[]
        for _,row in test.iterrows():
            event=row.event_type
            mu=row.predicted_surprise_z
            res=train_residuals.get(event,train_residuals["ALL"])
            raw_z=mu+np.quantile(res,np.linspace(.01,.99,81))
            sign=row.gold_surprise_sign
            hour=row.release_time.hour+row.release_time.minute/60
            rf=pd.DataFrame({"signed_surprise_z":sign*raw_z,"abs_surprise_z":np.abs(raw_z),
                "pre_return_24h":row.pre_return_24h,"pre_volatility_24h":row.pre_volatility_24h,
                "release_hour_sin":np.sin(2*np.pi*hour/24),"release_hour_cos":np.cos(2*np.pi*hour/24),"fomc_rate_change":0.0,
                    "ch_ret_1h":row.get("ch_ret_1h",np.nan),"ch_rsi_14":row.get("ch_rsi_14",np.nan),
                    "ch_range_pos_24h":row.get("ch_range_pos_24h",np.nan),"ch_vol_ratio":row.get("ch_vol_ratio",np.nan),"event_type":event})
            for feature in MARKET_FEATURES:
                rf[feature]=row.get(feature,np.nan)
            probs.append(pair["direction"].predict_proba(rf)[:,1].mean())
            expected.append(pair["return"].predict(rf).mean())
        valid=test[f"return_{h}"].notna().to_numpy()
        y=(test.loc[valid,f"return_{h}"]>0).astype(int)
        p=np.asarray(probs)[valid]
        pre_metrics.append({"horizon":h,"test_events":len(y),"direction_accuracy":accuracy_score(y,p>=.5),
                            "majority_accuracy":max(y.mean(),1-y.mean()),"roc_auc":roc_auc_score(y,p),
                            "mae_bps":mean_absolute_error(test.loc[valid,f"return_{h}"],np.asarray(expected)[valid])*10000})
    pd.DataFrame(pre_metrics).to_csv(output_path.with_suffix(".gold_metrics.csv"),index=False)
    bundle={"final_model":final,"surprise_residuals":residuals,"reaction_residuals":reaction_residuals,
            "features":NUM+CAT,"test_start":TEST_START}
    joblib.dump(bundle,output_path)
    metrics={"surprise_mae_z":mean_absolute_error(test.surprise_z,test.predicted_surprise_z),
             "surprise_correlation":test[["surprise_z","predicted_surprise_z"]].corr().iloc[0,1],
             "beat_accuracy":accuracy_score(test.surprise_z>0,test.predicted_surprise_z>0),
             "test_events":len(test)}
    test.to_csv(output_path.with_suffix(".validation.csv"),index=False)
    return metrics

if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    print(train(root/"data/processed/event_study.csv",root/"outputs/pre_release_models.joblib",root/"outputs/models.joblib"))
