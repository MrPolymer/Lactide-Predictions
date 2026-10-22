"""Flow ROP surrogate: kernel regularized least squares + constrained inverse design."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Flow ROP · KRLS Explorer', page_icon='🧪', layout='wide')
st.title('Flow ROP · KRLS Explorer')
st.caption('DBU / residence-time response surfaces and inverse process design')
DATA = Path(__file__).with_name('ALL.csv')
V_ML = 0.8345
MONOMER_M = 1.0
MW = 144.126  # g/mol, L-lactide, as in spreadsheet

@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    cols = ['time','catalyst','initiator','Mnexp','dispersity','conversion','rate','rateDBU']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=cols).loc[lambda d:(d.time>0)&(d.catalyst>0)].copy()

@st.cache_resource
def model_for(initiator):
    df = load_data()
    subset = df.loc[df.initiator==initiator].copy()
    # Standardized logarithmic reaction coordinates; kernel LS (RBF kernel ridge).
    xy = np.log(subset[['catalyst','time']].to_numpy(dtype=float))
    scaler=StandardScaler().fit(xy)
    xx=scaler.transform(xy)
    models={}
    for name in ('Mnexp','dispersity','conversion'):
        values = subset[name].to_numpy(float)
        # Mean + normalized residual model for numerically stable, comparable fits.
        center=float(np.mean(values)); scale=max(float(np.std(values)),1e-5)
        y=(values-center)/scale
        # At small n avoid trying to tune on 1-replicate response points.
        model=KernelRidge(kernel='rbf', gamma=0.5, alpha=0.05)
        model.fit(xx,y)
        models[name]=(model,center,scale)
    return scaler,models,subset

def predict(dbu,seconds,initiator):
    scaler,models,_=model_for(initiator)
    dbu,seconds=np.broadcast_arrays(np.asarray(dbu,float),np.asarray(seconds,float))
    flat=np.column_stack([dbu.ravel(),seconds.ravel()])
    X=scaler.transform(np.log(flat))
    out={}
    for name,(model,center,scale) in models.items():
        out[name]=(model.predict(X)*scale+center).reshape(dbu.shape)
    out['conversion']=np.clip(out['conversion'],0.,1.)
    out['Mnexp']=np.maximum(out['Mnexp'],0.)
    out['dispersity']=np.maximum(out['dispersity'],1.)
    # Equivalent to V[mL] * C[mol/L] * MW[g/mol] * 1000 * (60/time[s]) * X / 1000
    # = V * C * MW * 60 / time * conversion, in mg/min.
    out['rate']=(V_ML*MONOMER_M*MW*60./seconds)*out['conversion']
    out['rateDBU']=out['rate']/dbu  # mg/min per mM DBU; identical convention to uploaded csv
    return out

df=load_data()
init_options=sorted(df.initiator.unique().tolist())
initiator=st.selectbox('Initiator concentration (mM)',init_options,index=init_options.index(10.) if 10. in init_options else 0)
_,_,train=model_for(initiator)
lo_db,hi_db=float(train.catalyst.min()),float(train.catalyst.max())
lo_t,hi_t=float(train.time.min()),float(train.time.max())

with st.expander('Dataset and units'):
    st.write(f'{len(df)} experiments overall; **{len(train)}** at {initiator:g} mM initiator. DBU: {lo_db:g}–{hi_db:g} mM; time: {lo_t:g}–{hi_t:g} s.')
    st.write('Mn from Mnexp (kDa); Đ unitless; conversion fraction; rate in mg/min. Rate/DBU uses DBU in **mM**, matching the uploaded rateDBU column (mg/min per mM), not mg/min/mol.')
    st.write('Reactor volume 0.8345 mL, initial lactide concentration 1 M, monomer MW 144.126 g/mol. Rates are computed from conversion, residence time and these constants.')
    st.write('The model uses RBF-kernel regularized least squares (batch-equivalent KRLS-style regression). It is not the exact historical KRLS model unless its kernel, hyperparameters and training method match yours.')

st.subheader('1 · Predict from conditions')
a,b=st.columns(2)
q_db=a.number_input('DBU concentration (mM)',min_value=lo_db,max_value=hi_db,value=float(min(max(20.,lo_db),hi_db)),step=1.)
q_t=b.number_input('Residence time (seconds)',min_value=lo_t,max_value=hi_t,value=float(min(max(240.,lo_t),hi_t)),step=10.)
p=predict(q_db,q_t,initiator)
labels=[('Mnexp','Mn (kDa)'),('dispersity','Dispersity (Đ)'),('conversion','Conversion (%)'),('rate','Production (mg/min)'),('rateDBU','Production / DBU (mg/min/mM)')]
cols=st.columns(5)
for col,(key,label) in zip(cols,labels):
    v=float(p[key]);v=v*100 if key=='conversion' else v
    col.metric(label,f'{v:.2f}' if key!='rateDBU' else f'{v:.3f}')

st.subheader('2 · Inverse design')
mode=st.radio('Target', ['Production rate (mg/min)','Conversion (%)'],horizontal=True)
# Fine, vectorized grids for range estimation and candidate initialization.
D=np.linspace(lo_db,hi_db,181)
T=np.linspace(lo_t,hi_t,181)
DD,TT=np.meshgrid(D,T)
all_p=predict(DD,TT,initiator)
target_key='rate' if mode.startswith('Production') else 'conversion'
other_key='conversion' if target_key=='rate' else 'rate'
flat_target=all_p[target_key].ravel()
min_feasible,max_feasible=float(flat_target.min()),float(flat_target.max())
unit_factor=100 if target_key=='conversion' else 1
target=st.number_input('Requested '+mode,min_value=0.,value=float(round(np.median(flat_target)*unit_factor,2)),step=1. if target_key=='rate' else 0.5)
target_internal=target/unit_factor
st.caption(f'Predicted target range in selected domain: {min_feasible*unit_factor:.2f}–{max_feasible*unit_factor:.2f} {"mg/min" if target_key=="rate" else "%"}. Find conditions meeting the target (within numerical tolerance), then maximize {"conversion" if target_key=="rate" else "production rate"}.')

def solve_inverse(requested):
    if requested < min_feasible-1e-6 or requested > max_feasible+1e-6:
        return None
    # Multiple diverse initial points closest to the target, prioritized for secondary objective.
    gap=np.abs(flat_target-requested)
    near=np.argsort(gap)[:500]
    seeds=near[np.argsort(-all_p[other_key].ravel()[near])[:14]]
    seeds=np.unique(np.r_[seeds,np.argmin(gap),np.argmax(all_p[other_key].ravel())])
    bounds=[(lo_db,hi_db),(lo_t,hi_t)]
    scale_other=max(float(all_p[other_key].max()-all_p[other_key].min()),1e-8)
    def at(x): return {k:float(v) for k,v in predict(x[0],x[1],initiator).items()}
    def fun(x):return -at(x)[other_key]/scale_other
    constraints=[{'type':'eq','fun':lambda x: (at(x)[target_key]-requested)/max(max_feasible-min_feasible,0.001)}]
    solutions=[]
    for index in seeds:
        guess=np.array([DD.ravel()[index],TT.ravel()[index]])
        res=minimize(fun,guess,method='SLSQP',bounds=bounds,constraints=constraints,options={'maxiter':120,'ftol':1e-10})
        x=np.clip(res.x,[lo_db,lo_t],[hi_db,hi_t])
        pred=at(x)
        tol=max(0.002*(max_feasible-min_feasible),1e-3 if target_key=='rate' else 0.0002)
        if abs(pred[target_key]-requested)<=tol: solutions.append((pred[other_key],x,pred))
    if solutions:
        _,x,pred=max(solutions,key=lambda x:x[0]);return x,pred
    # If target is attained on the grid at the chosen precision, return the grid candidate.
    near_tol=max(0.002*(max_feasible-min_feasible),0.001 if target_key=='rate' else 0.0002)
    eligible=np.flatnonzero(gap <=near_tol)
    if len(eligible):
        index=eligible[np.argmax(all_p[other_key].ravel()[eligible])]
        x=np.array([DD.ravel()[index],TT.ravel()[index]])
        return x,at(x)
    return None

if st.button('Find optimal conditions',type='primary'):
    result=solve_inverse(target_internal)
    if result is None:
        st.error('No feasible conditions found for that exact target within the modeled DBU/time bounds. Choose a target inside the predicted range (or try another initiator concentration).')
    else:
        x,yp=result
        st.success('Recommended operating point')
        a,b,c,d=st.columns(4)
        a.metric('DBU (mM)',f'{x[0]:.2f}')
        b.metric('Residence time (s)',f'{x[1]:.1f}')
        c.metric('Conversion',f'{yp["conversion"]*100:.2f}%')
        d.metric('Production',f'{yp["rate"]:.2f} mg/min')
        st.write(f'Mn: **{yp["Mnexp"]:.2f} kDa** · Đ: **{yp["dispersity"]:.3f}** · Production/DBU: **{yp["rateDBU"]:.3f} mg/min/mM**')

st.subheader('3 · Contour maps')
view_labels={'Mn (kDa)':'Mnexp','Dispersity (Đ)':'dispersity','Conversion (%)':'conversion','Production (mg/min)':'rate','Production / DBU (mg/min/mM)':'rateDBU'}
plot_mode=st.selectbox('Surface',list(view_labels))
key=view_labels[plot_mode]
z=all_p[key]*(100 if key=='conversion' else 1)
fig,ax=plt.subplots(figsize=(8,5))
cs=ax.contourf(DD,TT,z,levels=22,cmap='viridis')
fig.colorbar(cs,ax=ax,label=plot_mode)
ax.scatter(train.catalyst,train.time,s=12,c='white',edgecolors='black',linewidths=.35,label='Experiments')
ax.scatter([q_db],[q_t],marker='x',c='red',s=95,label='Prediction point')
ax.set(xlabel='DBU (mM)',ylabel='Residence time (s)',title=f'{plot_mode} · initiator {initiator:g} mM')
ax.legend(loc='best',fontsize=8)
st.pyplot(fig)
plt.close(fig)
st.caption('Contours are interpolated within the rectangular experimental limits. Regions between sparsely sampled points can be uncertain; no extrapolation beyond the displayed bounds is permitted.')
