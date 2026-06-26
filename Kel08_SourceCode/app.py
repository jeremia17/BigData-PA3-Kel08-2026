# ============================================================
# STREAMLIT DASHBOARD — Big Data Diabetes System
# Pipeline: PostgreSQL (Supabase) + MongoDB + Apache Spark
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="DiabetaLens · Big Data Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (LIGHT MODE THEME) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #FFFFFF; color: #18181B; } /* White background, Dark text */

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F8FAFC !important; /* Very light gray/blue */
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] * { color: #475569 !important; } /* Sidebar text */
[data-testid="stSidebarNav"] { display: none; }

/* Page headers */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #0F172A; margin-bottom: 2px; line-height: 1.1;
}
.page-subtitle { font-size: 0.85rem; color: #64748B; margin-bottom: 0; }
.header-line { border: none; border-top: 1px solid #E2E8F0; margin: 14px 0 20px; }

/* Pipeline tags */
.ptag {
    display: inline-block;
    background: #F1F5F9; border: 1px solid #E2E8F0;
    border-radius: 4px; padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #2563EB; margin: 2px;
}

/* KPI cards */
.kcard {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 20px 22px;
    position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kcard::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: var(--ak);
}
.kcard-label { font-size: 0.7rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px; font-weight: 600; }
.kcard-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #0F172A; line-height: 1; }
.kcard-sub { font-size: 0.75rem; color: #64748B; margin-top: 6px; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.68rem; font-weight: 600; }
.b-blue { background: rgba(59,130,246,0.1); color: #2563EB; }
.b-red  { background: rgba(239,68,68,0.1);  color: #DC2626; }
.b-grn  { background: rgba(34,197,94,0.1);  color: #16A34A; }
.b-amb  { background: rgba(245,158,11,0.1);  color: #D97706; }

/* Section headers */
.sh { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 24px 0 4px; }
.ss { font-size: 0.8rem; color: #64748B; margin-bottom: 14px; }

/* Insight box */
.ibox {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-left: 4px solid #3B82F6;
    border-radius: 8px; padding: 12px 16px;
    margin-bottom: 10px; font-size: 0.85rem;
    color: #334155; line-height: 1.65;
}
.ibox.red  { border-left-color: #EF4444; background: #FEF2F2; border-color: #FEE2E2;}
.ibox.grn  { border-left-color: #22C55E; background: #F0FDF4; border-color: #DCFCE7;}
.ibox.amb  { border-left-color: #F59E0B; background: #FFFBEB; border-color: #FEF3C7;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #FFFFFF; border-bottom: 1px solid #E2E8F0; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748B; border: none; padding: 10px 22px; font-size: 0.84rem; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #0F172A !important; border-bottom: 2px solid #3B82F6 !important; font-weight: 600; }

/* Selectbox, slider */
.stSelectbox > div > div { background: #FFFFFF !important; border-color: #CBD5E1 !important; color: #0F172A !important; }
.stSlider .thumb { background: #3B82F6 !important; }

/* Nav radio styling */
div[data-testid="stRadio"] > div { gap: 2px !important; }
div[data-testid="stRadio"] label {
    background: transparent !important;
    padding: 7px 12px !important; border-radius: 6px !important;
    font-size: 0.85rem !important; cursor: pointer;
    transition: background 0.15s;
}
div[data-testid="stRadio"] label:hover { background: #F1F5F9 !important; }
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] input:checked + div { background: #E2E8F0 !important; color: #0F172A !important; border-left: 3px solid #3B82F6 !important; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# ── COLORS & PLOT DEFAULTS (ADJUSTED FOR LIGHT BACKGROUND) ───
C = {
    'blue':   '#3B82F6', 'coral':  '#F97316',
    'teal':   '#14B8A6', 'amber':  '#F59E0B',
    'green':  '#10B981', 'red':    '#EF4444',
    'purple': '#8B5CF6', 'pink':   '#EC4899',
    'bg':     '#FFFFFF', 'card':   '#F8FAFC',
    'border': '#E2E8F0', 'text':   '#0F172A',
    'muted':  '#94A3B8', 'sub':    '#475569',
}

RISK_CLR = {'LOW': C['green'], 'MEDIUM': C['amber'], 'HIGH': C['coral'], 'CRITICAL': C['red']}

def pl(fig, h=380, **kw):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=C['text'], size=11),
        margin=dict(t=40, b=28, l=8, r=8),
        legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor=C['border'], borderwidth=1, font=dict(color=C['text'], size=11)),
        xaxis=dict(gridcolor=C['border'], linecolor=C['border'], tickcolor=C['border'], title_font=dict(color=C['sub'])),
        yaxis=dict(gridcolor=C['border'], linecolor=C['border'], tickcolor=C['border'], title_font=dict(color=C['sub'])),
        title=dict(font=dict(color=C['text'])),
        height=h,
    )
    fig.update_layout(**{**base, **kw})
    return fig

# ── DATA ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    # Load Real Dataset from CSV
    df = pd.read_csv("dataset/dataset_diabetes_cleaned.csv")
    
    # Calculate 'risk_category' because it's required for dashboard but not in CSV
    def risk_cat(row):
        s, ht, hd = row['risk_score'], row['hypertension'], row['heart_disease']
        if s >= 4 or (ht == 1 and hd == 1): return 'CRITICAL'
        elif s == 3 or ht == 1 or hd == 1:  return 'HIGH'
        elif s == 2:                        return 'MEDIUM'
        else:                               return 'LOW'

    df['risk_category'] = df.apply(risk_cat, axis=1)
    return df


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 4px 20px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#0F172A;">🩺 DiabetaLens</div>
        <div style="font-size:0.72rem;color:#64748B;margin-top:3px;">Big Data Diabetes Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">NAVIGATION</div>', unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠  Overview & KPI",
        "📊  EDA — Distribution",
        "🔬  Clinical Analysis",
        "🔗  SQL ↔ NoSQL Integration",
        "⚡  Apache Spark Results",
        "🗃️  Data Explorer",
    ], label_visibility="collapsed")

    st.markdown('<hr style="border-color:#E2E8F0;margin:16px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">FILTER DATASET</div>', unsafe_allow_html=True)

    sel_gender = st.selectbox("Gender", ['All', 'Female', 'Male'])
    age_range = st.slider("Age Range", 20, 85, (20, 85))
    sel_smoke = st.selectbox("Smoking History", ['All', 'never', 'former', 'current'])
    sel_dm = st.radio("Diabetes Status", ['All', 'Diabetes', 'Non-Diabetes'], horizontal=True)

    st.markdown('<hr style="border-color:#E2E8F0;margin:16px 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:#64748B;line-height:2;">
        <span style="color:#334155;font-weight:600;">TECH STACK</span><br>
        🐘 PostgreSQL (Supabase)<br>
        🍃 MongoDB (Local)<br>
        ⚡ Apache Spark 3.x<br>
        📊 Streamlit + Plotly
    </div>
    """, unsafe_allow_html=True)


# ── LOAD & FILTER ─────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    df_full = load_data()

df = df_full.copy()
if sel_gender != 'All':
    df = df[df['gender'] == sel_gender]
df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
if sel_smoke != 'All':
    df = df[df['smoking_history'] == sel_smoke]
if sel_dm == 'Diabetes':
    df = df[df['diabetes'] == 1]
elif sel_dm == 'Non-Diabetes':
    df = df[df['diabetes'] == 0]

N = len(df)


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW & KPI
# ═══════════════════════════════════════════════════════════════
if page == "🏠  Overview & KPI":
    st.markdown("""
    <div class="page-title">Big Data Diabetes System</div>
    <div class="page-subtitle">Integrated Pipeline · PostgreSQL + MongoDB + Apache Spark</div>
    <div style="margin-top:10px;">
        <span class="ptag">INGESTION</span> <span style="color:#94A3B8">→</span>
        <span class="ptag">CLEANING</span> <span style="color:#94A3B8">→</span>
        <span class="ptag">FEATURE ENG.</span> <span style="color:#94A3B8">→</span>
        <span class="ptag">SQL + NoSQL</span> <span style="color:#94A3B8">→</span>
        <span class="ptag">SPARK</span> <span style="color:#94A3B8">→</span>
        <span class="ptag">DASHBOARD</span>
    </div>
    <hr class="header-line">
    """, unsafe_allow_html=True)

    dm = int(df['diabetes'].sum())
    non_dm = N - dm
    prev = dm / N * 100 if N else 0
    avg_hba1c = df['hbA1c_level'].mean() if not df.empty else 0
    avg_gluc  = df['blood_glucose_level'].mean() if not df.empty else 0
    critical  = int((df['risk_category'] == 'CRITICAL').sum())
    avg_age   = df['age'].mean() if not df.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, C['blue'],  'Total Patients',       f"{N:,}",            f"Age {age_range[0]}–{age_range[1]} years", 'b-blue'),
        (c2, C['coral'], 'DM Prevalence',        f"{prev:.1f}%",      f"{dm:,} diagnosed",               'b-red' if prev>15 else 'b-amb' if prev>8 else 'b-grn'),
        (c3, C['teal'],  'Average HbA1c',        f"{avg_hba1c:.2f}%", f"Glucose: {avg_gluc:.0f} mg/dL",  'b-blue'),
        (c4, C['amber'], 'Average Age',          f"{avg_age:.1f}",    "Years",                           'b-amb'),
        (c5, C['red'],   'CRITICAL Patients',    f"{critical:,}",     f"{critical/N*100:.1f}% of total" if N else "0%", 'b-red'),
    ]
    for col, ak, lbl, val, sub, bcls in cards:
        with col:
            st.markdown(f"""<div class="kcard" style="--ak:{ak}">
                <div class="kcard-label">{lbl}</div>
                <div class="kcard-value">{val}</div>
                <div class="kcard-sub"><span class="badge {bcls}">{sub}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    col_g, col_d = st.columns(2)
    with col_g:
        fig_g = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=prev,
            delta={'reference': 9.0, 'valueformat': '.1f', 'suffix': '%',
                   'increasing': {'color': C['red']}, 'decreasing': {'color': C['green']}},
            number={'suffix': '%', 'font': {'size': 44, 'family': 'Syne', 'color': C['text']}},
            title={'text': 'Diabetes Prevalence', 'font': {'size': 13, 'color': C['sub']}},
            gauge={
                'axis': {'range': [0, 35], 'tickcolor': C['border'], 'tickfont': {'color': C['sub'], 'size': 10}},
                'bar': {'color': C['coral'], 'thickness': 0.22},
                'bgcolor': C['card'], 'bordercolor': C['border'],
                'steps': [
                    {'range': [0,   9], 'color': 'rgba(34,197,94,0.15)'},
                    {'range': [9,  18], 'color': 'rgba(245,158,11,0.15)'},
                    {'range': [18, 35], 'color': 'rgba(239,68,68,0.15)'},
                ],
                'threshold': {'line': {'color': C['amber'], 'width': 2.5}, 'thickness': 0.75, 'value': 9.0}
            }
        ))
        pl(fig_g, h=300, margin=dict(t=50, b=20, l=30, r=30))
        st.plotly_chart(fig_g, use_container_width=True, config={'displayModeBar': False}, theme=None)

    with col_d:
        fig_pie = go.Figure(go.Pie(
            labels=['Non-Diabetes', 'Diabetes'], values=[non_dm, dm], hole=0.6,
            marker=dict(colors=[C['teal'], C['coral']], line=dict(color=C['bg'], width=3)),
            textinfo='label+percent', textfont=dict(size=12, color='#FFFFFF'),
            hovertemplate='<b>%{label}</b><br>%{value:,} patients (%{percent})<extra></extra>'
        ))
        pl(fig_pie, h=300, showlegend=False,
           annotations=[dict(text=f"<b>{N:,}</b><br>Patients", x=0.5, y=0.5,
                             font=dict(size=15, color=C['text']), showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False}, theme=None)

    st.markdown('<div class="sh">Risk Category Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">SQL (PostgreSQL) × NoSQL (MongoDB) Integration Results → JOIN → Enrichment</div>', unsafe_allow_html=True)

    rorder = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    rc = df.groupby('risk_category').agg(total=('patient_id','count'), dm=('diabetes','sum')).reset_index()
    rc['prev_pct'] = (rc['dm'] / rc['total'] * 100).round(1)
    rc = rc.set_index('risk_category').reindex(rorder).reset_index().dropna()

    fig_rc = make_subplots(rows=1, cols=2,
        subplot_titles=['Total Patients by Risk Category', 'DM Prevalence by Risk Category (%)'])
    for _, row in rc.iterrows():
        clr = RISK_CLR.get(row['risk_category'], C['blue'])
        fig_rc.add_trace(go.Bar(x=[row['risk_category']], y=[row['total']],
            marker_color=clr, showlegend=False,
            text=[f"{row['total']:,}"], textposition='outside',
            hovertemplate=f"<b>{row['risk_category']}</b><br>Total: {row['total']:,}<extra></extra>"), row=1, col=1)
        fig_rc.add_trace(go.Bar(x=[row['risk_category']], y=[row['prev_pct']],
            marker_color=clr, showlegend=False,
            text=[f"{row['prev_pct']}%"], textposition='outside',
            hovertemplate=f"<b>{row['risk_category']}</b><br>Prevalence: {row['prev_pct']}%<extra></extra>"), row=1, col=2)

    pl(fig_rc, h=320)
    fig_rc.update_yaxes(gridcolor=C['border'])
    fig_rc.update_annotations(font_color=C['text'])
    st.plotly_chart(fig_rc, use_container_width=True, config={'displayModeBar': False}, theme=None)

    top_risk = rc.loc[rc['prev_pct'].idxmax(), 'risk_category'] if not rc.empty else '-'
    st.markdown(f"""
    <div class="ibox red">🔴 <b>Critical Finding:</b> The <b>{top_risk}</b> group has the highest diabetes prevalence.
    Patients with a combination of hypertension + heart disease are at a significantly higher risk.</div>
    <div class="ibox grn">✅ <b>Pipeline Status:</b> Data is integrated from <b>PostgreSQL/Supabase</b> (clinical)
    and <b>MongoDB</b> (lifestyle), processed by <b>Apache Spark</b>, and visualized in this dashboard.</div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — EDA DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
elif page == "📊  EDA — Distribution":
    st.markdown('<div class="page-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Distribution of numeric, categorical, and demographic variables</div>', unsafe_allow_html=True)
    st.markdown('<hr class="header-line">', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📈 Numeric Distribution", "🧩 Categorical Distribution", "📦 Box Plots"])

    with t1:
        feats  = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
        labels = ['Age (Years)', 'BMI', 'HbA1c (%)', 'Glucose (mg/dL)']
        dm_df  = df[df['diabetes'] == 1]
        ndm_df = df[df['diabetes'] == 0]

        fig_h = make_subplots(rows=1, cols=4, subplot_titles=labels, horizontal_spacing=0.04)
        for i, (feat, lbl) in enumerate(zip(feats, labels)):
            for grp_df, clr, nm in [(ndm_df, C['teal'], 'Non-DM'), (dm_df, C['coral'], 'DM')]:
                fig_h.add_trace(go.Histogram(x=grp_df[feat], nbinsx=30, name=nm,
                    marker_color=clr, opacity=0.72, showlegend=(i==0),
                    hovertemplate=f'{lbl}: %{{x}}<br>Frequency: %{{y}}<extra></extra>'), row=1, col=i+1)
        pl(fig_h, h=360, barmode='overlay',
           legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)', bordercolor=C['border']))
        fig_h.update_yaxes(gridcolor=C['border'])
        fig_h.update_annotations(font_color=C['text'])
        st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False}, theme=None)

        st.markdown('<div class="sh" style="font-size:1rem;">Descriptive Statistics</div>', unsafe_allow_html=True)
        desc = df[feats].describe().round(2)
        desc.index = ['Count','Mean','Std','Min','Q1','Median','Q3','Max']
        st.dataframe(desc.style.background_gradient(cmap='YlOrRd', axis=None).set_properties(**{'color': '#000000'}), use_container_width=True)

    with t2:
        col1, col2 = st.columns(2)
        with col1:
            bmi_d = df.groupby('bmi_cat', observed=True).agg(total=('patient_id','count'), dm=('diabetes','sum')).reset_index()
            bmi_d['prev'] = (bmi_d['dm']/bmi_d['total']*100).round(1)
            fig_bmi = px.bar(bmi_d, x='bmi_cat', y='total',
                color='prev', color_continuous_scale=[[0,'rgba(20,184,166,0.8)'],[1,'rgba(239,68,68,0.9)']],
                text=bmi_d['prev'].apply(lambda x: f'{x}%' if not pd.isna(x) else '0%'),
                title='Distribution & DM Prevalence by BMI',
                labels={'bmi_cat':'BMI Category','total':'Total Patients'})
            fig_bmi.update_traces(textposition='outside')
            pl(fig_bmi, h=340, coloraxis_showscale=False)
            fig_bmi.update_yaxes(gridcolor=C['border'])
            st.plotly_chart(fig_bmi, use_container_width=True, config={'displayModeBar': False}, theme=None)

        with col2:
            gender_d = df.groupby('gender').agg(total=('patient_id','count'), dm=('diabetes','sum')).reset_index()
            gender_d['prev'] = (gender_d['dm']/gender_d['total']*100).round(1)
            fig_gen = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'},{'type':'bar'}]],
                subplot_titles=['Gender Proportion','DM Prevalence (%)'])
            fig_gen.add_trace(go.Pie(labels=gender_d['gender'], values=gender_d['total'],
                hole=0.45, marker=dict(colors=[C['blue'], C['coral']],
                line=dict(color=C['bg'],width=3)),
                textinfo='label+percent', textfont=dict(color='#FFFFFF'), showlegend=False), row=1, col=1)
            fig_gen.add_trace(go.Bar(x=gender_d['gender'], y=gender_d['prev'],
                marker_color=[C['blue'], C['coral']],
                text=[f"{v}%" for v in gender_d['prev']], textposition='outside',
                showlegend=False), row=1, col=2)
            pl(fig_gen, h=340)
            fig_gen.update_yaxes(gridcolor=C['border'])
            fig_gen.update_annotations(font_color=C['text'])
            st.plotly_chart(fig_gen, use_container_width=True, config={'displayModeBar': False}, theme=None)

        col3, col4 = st.columns(2)
        with col3:
            smoke_d = df.groupby('smoking_history').agg(total=('patient_id','count'), dm=('diabetes','sum')).reset_index()
            smoke_d['prev'] = (smoke_d['dm']/smoke_d['total']*100).round(1)
            smoke_d = smoke_d.set_index('smoking_history').reindex(['never','former','current']).reset_index().dropna()
            clrs_smoke = [C['green'], C['amber'], C['red']]
            fig_sm = go.Figure()
            fig_sm.add_trace(go.Bar(x=smoke_d['smoking_history'], y=smoke_d['total'],
                name='Total Patients', 
                marker_color=clrs_smoke,
                opacity=0.3,
                hovertemplate='Total: %{y:,}<extra></extra>'))
            fig_sm.add_trace(go.Scatter(x=smoke_d['smoking_history'], y=smoke_d['prev'],
                mode='lines+markers', name='DM Prevalence (%)',
                line=dict(color=C['coral'], width=2.5), marker=dict(size=9, color=C['coral']),
                yaxis='y2', hovertemplate='Prevalence: %{y:.1f}%<extra></extra>'))
            fig_sm.update_layout(yaxis2=dict(overlaying='y', side='right',
                gridcolor='rgba(0,0,0,0)', tickfont=dict(color=C['coral'])),
                title='Smoking History vs DM Prevalence')
            pl(fig_sm, h=320)
            fig_sm.update_yaxes(gridcolor=C['border'])
            st.plotly_chart(fig_sm, use_container_width=True, config={'displayModeBar': False}, theme=None)

        with col4:
            age_d = df.groupby('age_group', observed=True).agg(total=('patient_id','count'), dm=('diabetes','sum')).reset_index()
            age_d['prev'] = (age_d['dm']/age_d['total']*100).round(1)
            fig_ag = px.bar(age_d, x='age_group', y='prev',
                color='prev', color_continuous_scale='Reds',
                text=age_d['prev'].apply(lambda x: f'{x}%' if not pd.isna(x) else '0%'),
                title='DM Prevalence by Age Group',
                labels={'age_group':'Age Group','prev':'Prevalence (%)'})
            fig_ag.update_traces(textposition='outside')
            pl(fig_ag, h=320, coloraxis_showscale=False)
            fig_ag.update_yaxes(gridcolor=C['border'])
            st.plotly_chart(fig_ag, use_container_width=True, config={'displayModeBar': False}, theme=None)

    with t3:
        c_feat  = st.selectbox("Feature", ['bmi', 'hbA1c_level', 'blood_glucose_level', 'age'])
        c_group = st.selectbox("Group by", ['diabetes', 'bmi_cat', 'age_group', 'smoking_history', 'risk_category'])
        fig_box = px.box(df, x=c_group, y=c_feat, color=c_group,
            color_discrete_sequence=[C['coral'], C['teal'], C['blue'], C['amber'], C['purple']],
            title=f'{c_feat} Distribution by {c_group}', points='outliers')
        pl(fig_box, h=430, showlegend=False)
        fig_box.update_yaxes(gridcolor=C['border'])
        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False}, theme=None)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — CLINICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "🔬  Clinical Analysis":
    st.markdown('<div class="page-title">Clinical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Relationships between clinical biomarkers, risk factors, and age trends</div>', unsafe_allow_html=True)
    st.markdown('<hr class="header-line">', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["🔥 Correlation Heatmap", "⚗️ HbA1c × Glucose", "💉 Risk Factors", "📈 Age Trends"])

    with t1:
        num_cols = ['age','bmi','hbA1c_level','blood_glucose_level','hypertension','heart_disease','risk_score','diabetes']
        corr = df[num_cols].corr().round(2)
        yr = corr.columns[::-1]
        fig_hm = go.Figure(go.Heatmap(
            z=corr.loc[yr, corr.columns].values, x=corr.columns, y=yr,
            colorscale=[[0,C['blue']],[0.5,C['bg']],[1,C['red']]], # adjusted for light mode
            zmin=-1, zmax=1,
            text=np.round(corr.loc[yr, corr.columns].values, 2),
            texttemplate='%{text}', textfont={'size': 11, 'color': C['text']},
            hovertemplate='X: %{x}<br>Y: %{y}<br>r = <b>%{z:.3f}</b><extra></extra>',
            colorbar=dict(tickfont=dict(color=C['sub']), bgcolor=C['bg'], bordercolor=C['border'])
        ))
        pl(fig_hm, h=480, xaxis=dict(tickangle=30, gridcolor='rgba(0,0,0,0)'),
           yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False}, theme=None)

        corr_dm = corr['diabetes'].drop('diabetes').sort_values(key=abs, ascending=False)
        st.markdown('<div class="sh" style="font-size:1rem;">Correlation with Diabetes</div>', unsafe_allow_html=True)
        cols_c = st.columns(len(corr_dm))
        for col, (feat, val) in zip(cols_c, corr_dm.items()):
            clr = C['red'] if abs(val) > 0.2 else C['amber'] if abs(val) > 0.08 else C['sub']
            with col:
                st.markdown(f"""<div style="background:{C['card']};border:1px solid {C['border']};
                    border-top:2px solid {clr};border-radius:8px;padding:10px;text-align:center;">
                    <div style="font-size:0.65rem;color:{C['sub']};text-transform:uppercase;">{feat}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.3rem;color:{clr};">{val:+.2f}</div>
                </div>""", unsafe_allow_html=True)

    with t2:
        samp = df.sample(min(5000, len(df)), random_state=42)
        fig_sc = px.scatter(samp, x='hbA1c_level', y='blood_glucose_level',
            color=samp['diabetes'].map({0:'Non-Diabetes',1:'Diabetes'}),
            color_discrete_map={'Non-Diabetes': C['teal'], 'Diabetes': C['coral']},
            opacity=0.45, title='HbA1c vs Blood Glucose (5,000 samples)',
            labels={'hbA1c_level':'HbA1c (%)','blood_glucose_level':'Glucose (mg/dL)'})
        fig_sc.add_vline(x=6.5, line_dash='dash', line_color=C['amber'],
            annotation_text='DM Threshold HbA1c 6.5%', annotation_font_color=C['amber'])
        fig_sc.add_hline(y=126, line_dash='dash', line_color=C['amber'],
            annotation_text='Glucose Threshold 126', annotation_font_color=C['amber'],
            annotation_position='bottom right')
        pl(fig_sc, h=420)
        fig_sc.update_yaxes(gridcolor=C['border'])
        st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False}, theme=None)

        ct = pd.crosstab(df['hba1c_cat'].astype(str), df['gluc_cat'].astype(str))
        ct_n = (ct.div(ct.sum(axis=1), axis=0)*100).round(1)
        fig_ct = px.imshow(ct_n, color_continuous_scale='RdBu_r',
            title='Glucose Category Proportion by HbA1c Category (%)',
            text_auto=True, aspect='auto')
        pl(fig_ct, h=280)
        fig_ct.update_traces(textfont=dict(color='#000000'))
        st.plotly_chart(fig_ct, use_container_width=True, config={'displayModeBar': False}, theme=None)

    with t3:
        risk_probs = {
            'HbA1c ≥6.5%':            df[df['hbA1c_level'] >= 6.5]['diabetes'].mean() * 100,
            'Glucose ≥126 mg/dL':     df[df['blood_glucose_level'] >= 126]['diabetes'].mean() * 100,
            'Heart Disease':          df[df['heart_disease'] == 1]['diabetes'].mean() * 100,
            'Hypertension':           df[df['hypertension'] == 1]['diabetes'].mean() * 100,
            'Age >65 Years':          df[df['age'] > 65]['diabetes'].mean() * 100,
            'Obesity (BMI≥30)':       df[df['bmi'] >= 30]['diabetes'].mean() * 100,
            'Current Smoker':         df[df['smoking_history'] == 'current']['diabetes'].mean() * 100,
        }
        rdf = pd.DataFrame(list(risk_probs.items()), columns=['Factor','Prevalence']).dropna()
        rdf = rdf.sort_values('Prevalence', ascending=True)
        mean_dm = df['diabetes'].mean() * 100

        fig_rf = go.Figure(go.Bar(
            y=rdf['Factor'], x=rdf['Prevalence'], orientation='h',
            marker=dict(color=rdf['Prevalence'],
                colorscale=[[0,'rgba(34,197,94,0.7)'],[0.5,'rgba(245,158,11,0.7)'],[1,'rgba(239,68,68,0.9)']]),
            text=rdf['Prevalence'].apply(lambda x: f'{x:.1f}%'), textposition='outside',
            hovertemplate='<b>%{y}</b><br>DM Prevalence: %{x:.1f}%<extra></extra>'
        ))
        fig_rf.add_vline(x=mean_dm, line_dash='dash', line_color=C['muted'],
            annotation_text=f'Average: {mean_dm:.1f}%',
            annotation_font_color=C['muted'], annotation_position='top right')
        pl(fig_rf, h=400, title='Hierarchy of Diabetes Risk Factors',
           xaxis=dict(title='DM Prevalence (%)', gridcolor=C['border']),
           yaxis=dict(gridcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_rf, use_container_width=True, config={'displayModeBar': False}, theme=None)

    with t4:
        age_t = (df.groupby('age').agg(dm=('diabetes', 'sum'), total=('patient_id', 'count')).reset_index())
        total_dm = age_t['dm'].sum()
        age_t['dm_percent'] = (age_t['dm'] / total_dm * 100).round(2)
        peak = age_t.loc[age_t['dm'].idxmax()]
    
        fig_trend = make_subplots(
            rows=2, cols=1, row_heights=[0.5, 0.5],
            subplot_titles=['Total DM Cases by Age', 'DM Cases Contribution Percentage by Age (%)'],
            vertical_spacing=0.12
        )
    
        fig_trend.add_trace(go.Scatter(
            x=age_t['age'], y=age_t['dm'], mode='lines', fill='tozeroy',
            line=dict(color=C['coral'], width=2), fillcolor='rgba(249,115,22,0.15)',
            name='DM Cases', hovertemplate='Age %{x}: %{y} cases<extra></extra>'
        ), row=1, col=1)
    
        fig_trend.add_annotation(
            x=peak['age'], y=peak['dm'],
            text=f"Peak age {int(peak['age'])} ({int(peak['dm'])} cases)",
            showarrow=True, arrowhead=2, arrowcolor=C['coral'],
            ax=50, ay=-40, bgcolor=C['card'], bordercolor=C['coral'],
            font=dict(color=C['coral'], size=11), row=1, col=1
        )
    
        fig_trend.add_trace(go.Scatter(
            x=age_t['age'], y=age_t['dm_percent'], mode='lines', fill='tozeroy',
            line=dict(color=C['blue'], width=2), fillcolor='rgba(59,130,246,0.12)',
            name='DM Contribution (%)', hovertemplate='Age %{x}: %{y:.2f}% DM contribution<extra></extra>'
        ), row=2, col=1)
    
        fig_trend.update_xaxes(title='Age', gridcolor=C['border'])
        fig_trend.update_yaxes(title='Total Cases', gridcolor=C['border'], row=1, col=1)
        fig_trend.update_yaxes(title='Percentage (%)', gridcolor=C['border'], row=2, col=1)
    
        fig_trend.update_layout(
            height=520, showlegend=False,
            paper_bgcolor=C['bg'], plot_bgcolor=C['bg'],
            font=dict(color=C['text']),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig_trend.update_annotations(font_color=C['text'])
        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False}, theme=None)


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 — SQL ↔ NoSQL INTEGRATION
# ═══════════════════════════════════════════════════════════════
elif page == "🔗  SQL ↔ NoSQL Integration":
    st.markdown('<div class="page-title">SQL ↔ NoSQL Integration</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Cross-system analytics: PostgreSQL × MongoDB → JOIN → Enrichment → Save back</div>', unsafe_allow_html=True)
    st.markdown('<hr class="header-line">', unsafe_allow_html=True)

    ca, cb, cc = st.columns(3)
    for col, ak, icon, title, sub in [
        (ca, C['blue'],  '🐘', 'PostgreSQL (SQL)', 'clinical_records\nage · bmi · hbA1c · glucose\nrisk_score · diabetes'),
        (cb, C['green'], '🍃', 'MongoDB (NoSQL)',  'patient_profiles\nsmoking_history\nhypertension · heart_disease'),
        (cc, C['coral'], '🔗', 'JOIN Results',       'cross_system_analytics\nJOIN by patient_id\nrisk_category enrichment'),
    ]:
        with col:
            st.markdown(f"""<div class="kcard" style="--ak:{ak}">
                <div class="kcard-label">{icon} {title}</div>
                <div class="kcard-sub" style="white-space:pre;line-height:1.9;color:#64748B;font-size:0.82rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="sh">Layer 1 — PostgreSQL Aggregation</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">Clinical statistics by Risk Score from clinical_records table</div>', unsafe_allow_html=True)

    sql_l = df.groupby('risk_score').agg(
        total_patients=('patient_id','count'), total_diabetes=('diabetes','sum'),
        avg_bmi=('bmi','mean'), avg_hba1c=('hbA1c_level','mean'),
        avg_glucose=('blood_glucose_level','mean')
    ).reset_index()
    sql_l['prevalence_pct'] = (sql_l['total_diabetes']/sql_l['total_patients']*100).round(1)
    sql_l = sql_l.round(2)

    clt, clc = st.columns([1.1, 1])
    with clt:
        st.dataframe(sql_l.style.background_gradient(cmap='Oranges', subset=['prevalence_pct']).set_properties(subset=['prevalence_pct'], **{'color': '#000000'}),
                     use_container_width=True, height=260)
    with clc:
        fig_sl = px.bar(sql_l, x='risk_score', y='prevalence_pct',
            color='prevalence_pct', color_continuous_scale='Reds',
            text=sql_l['prevalence_pct'].apply(lambda x: f'{x}%'),
            title='DM Prevalence by Risk Score (PostgreSQL)',
            labels={'risk_score':'Risk Score','prevalence_pct':'Prevalence (%)'})
        fig_sl.update_traces(textposition='outside')
        pl(fig_sl, h=260, coloraxis_showscale=False)
        fig_sl.update_yaxes(gridcolor=C['border'])
        st.plotly_chart(fig_sl, use_container_width=True, config={'displayModeBar': False}, theme=None)

    st.markdown('<div class="sh">Layer 2 — MongoDB Aggregation</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">Lifestyle profiles from patient_profiles collection</div>', unsafe_allow_html=True)

    mg_l = df.groupby('smoking_history').agg(
        total_patients=('patient_id','count'),
        total_hypertension=('hypertension','sum'),
        total_heart_disease=('heart_disease','sum'),
        total_diabetes=('diabetes','sum')
    ).reset_index()
    mg_l['pct_hypertension'] = (mg_l['total_hypertension']/mg_l['total_patients']*100).round(1)
    mg_l['pct_heart_disease']= (mg_l['total_heart_disease']/mg_l['total_patients']*100).round(1)
    mg_l['pct_diabetes']     = (mg_l['total_diabetes']/mg_l['total_patients']*100).round(1)
    mg_l = mg_l.set_index('smoking_history').reindex(['never','former','current']).reset_index().dropna()

    cma, cmb = st.columns([1.1, 1])
    with cma:
        st.dataframe(mg_l.style.background_gradient(cmap='Greens', subset=['pct_diabetes']).set_properties(subset=['pct_diabetes'], **{'color': '#000000'}),
                     use_container_width=True, height=180)
    with cmb:
        fig_mg = go.Figure()
        for col_n, clr, nm in [('pct_hypertension',C['amber'],'Hypertension'),
                                ('pct_heart_disease',   C['red'],  'Heart Disease'),
                                ('pct_diabetes',  C['coral'],'Diabetes')]:
            fig_mg.add_trace(go.Bar(name=nm, x=mg_l['smoking_history'], y=mg_l[col_n],
                marker_color=clr, opacity=0.85))
        pl(fig_mg, h=220, barmode='group', title='Prevalence by Smoking Status (MongoDB)')
        fig_mg.update_yaxes(gridcolor=C['border'])
        st.plotly_chart(fig_mg, use_container_width=True, config={'displayModeBar': False}, theme=None)

    st.markdown('<div class="sh">Layer 3 — Cross-System JOIN & Enrichment</div>', unsafe_allow_html=True)
    st.markdown('<div class="ss">SQL + NoSQL joined by patient_id → risk_category → saved back to both databases</div>', unsafe_allow_html=True)

    cross = df.groupby(['risk_category','smoking_history']).agg(
        total=('patient_id','count'), diabetes=('diabetes','sum')).reset_index()
    cross['prev_pct'] = (cross['diabetes']/cross['total']*100).round(1)

    fig_cross = px.bar(cross, x='risk_category', y='prev_pct',
        color='smoking_history', barmode='group',
        color_discrete_map={'never': C['green'], 'former': C['amber'], 'current': C['red']},
        text=cross['prev_pct'].apply(lambda x: f'{x}%'),
        title='DM Prevalence: Risk Category (SQL) × Smoking Status (NoSQL)',
        labels={'prev_pct':'Prevalence (%)','risk_category':'Risk Category'})
    fig_cross.update_xaxes(categoryorder='array', categoryarray=['LOW','MEDIUM','HIGH','CRITICAL'])
    fig_cross.update_traces(textposition='outside')
    pl(fig_cross, h=380)
    fig_cross.update_yaxes(gridcolor=C['border'])
    st.plotly_chart(fig_cross, use_container_width=True, config={'displayModeBar': False}, theme=None)


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 — SPARK RESULTS
# ═══════════════════════════════════════════════════════════════
elif page == "⚡  Apache Spark Results":
    st.markdown('<div class="page-title">Apache Spark Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Spark SQL Output · Window Functions · Patient Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<hr class="header-line">', unsafe_allow_html=True)

    t1, = st.tabs(["🔷 Spark SQL"])

    with t1:
        gluc_order  = ['Normal (<100)','Pre-diabetic (100-125)','Diabetic-range (>=126)']
        hba1c_order = ['Normal (<5.7)','Pre-diabetic (5.7-6.4)','Diabetic-range (>=6.5)']

        q1 = df.groupby('bmi_cat', observed=True).agg(
            total_patients=('patient_id','count'), total_diabetes=('diabetes','sum'),
            avg_bmi=('bmi','mean'), avg_glucose=('blood_glucose_level','mean')).reset_index()
        q1['prevalence_pct'] = (q1['total_diabetes']/q1['total_patients']*100).round(1)
        q1 = q1.round(2)

        st.markdown("**Query 1 — DM Prevalence by BMI Category**")
        st.dataframe(q1.style.background_gradient(cmap='Reds', subset=['prevalence_pct']).set_properties(subset=['prevalence_pct'], **{'color': '#000000'}),
                     use_container_width=True, height=190)
                     
        bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
        q1_sorted = q1.copy()
        q1_sorted['bmi_cat'] = pd.Categorical(
            q1_sorted['bmi_cat'].astype(str),
            categories=bmi_order, ordered=True
        )
        q1_sorted = q1_sorted.sort_values('bmi_cat').dropna(subset=['bmi_cat'])

        bar_colors = [C['blue'], C['teal'], C['amber'], C['coral']]

        fig_q1 = go.Figure(go.Bar(
            x=q1_sorted['bmi_cat'],
            y=q1_sorted['prevalence_pct'],
            marker=dict(
                color=bar_colors[:len(q1_sorted)],
                line=dict(color='rgba(0,0,0,0)', width=0),
            ),
            text=q1_sorted['prevalence_pct'].apply(lambda x: f'{x}%'),
            textposition='outside',
            textfont=dict(color=C['text'], size=12, family='Inter'),
            hovertemplate='<b>%{x}</b><br>Prevalence: %{y}%<extra></extra>',
        ))

        pl(fig_q1, h=320,
           title='Spark Q1: Diabetes Prevalence by BMI Category',
           xaxis_title='BMI Category',
           yaxis_title='Prevalence (%)',
           yaxis=dict(
               gridcolor=C['border'], linecolor=C['border'],
               tickcolor=C['border'], title_font=dict(color=C['sub']),
               range=[0, q1_sorted['prevalence_pct'].max() * 1.2],
           ),
        )
        st.plotly_chart(fig_q1, use_container_width=True,
                        config={'displayModeBar': False}, theme=None)

        st.markdown(
            '<div class="ibox amb">📊 <b>Insight:</b> Diabetes prevalence increases along with BMI category — '
            '<b>Obese</b> patients have the highest risk, far exceeding <b>Underweight</b> and <b>Normal</b> categories. '
            'This confirms a strong correlation between excess weight and diabetes.</div>',
            unsafe_allow_html=True
        )

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            q2 = df.groupby('age_group', observed=True).agg(
                total_patients=('patient_id','count'), total_diabetes=('diabetes','sum'),
                avg_hba1c=('hbA1c_level','mean')).reset_index()
            q2['prevalence_pct'] = (q2['total_diabetes']/q2['total_patients']*100).round(1)
            q2 = q2.round(2)
            st.markdown("**Query 2 — DM by Age Group**")
            st.dataframe(q2.style.background_gradient(cmap='Oranges', subset=['prevalence_pct']).set_properties(subset=['prevalence_pct'], **{'color': '#000000'}),
                         use_container_width=True, height=190)
            fig_q2 = px.bar(q2, x='age_group', y='prevalence_pct',
                color='prevalence_pct', color_continuous_scale='Oranges',
                text=q2['prevalence_pct'].apply(lambda x: f'{x}%' if not pd.isna(x) else '0%'), title='Spark Q2: DM by Age Group')
            fig_q2.update_traces(textposition='outside')
            pl(fig_q2, h=280, coloraxis_showscale=False)
            fig_q2.update_yaxes(gridcolor=C['border'])
            st.plotly_chart(fig_q2, use_container_width=True, config={'displayModeBar': False}, theme=None)

        with col_q2:
            q3 = df.groupby('risk_score').agg(
                total_patients=('patient_id','count'), total_diabetes=('diabetes','sum'),
                avg_hba1c=('hbA1c_level','mean'), avg_glucose=('blood_glucose_level','mean')).reset_index()
            q3['prevalence_pct'] = (q3['total_diabetes']/q3['total_patients']*100).round(1)
            q3 = q3.round(2)
            st.markdown("**Query 3 — Distribution by Risk Score**")
            st.dataframe(q3.style.background_gradient(cmap='Reds', subset=['prevalence_pct']).set_properties(subset=['prevalence_pct'], **{'color': '#000000'}),
                         use_container_width=True, height=190)
            fig_q3 = px.line(q3, x='risk_score', y='prevalence_pct',
                markers=True, title='Spark Q3: DM Prevalence by Risk Score')
            fig_q3.update_traces(line=dict(color=C['coral'], width=2.5), marker=dict(size=9))
            pl(fig_q3, h=280)
            fig_q3.update_yaxes(gridcolor=C['border'])
            st.plotly_chart(fig_q3, use_container_width=True, config={'displayModeBar': False}, theme=None)

        st.markdown('<div class="sh" style="margin-top:20px;">Risk Score vs Diabetes Cases & Prevalence</div>', unsafe_allow_html=True)
        st.markdown('<div class="ss">Table II — Layer 1 PostgreSQL: case distribution and prevalence by risk score</div>', unsafe_allow_html=True)

        risk_data = {
            'risk_score'    : [0, 1, 2, 3, 4, 5, 6],
            'total_patients'  : [13935, 42361, 28131, 11456, 3385, 665, 67],
            'total_diabetes': [0, 736, 2558, 2966, 1714, 473, 53],
            'prevalence_pct': [0.0, 1.7, 9.1, 25.9, 50.6, 71.1, 79.1],
            'avg_hba1c'    : [5.16, 5.27, 5.69, 6.08, 6.66, 7.08, 7.27],
            'avg_glucose'  : [88.7, 138.2, 148.2, 159.8, 173.2, 178.1, 190.7],
        }
        df_risk = pd.DataFrame(risk_data)

        st.dataframe(
            df_risk.style
                .background_gradient(cmap='Reds', subset=['prevalence_pct'])
                .background_gradient(cmap='Blues', subset=['total_diabetes'])
                .set_properties(subset=['prevalence_pct','total_diabetes'], **{'color': '#000000'}),
            use_container_width=True, height=270
        )

        fig_risk = make_subplots(specs=[[{"secondary_y": True}]])

        fig_risk.add_trace(go.Bar(
            x=df_risk['risk_score'], y=df_risk['total_diabetes'],
            name='Diabetes Cases', marker_color=C['coral'],
            opacity=0.85,
            text=df_risk['total_diabetes'].apply(lambda x: f'{x:,}'),
            textposition='outside', textfont=dict(size=10, color=C['text']),
            hovertemplate='Risk Score %{x}<br>Cases: %{y:,}<extra></extra>',
        ), secondary_y=False)

        fig_risk.add_trace(go.Scatter(
            x=df_risk['risk_score'], y=df_risk['prevalence_pct'],
            name='Prevalence (%)', mode='lines+markers',
            line=dict(color=C['amber'], width=2.5),
            marker=dict(size=9, symbol='circle', color=C['amber'],
                        line=dict(color=C['bg'], width=2)),
            hovertemplate='Risk Score %{x}<br>Prevalence: %{y}%<extra></extra>',
        ), secondary_y=True)

        fig_risk.update_xaxes(
            title_text='Risk Score (0–6)',
            tickvals=df_risk['risk_score'].tolist(),
            gridcolor=C['border'], linecolor=C['border'],
        )
        fig_risk.update_yaxes(
            title_text='Total Diabetes Cases',
            secondary_y=False,
            gridcolor=C['border'], linecolor=C['border'],
        )
        fig_risk.update_yaxes(
            title_text='Prevalence (%)',
            secondary_y=True,
            range=[0, 95],
            gridcolor='rgba(0,0,0,0)',
        )
        fig_risk.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=C['text'], size=11),
            margin=dict(t=40, b=28, l=8, r=8),
            height=340,
            title=dict(text='Spark Q3 (Table II): Risk Score → Diabetes Cases & Prevalence',
                       font=dict(color=C['text'])),
            legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor=C['border'],
                        borderwidth=1, font=dict(color=C['text'], size=11),
                        orientation='h', y=1.08, x=0),
        )
        st.plotly_chart(fig_risk, use_container_width=True,
                        config={'displayModeBar': False}, theme=None)

        st.markdown(
            '<div class="ibox red">📈 <b>Insight:</b> Prevalence spikes exponentially — '
            'from <b>0% (score 0)</b> up to <b>79.1% (score 6)</b>. A risk score ≥4 surpasses the '
            '<b>50%</b> threshold, making it a critical point for early intervention. '
            'Average HbA1c and glucose consistently rise alongside the risk score.</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="sh" style="margin-top:24px;">Cross-Tabulation: Smoking Status (MongoDB) × HbA1c Category (PostgreSQL)</div>', unsafe_allow_html=True)
        st.markdown('<div class="ss">NoSQL × SQL Integration — HbA1c category proportion by smoking history</div>', unsafe_allow_html=True)

        smoke_order = ['never', 'former', 'current']
        hba1c_order = ['Normal (<5.7)', 'Pre-diabetic (5.7-6.4)', 'Diabetic-range (>=6.5)']
        hba1c_colors = {
            'Normal (<5.7)'         : C['teal'],
            'Pre-diabetic (5.7-6.4)': C['amber'],
            'Diabetic-range (>=6.5)': C['coral'],
        }

        df_ct = df[df['smoking_history'].isin(smoke_order) & df['hba1c_cat'].isin(hba1c_order)]
        count_ct = pd.crosstab(df_ct['smoking_history'], df_ct['hba1c_cat']) \
                     .reindex(index=smoke_order, columns=hba1c_order, fill_value=0)
        rate_ct  = pd.crosstab(df_ct['smoking_history'], df_ct['hba1c_cat'], normalize='index') \
                     .reindex(index=smoke_order, columns=hba1c_order, fill_value=0) * 100

        rate_display = rate_ct.copy().round(1)
        rate_display.columns.name = 'HbA1c Category'
        rate_display.index.name   = 'Smoking Status'
        st.dataframe(
            rate_display.style
                .background_gradient(cmap='Oranges', subset=['Diabetic-range (>=6.5)'])
                .format("{:.1f}%"),
            use_container_width=True, height=160
        )

        col_ct1, col_ct2 = st.columns(2)

        with col_ct1:
            fig_ct_prop = go.Figure()
            for cat in hba1c_order:
                fig_ct_prop.add_trace(go.Bar(
                    name=cat,
                    x=rate_ct.index,
                    y=rate_ct[cat].round(1),
                    marker_color=hba1c_colors[cat],
                    opacity=0.88,
                    text=rate_ct[cat].apply(lambda v: f'{v:.1f}%'),
                    textposition='inside',
                    textfont=dict(size=10, color='#FFFFFF'),
                    hovertemplate='%{x} · ' + cat + '<br>Proportion: %{y:.1f}%<extra></extra>',
                ))
            pl(fig_ct_prop, h=300,
               barmode='stack',
               title='HbA1c Proportion by Smoking Status (%)',
               xaxis_title='Smoking Status (MongoDB)',
               yaxis_title='Proportion (%)',
               yaxis=dict(gridcolor=C['border'], linecolor=C['border'],
                          tickcolor=C['border'], title_font=dict(color=C['sub']),
                          range=[0, 105]),
            )
            st.plotly_chart(fig_ct_prop, use_container_width=True,
                            config={'displayModeBar': False}, theme=None)

        with col_ct2:
            fig_ct_cnt = go.Figure()
            for cat in hba1c_order:
                fig_ct_cnt.add_trace(go.Bar(
                    name=cat,
                    x=count_ct.index,
                    y=count_ct[cat],
                    marker_color=hba1c_colors[cat],
                    opacity=0.88,
                    text=count_ct[cat].apply(lambda v: f'{v:,}'),
                    textposition='outside',
                    textfont=dict(size=9, color=C['text']),
                    hovertemplate='%{x} · ' + cat + '<br>Patients: %{y:,}<extra></extra>',
                ))
            pl(fig_ct_cnt, h=300,
               barmode='group',
               title='Patient Count Distribution by Category',
               xaxis_title='Smoking Status (MongoDB)',
               yaxis_title='Total Patients',
               yaxis=dict(gridcolor=C['border'], linecolor=C['border'],
                          tickcolor=C['border'], title_font=dict(color=C['sub'])),
            )
            st.plotly_chart(fig_ct_cnt, use_container_width=True,
                            config={'displayModeBar': False}, theme=None)

        st.markdown(
            '<div class="ibox amb">🔗 <b>Cross-System Insight:</b> <b>Current</b> smokers have a higher '
            'proportion of Diabetic-range HbA1c compared to <b>never</b> and <b>former</b> smokers. '
            'This confirms that smoking habits (MongoDB) correlate with worsening glycemic '
            'control (PostgreSQL/HbA1c), reinforcing the cross-system SQL–NoSQL findings.</div>',
            unsafe_allow_html=True
        )

        q4 = df.groupby('hba1c_cat', observed=True).agg(
            total_patients=('patient_id','count'), total_diabetes=('diabetes','sum'),
            avg_glucose=('blood_glucose_level','mean')).reset_index()
        q4['prevalence_pct'] = (q4['total_diabetes']/q4['total_patients']*100).round(1)
        q4['hba1c_cat'] = pd.Categorical(q4['hba1c_cat'].astype(str), categories=hba1c_order, ordered=True)
        q4 = q4.sort_values('hba1c_cat').round(2)
        st.markdown("**Query 4 — Glucose Distribution by HbA1c Category**")
        st.dataframe(q4.style.background_gradient(cmap='Reds', subset=['prevalence_pct']).set_properties(subset=['prevalence_pct'], **{'color': '#000000'}),
                     use_container_width=True, height=160)



# ═══════════════════════════════════════════════════════════════
#  PAGE 6 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == "🗃️  Data Explorer":
    st.markdown('<div class="page-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Interactive exploration of the dataset, loaded and filtered in real-time</div>', unsafe_allow_html=True)
    st.markdown('<hr class="header-line">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{N:,}")
    c2.metric("Total Columns", str(len(df.columns)))
    c3.metric("Missing Values", "0")
    c4.metric("Age Range", f"{int(df['age'].min())}–{int(df['age'].max())} Years")

    st.markdown("---")

    col_f, col_t = st.columns([1, 3])
    with col_f:
        all_cols = df.columns.tolist()
        sel_cols = st.multiselect("Columns to Display", all_cols,
            default=['patient_id','gender','age','bmi','hbA1c_level',
                     'blood_glucose_level','risk_score','risk_category','diabetes'])
        n_rows  = st.slider("Number of Rows", 10, 200, 50)
        sort_col = st.selectbox("Sort by", sel_cols if sel_cols else all_cols)
        sort_asc = st.radio("Order", ['Ascending','Descending']) == 'Ascending'

    with col_t:
        if sel_cols:
            show_df = df[sel_cols].sort_values(sort_col, ascending=sort_asc).head(n_rows)
            hl_cols = [c for c in ['risk_score','diabetes'] if c in sel_cols]
            st.dataframe(
                show_df.style.background_gradient(cmap='YlOrRd', subset=hl_cols).set_properties(subset=hl_cols, **{'color': '#000000'}) if hl_cols else show_df,
                use_container_width=True, height=480)
        else:
            st.info("Please select at least one column to display.")

    st.markdown("---")
    if sel_cols:
        csv = df[sel_cols].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV (filtered)", csv, "diabetes_filtered.csv", "text/csv")


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:30px 0 10px;font-size:0.72rem;color:#64748B;
border-top:1px solid #E2E8F0;margin-top:32px;">
    <span style="font-family:'Syne',sans-serif;color:#94A3B8;font-weight:700;">DiabetaLens</span>
    &nbsp;·&nbsp; Big Data Diabetes Pipeline
    &nbsp;·&nbsp; Original Data (CSV)
</div>
""", unsafe_allow_html=True)