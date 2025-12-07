import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import requests
import json
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Financial Forecaster Pro",
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- DYNAMIC THEME CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .stCard {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    div[data-baseweb="select"] > div {
        cursor: pointer;
    }

    .streamlit-expanderHeader {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
    }
    /* Hide the default anchor links on headers so they don't behave like links */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
    h1, h2, h3, h4, h5, h6 { cursor: default !important; pointer-events: none !important; }
    
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'data_loaded' not in st.session_state: st.session_state['data_loaded'] = False
if 'df' not in st.session_state: st.session_state['df'] = None
if 'raw_df' not in st.session_state: st.session_state['raw_df'] = None
if 'extra_cols' not in st.session_state: st.session_state['extra_cols'] = []

# --- HELPER FUNCTIONS ---
def reset_app():
    st.session_state['data_loaded'] = False
    st.session_state['df'] = None
    st.session_state['raw_df'] = None
    st.rerun()

@st.cache_data
def load_data(file=None, api_url=None):
    df = None
    try:
        if file is not None:
            if file.name.endswith('.csv'): df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'): df = pd.read_excel(file)
            elif file.name.endswith('.json'):
                try: data = json.load(file); df = pd.DataFrame(data)
                except: file.seek(0); df = pd.read_json(file)
        
        elif api_url:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            try: 
                data = response.json()
                if isinstance(data, dict) and 'prices' in data:
                    df = pd.DataFrame(data['prices'], columns=['date', 'amount'])
                    df['date'] = pd.to_datetime(df['date'], unit='ms')
                    df['category'] = 'Market Asset'
                    df['type'] = 'Asset Value'
                else:
                    df = pd.DataFrame(data)
            except ValueError: 
                df = pd.read_csv(io.StringIO(response.text))
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def detect_columns(df):
    cols = df.columns.tolist()
    lower_cols = [c.lower() for c in cols]
    mapping = {}
    
    date_kws = ['date', 'dt', 'time', 'timestamp', 'created_at', 'day']
    for kw in date_kws:
        matches = [c for c in lower_cols if kw in c]
        if matches: mapping['date'] = cols[lower_cols.index(matches[0])]; break
    if 'date' not in mapping: mapping['date'] = cols[0]

    amt_kws = ['amount', 'amt', 'value', 'val', 'price', 'cost', 'total', 'size']
    for kw in amt_kws:
        matches = [c for c in lower_cols if kw in c]
        if matches: mapping['amount'] = cols[lower_cols.index(matches[0])]; break
    if 'amount' not in mapping: mapping['amount'] = cols[1] if len(cols) > 1 else cols[0]

    cat_kws = ['category', 'cat', 'desc', 'description', 'narrative', 'details', 'name']
    for kw in cat_kws:
        matches = [c for c in lower_cols if kw in c]
        if matches: mapping['category'] = cols[lower_cols.index(matches[0])]; break
    if 'category' not in mapping: mapping['category'] = cols[2] if len(cols) > 2 else cols[0]

    type_kws = ['type', 'txn_type', 'transaction_type', 'cr_dr', 'language', 'kind']
    for kw in type_kws:
        matches = [c for c in lower_cols if kw in c]
        if matches: mapping['type'] = cols[lower_cols.index(matches[0])]; break
    if 'type' not in mapping: mapping['type'] = None

    return mapping

def process_dataframe(df, mapping):
    standard_cols = [mapping['date'], mapping['amount'], mapping['category']]
    if mapping['type']: standard_cols.append(mapping['type'])
    extra_cols = [c for c in df.columns if c not in standard_cols]
    
    rename_dict = {
        mapping['date']: 'date',
        mapping['amount']: 'amount',
        mapping['category']: 'category'
    }
    if mapping['type']: rename_dict[mapping['type']] = 'type'
        
    processed_df = df.rename(columns=rename_dict).copy()
    processed_df['date'] = pd.to_datetime(processed_df['date'], utc=True).dt.tz_localize(None)
    processed_df['amount'] = pd.to_numeric(processed_df['amount'], errors='coerce').fillna(0)
    
    if 'type' not in processed_df.columns:
        if (processed_df['amount'] < 0).any():
            processed_df['type'] = processed_df['amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')
            processed_df['amount'] = processed_df['amount'].abs()
        else:
            processed_df['type'] = 'Expense' 
            
    processed_df = processed_df.sort_values('date')
    return processed_df, extra_cols

def get_prophet_model(df, metric_type, periods=12):
    if metric_type == 'Net Profit':
        temp = df.groupby(['date', 'type'])['amount'].sum().unstack().fillna(0)
        if 'Income' not in temp.columns: temp['Income'] = 0
        if 'Expense' not in temp.columns: temp['Expense'] = 0
        if 'Asset Value' in temp.columns: 
             temp['y'] = temp['Asset Value']
        else:
             temp['y'] = temp['Income'] - temp['Expense']
        prophet_df = temp[['y']].reset_index().rename(columns={'date': 'ds'})
    else:
        prophet_df = df[df['type'] == metric_type].groupby('date')['amount'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']

    prophet_df = prophet_df.set_index('ds').resample('M').sum().reset_index()
    if len(prophet_df) < 2: return None, None

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=periods, freq='M')
    forecast = m.predict(future)
    return m, forecast

# --- VIEW 1: LANDING PAGE ---
if not st.session_state['data_loaded']:
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.title("📈 Financial Forecaster ")
        st.markdown(" Upload Data for Analysis")
        st.divider()
        
        tab_upload, tab_api = st.tabs(["📁 Upload File", "🔗 Connect API"])
        raw_data = None
        
        with tab_upload:
            file = st.file_uploader("Drop your file here", type=['csv', 'xlsx', 'json'], label_visibility="collapsed")
            if file: raw_data = load_data(file=file)
                
        with tab_api:
            url = st.text_input("Paste API URL", placeholder="Paste the URL..")
            if url:
                if st.button("Fetch Data"):
                    with st.spinner("Connecting to API..."):
                        raw_data = load_data(api_url=url)

        if raw_data is not None:
            mapping = detect_columns(raw_data)
            processed_df, extras = process_dataframe(raw_data, mapping)
            st.session_state['df'] = processed_df
            st.session_state['raw_df'] = raw_data 
            st.session_state['extra_cols'] = extras
            st.session_state['data_loaded'] = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- VIEW 2: MAIN DASHBOARD ---
else:
    df = st.session_state['df']
    
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    col_head, col_set = st.columns([6, 2])
    with col_head: st.subheader("📈 Financial Dashboard")
    with col_set:
        with st.expander("⚙️ Settings", expanded=False):
            currency = st.selectbox("Currency", ["₹", "$", "€", "£", "¥", "Rp"], index=0)
            st.divider()
            if st.button("🔄 Reset App", use_container_width=True): reset_app()
    st.markdown("</div>", unsafe_allow_html=True)

    # --- KPI & FILTER ---
    min_date = df['date'].min(); max_date = df['date'].max()
    with st.expander("📅 Filter by Date Range", expanded=False):
        date_range = st.date_input("Select Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
        filtered_df = df.loc[mask]
    else: filtered_df = df

    inc = filtered_df[filtered_df['type'] == 'Income']['amount'].sum()
    exp = filtered_df[filtered_df['type'] == 'Expense']['amount'].sum()
    asset_val = filtered_df[filtered_df['type'] == 'Asset Value']['amount'].iloc[-1] if not filtered_df[filtered_df['type'] == 'Asset Value'].empty else 0
    net = inc - exp
    
    k1, k2, k3 = st.columns(3)
    if asset_val > 0:
         k1.metric("Current Asset Value", f"{currency}{asset_val:,.2f}")
         k2.metric("Total Income", f"{currency}{inc:,.2f}")
         k3.metric("Total Expenses", f"{currency}{exp:,.2f}")
    else:
         k1.metric("Total Income", f"{currency}{inc:,.2f}")
         k2.metric("Total Expenses", f"{currency}{exp:,.2f}")
         k3.metric("Net Profit", f"{currency}{net:,.2f}", delta_color="normal")

    st.divider()

    t1, t2, t3 = st.tabs(["📉 Historical Trends", "✨ AI Forecast", "🗃️ Raw Data"])

    # 1. HISTORICAL TAB
    with t1:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("##### Trends")
            monthly = filtered_df.groupby([pd.Grouper(key='date', freq='M'), 'type'])['amount'].sum().reset_index()
            fig_bar = px.bar(monthly, x='date', y='amount', color='type', barmode='group',
                             color_discrete_map={'Income': '#4CAF50', 'Expense': '#FF5252', 'Asset Value': '#2196F3'})
            fig_bar.update_traces(hovertemplate='%{fullData.name}<br>%{x|%d %b %Y}<br>' + f'{currency}%{{y:,.2f}}<extra></extra>')
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="gray", margin=dict(t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_b:
            st.markdown("##### Breakdown")
            pie_df = filtered_df[filtered_df['type'].isin(['Expense', 'Income'])]
            if not pie_df.empty:
                fig_pie = px.pie(pie_df, values='amount', names='category', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
                fig_pie.update_traces(hovertemplate='<b>%{label}</b><br>' + f'{currency}%{{value:,.2f}}<extra></extra>')
                fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
            else: st.info("No categorical data to display.")

        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("📈 Performance Trend")
        
        if 'Asset Value' in filtered_df['type'].unique():
             line_df = filtered_df[filtered_df['type'] == 'Asset Value']
             y_col = 'amount'
             title = 'Asset Value'
        else:
            pivot_df = monthly.pivot(index='date', columns='type', values='amount').fillna(0)
            if 'Income' not in pivot_df.columns: pivot_df['Income'] = 0
            if 'Expense' not in pivot_df.columns: pivot_df['Expense'] = 0
            pivot_df['Net Profit'] = pivot_df['Income'] - pivot_df['Expense']
            line_df = pivot_df
            y_col = 'Net Profit'
            title = 'Net Profit'
        
        fig_line = px.line(line_df, x=line_df.index if y_col == 'Net Profit' else 'date', y=y_col, markers=True)
        fig_line.update_traces(hovertemplate='%{fullData.name}<br>%{x|%d %b %Y}<br>' + f'{currency}%{{y:,.2f}}<extra></extra>')
        fig_line.update_traces(line_color='#00CC96', line_width=3)
        if y_col == 'Net Profit': fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Date", yaxis_title=title, font_color="gray")
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. AI FORECAST TAB
    with t2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        c_opt, c_slider = st.columns([3, 1])
        
        available_types = list(df['type'].unique())
        
        if 'Asset Value' in available_types:
             options = ['Asset Value']
        else:
             options = ["Compare All", "Net Profit", "Income", "Expense"]
             
        with c_opt: mode = st.radio("Predict", options, horizontal=True)
        with c_slider: horizon = st.slider("Prediction Horizon (Months)", 6, 24, 24)
        
        with st.spinner("Analyzing trends..."):
            fig_fc = go.Figure()
            palette = {'Net Profit': ('#1B5E20', '#69F0AE'), 'Income': ('#0D47A1', '#448AFF'), 'Expense': ('#B71C1C', '#FF5252'), 'Asset Value': ('#F57F17', '#FFD54F')}
            
            if mode == "Compare All":
                metrics_to_plot = ["Income", "Expense", "Net Profit"]
            else:
                metrics_to_plot = [mode]

            for metric in metrics_to_plot:
                m, fcst = get_prophet_model(df, metric, horizon)
                if fcst is not None:
                    c_hist, c_fut = palette.get(metric, ('gray', 'silver'))
                    
                    fig_fc.add_trace(go.Scatter(
                        x=m.history['ds'], y=m.history['y'], mode='lines', 
                        name=f"{metric} (Hist)", 
                        line=dict(color=c_hist, width=2),
                        hovertemplate=f'{metric} (History)<br>%{{x|%d %b %Y}}<br>{currency}%{{y:,.2f}}<extra></extra>'
                    ))
                    
                    last_hist_date = m.history['ds'].max()
                    fut_part = fcst[fcst['ds'] >= last_hist_date]
                    
                    fig_fc.add_trace(go.Scatter(
                        x=fut_part['ds'], y=fut_part['yhat'], mode='lines', 
                        name=f"{metric} (Fut)", 
                        line=dict(color=c_fut, width=3, dash='dash'),
                        hovertemplate=f'{metric} (Forecast)<br>%{{x|%d %b %Y}}<br>{currency}%{{y:,.2f}}<extra></extra>'
                    ))
            
            fig_fc.update_layout(hovermode="x unified", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="gray", margin=dict(t=30, b=10))
            st.plotly_chart(fig_fc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. RAW DATA TAB
    with t3:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        cols_to_show = ['date', 'category', 'amount', 'type']
        
        if st.session_state['extra_cols']:
            st.markdown("##### ➕ Include Extra Columns")
            selected_extras = st.multiselect("Select additional data:", st.session_state['extra_cols'])
            cols_to_show.extend(selected_extras)
        
        display_df = filtered_df[cols_to_show].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['amount'] = display_df['amount'].apply(lambda x: f"{currency}{x:,.2f}")

        st.dataframe(display_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)