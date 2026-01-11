import sqlite3

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ShardSense Dashboard", layout="wide")

st.title("⚡ ShardSense Live Dashboard")

def get_connection():
    return sqlite3.connect("shardsense.db")

# Auto-refresh mechanism
if st.button("Refresh Data"):
    st.rerun()

try:
    conn = get_connection()
    
    # 1. Timeline of Worker Performance
    st.subheader("Worker Batch Times (ms)")
    df_workers = pd.read_sql(
        "SELECT timestamp, worker_id, batch_time_ms FROM worker_metrics ORDER BY timestamp DESC LIMIT 500", 
        conn
    )
    if not df_workers.empty:
        # Convert timestamp to something readable relative to start
        start_time = df_workers["timestamp"].min()
        df_workers["Time (s)"] = df_workers["timestamp"] - start_time
        
        chart = alt.Chart(df_workers).mark_line().encode(
            x='Time (s)',
            y='batch_time_ms',
            color='worker_id:N'
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No worker metrics found yet.")

    # 2. Current Assignment Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Shard Distribution")
        df_assign = pd.read_sql(
            "SELECT epoch, worker_id, COUNT(shard_id) as shard_count FROM assignments GROUP BY epoch, worker_id ORDER BY epoch DESC", 
            conn
        )
        
        if not df_assign.empty:
            latest_epoch = df_assign["epoch"].max()
            current_dist = df_assign[df_assign["epoch"] == latest_epoch]
            st.bar_chart(current_dist.set_index("worker_id")["shard_count"])
            st.caption(f"Epoch {latest_epoch}")
        else:
            st.info("No assignments logged yet.")

    with col2:
        st.subheader("Straggler Analysis")
        if not df_workers.empty:
             recent = df_workers.head(100)
             avg_times = recent.groupby("worker_id")["batch_time_ms"].mean().reset_index()
             st.dataframe(avg_times.style.highlight_max(axis=0, color='red'), hide_index=True)
    
    conn.close()

except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.info("Run 'python demo_real.py' to generate data.")

st.markdown("---")
st.text("Run with: streamlit run dashboard.py")
