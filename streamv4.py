import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from models.cbr_model import CBRModel
from models.gru_model import GRUModel
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Help Desk Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS - Dark Cosmic Nebula Theme with Glassmorphism
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main { 
        background: linear-gradient(180deg, #0a0e1a 0%, #1a1f35 30%, #2d3650 60%, #3d4a6b 100%);
        position: relative;
        overflow: hidden;
    }
    
    .stApp { 
        background: linear-gradient(180deg, #0a0e1a 0%, #1a1f35 30%, #2d3650 60%, #3d4a6b 100%);
    }
    
    /* Nebula cloud effect */
    .main::before {
        content: "";
        position: fixed;
        top: -50%;
        right: -20%;
        width: 100%;
        height: 200%;
        background: radial-gradient(ellipse at center, rgba(139, 166, 204, 0.15) 0%, transparent 60%);
        filter: blur(100px);
        pointer-events: none;
        z-index: 1;
        animation: float 20s ease-in-out infinite;
    }
    
    .main::after {
        content: "";
        position: fixed;
        bottom: -30%;
        left: -10%;
        width: 80%;
        height: 150%;
        background: radial-gradient(ellipse at center, rgba(96, 125, 166, 0.2) 0%, transparent 60%);
        filter: blur(80px);
        pointer-events: none;
        z-index: 1;
        animation: float 25s ease-in-out infinite reverse;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) translateX(0); }
        50% { transform: translateY(-30px) translateX(20px); }
    }
    
    /* Stars effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 10% 20%, rgba(255, 255, 255, 0.9), transparent),
            radial-gradient(1px 1px at 20% 80%, rgba(255, 255, 255, 0.7), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(255, 255, 255, 0.8), transparent),
            radial-gradient(2px 2px at 80% 10%, rgba(255, 255, 255, 0.6), transparent),
            radial-gradient(1px 1px at 90% 60%, rgba(255, 255, 255, 0.9), transparent),
            radial-gradient(1px 1px at 33% 70%, rgba(255, 255, 255, 0.7), transparent),
            radial-gradient(1px 1px at 66% 40%, rgba(255, 255, 255, 0.8), transparent),
            radial-gradient(2px 2px at 15% 45%, rgba(255, 255, 255, 0.6), transparent);
        background-size: 200% 200%;
        background-position: 0% 0%;
        opacity: 0.6;
        pointer-events: none;
        z-index: 1;
    }
    
    .main > div, .stApp > div {
        position: relative;
        z-index: 2;
    }
    
    /* Glassmorphism cards with cosmic glow */
    .solution-card {
        background: rgba(45, 54, 80, 0.3);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid rgba(139, 166, 204, 0.3);
        box-shadow: 
            0 8px 32px 0 rgba(0, 0, 0, 0.4),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.1);
        margin: 12px 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .solution-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 20px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(139, 166, 204, 0.4), transparent);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }
    
    .solution-card:hover {
        background: rgba(61, 74, 107, 0.4);
        transform: translateY(-4px);
        box-shadow: 
            0 12px 40px 0 rgba(139, 166, 204, 0.3),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(139, 166, 204, 0.5);
    }
    
    .solution-card h4 {
        color: #e8eef7 !important;
        font-weight: 600;
        margin-bottom: 12px;
        text-shadow: 0 2px 10px rgba(139, 166, 204, 0.3);
    }
    
    .solution-card p {
        color: rgba(232, 238, 247, 0.9);
        line-height: 1.7;
    }
    
    .solution-card strong {
        color: #8ba6cc;
        font-weight: 600;
    }
    
    h1, h2, h3 { 
        color: #e8eef7 !important; 
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 20px rgba(139, 166, 204, 0.4);
    }
    
    /* Glass buttons with cosmic glow */
    .stButton>button {
        background: rgba(61, 74, 107, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #e8eef7;
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid rgba(139, 166, 204, 0.4);
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button:hover {
        background: rgba(61, 74, 107, 0.6);
        border: 1px solid rgba(139, 166, 204, 0.6);
        box-shadow: 0 8px 25px rgba(139, 166, 204, 0.4);
        transform: translateY(-2px);
    }
    
    /* Primary button with nebula glow */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, rgba(96, 125, 166, 0.6) 0%, rgba(61, 74, 107, 0.6) 100%);
        border: 1px solid rgba(139, 166, 204, 0.5);
        box-shadow: 0 4px 20px rgba(96, 125, 166, 0.4);
    }
    
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(96, 125, 166, 0.8) 0%, rgba(61, 74, 107, 0.8) 100%);
        box-shadow: 0 8px 30px rgba(139, 166, 204, 0.6);
    }
    
    /* Glass text area */
    .stTextArea textarea {
        background: rgba(26, 31, 53, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 166, 204, 0.3);
        border-radius: 12px;
        color: #e8eef7;
        font-size: 15px;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(139, 166, 204, 0.6);
        box-shadow: 0 0 0 3px rgba(139, 166, 204, 0.2);
        background: rgba(26, 31, 53, 0.6);
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(232, 238, 247, 0.4);
    }
    
    .stTextArea label {
        color: #e8eef7 !important;
    }
    
    /* Sidebar glass effect */
    [data-testid="stSidebar"] {
        background: rgba(26, 31, 53, 0.5);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(139, 166, 204, 0.2);
        overflow-y: auto;
        max-height: 100vh;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto;
        max-height: 100vh;
        padding-bottom: 2rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #e8eef7;
    }
    
    /* Metrics with glow */
    [data-testid="stMetricValue"] {
        color: #e8eef7;
        font-size: 28px;
        font-weight: 700;
        text-shadow: 0 2px 15px rgba(139, 166, 204, 0.4);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(232, 238, 247, 0.7);
    }
    
    /* Tabs with cosmic design */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(26, 31, 53, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 6px;
        border: 1px solid rgba(139, 166, 204, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(232, 238, 247, 0.6);
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(96, 125, 166, 0.4);
        color: #e8eef7;
        box-shadow: 0 2px 10px rgba(139, 166, 204, 0.3);
    }
    
    /* Info boxes */
    .stInfo, .stSuccess, .stWarning {
        background: rgba(61, 74, 107, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(139, 166, 204, 0.3);
        color: #e8eef7;
    }
    
    .stInfo [data-testid="stMarkdownContainer"], 
    .stSuccess [data-testid="stMarkdownContainer"], 
    .stWarning [data-testid="stMarkdownContainer"] {
        color: #e8eef7;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(45, 54, 80, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(139, 166, 204, 0.3);
        color: #e8eef7;
        font-weight: 600;
    }
    
    /* Text input */
    .stTextInput input {
        background: rgba(26, 31, 53, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 166, 204, 0.3);
        border-radius: 8px;
        color: #e8eef7;
    }
    
    .stTextInput input:focus {
        border-color: rgba(139, 166, 204, 0.6);
        box-shadow: 0 0 0 2px rgba(139, 166, 204, 0.2);
    }
    
    .stTextInput label {
        color: #e8eef7 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(26, 31, 53, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 12px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #8ba6cc !important;
    }
    
    /* Markdown content */
    .element-container div[data-testid="stMarkdownContainer"] p {
        color: #e8eef7;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #e8eef7;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []

# Load models
@st.cache_resource
def load_models():
    try:
        # Load CBR Model
        cbr = CBRModel(n_neighbors=3)
        cbr.load(r'artifacts\cbr')
        
        # Load GRU Model
        gru = GRUModel(vocab_size=10000, max_len=100, num_classes=6)
        gru.load(r'artifacts\gru')
        
        # Load tokenizer
        tokenizer = joblib.load(r'artifacts\tokenizer.joblib')
        
        # Load case base for stats
        case_base = pd.read_csv(r'artifacts\cbr_casebase.csv')
        
        # Get categories
        if 'topic_category' in case_base.columns:
            categories = sorted(case_base['topic_category'].unique())
        else:
            categories = case_base['solution'].unique() if 'solution' in case_base.columns else []
        
        return cbr, gru, tokenizer, categories, case_base
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None

# Visualization Functions
def plot_similarity_scores(results):
    """Bar chart for CBR similarity scores"""
    cases = [f"Case {r['case_id']}" for r in results]
    scores = [r['similarity'] * 100 for r in results]
    
    fig = go.Figure(data=[
        go.Bar(x=cases, y=scores, 
               marker=dict(
                   color=scores,
                   colorscale=[[0, '#3d4a6b'], [0.5, '#607da6'], [1, '#8ba6cc']],
                   line=dict(color='rgba(139, 166, 204, 0.4)', width=1)
               ),
               text=[f"{s:.1f}%" for s in scores],
               textposition='auto',
               textfont=dict(color='#e8eef7', size=14, family='Inter'))
    ])
    fig.update_layout(
        title=dict(text="CBR: Top 3 Similar Cases", font=dict(color='#e8eef7', size=18, family='Inter')),
        xaxis_title="Cases",
        yaxis_title="Similarity (%)",
        plot_bgcolor='rgba(26, 31, 53, 0.3)',
        paper_bgcolor='rgba(45, 54, 80, 0.2)',
        font=dict(color='#e8eef7', family='Inter'),
        height=300,
        xaxis=dict(gridcolor='rgba(139, 166, 204, 0.1)'),
        yaxis=dict(gridcolor='rgba(139, 166, 204, 0.1)')
    )
    return fig

def plot_confidence_gauge(confidence):
    """Gauge chart for GRU confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "GRU Confidence", 'font': {'color': '#e8eef7', 'size': 18, 'family': 'Inter'}},
        number={'suffix': "%", 'font': {'color': '#e8eef7', 'size': 48, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': '#e8eef7', 'tickfont': {'color': '#e8eef7'}},
            'bar': {'color': "#8ba6cc", 'thickness': 0.75},
            'bgcolor': 'rgba(26, 31, 53, 0.4)',
            'borderwidth': 2,
            'bordercolor': 'rgba(139, 166, 204, 0.3)',
            'steps': [
                {'range': [0, 50], 'color': "rgba(45, 54, 80, 0.4)"},
                {'range': [50, 75], 'color': "rgba(61, 74, 107, 0.4)"},
                {'range': [75, 100], 'color': "rgba(96, 125, 166, 0.4)"}
            ],
            'threshold': {
                'line': {'color': "#e8eef7", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(45, 54, 80, 0.2)',
        height=300,
        font={'color': '#e8eef7', 'family': 'Inter'}
    )
    return fig

def plot_category_distribution(case_base):
    """Pie chart for categories"""
    if 'topic_category' in case_base.columns:
        counts = case_base['topic_category'].value_counts()
    elif 'solution' in case_base.columns:
        counts = case_base['solution'].value_counts().head(6)
    else:
        return None
    
    fig = px.pie(values=counts.values, names=counts.index,
                 title="Case Distribution",
                 color_discrete_sequence=['#2d3650', '#3d4a6b', '#607da6', '#8ba6cc', '#4d5a7a', '#5a6b8f'])
    fig.update_layout(
        paper_bgcolor='rgba(45, 54, 80, 0.2)',
        font=dict(color='#e8eef7', family='Inter'),
        height=300,
        showlegend=True,
        legend=dict(font=dict(color='#e8eef7'))
    )
    fig.update_traces(textfont=dict(color='#e8eef7', size=12))
    return fig


#Function for saving the feedbacks 

def save_feedback_to_file(feedback_data):
    """Save feedback to JSON file for later use"""
    import json
    with open('feedback.jsonl', 'a') as f:
        f.write(json.dumps(feedback_data) + '\n')




# Main App
def main():
    st.title("🤖 AI-Powered IT Help Desk Agent")
    st.markdown("### Intelligent Tier-1 Support with CBR & Deep Learning")
    
    # Load models
    cbr, gru, tokenizer, categories, case_base = load_models()
    
    if cbr is None or gru is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Dashboard")
        st.metric("Total Cases", len(case_base))
        st.metric("Models Active", "2")
        st.metric("Feedback Count", len(st.session_state.feedback_history))
        
        st.markdown("---")
        st.markdown("### 🎯 Models")
        st.info("**CBR**: Case-Based Reasoning")
        st.info("**GRU**: Deep Learning")
        
        if st.checkbox("📈 Show Statistics"):
            fig = plot_category_distribution(case_base)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Main Input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Describe Your Issue")
        user_query = st.text_area(
            "Issue Description:",
            height=120,
            placeholder="Example: My computer won't connect to WiFi..."
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            search_btn = st.button("🔍 Get Solutions", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    with col2:
        st.markdown("## 💡 Tips")
        st.markdown("""
        - Be specific
        - Include error messages
        - Mention device type
        - What have you tried?
        """)
    
    if clear_btn:
        st.rerun()
    
    # Process Query
    if search_btn and user_query.strip():
        with st.spinner("🔄 Analyzing..."):
            # CBR: Retrieve similar cases
            cbr_results = cbr.retrieve(user_query, top_k=3)
            
            # GRU: Predict category and solution
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = tokenizer.texts_to_sequences([user_query])
            seq_padded = pad_sequences(seq, maxlen=100, padding='post')
            class_id, confidence = gru.predict(seq_padded[0])
            
            # Map prediction to category
            predicted_category = categories[class_id] if class_id < len(categories) else "Unknown"
            
            # Solution mapping
            solution_map = {
   
    'Hardware': "Restart the device completely, check all cable connections, and update device drivers from manufacturer's website.",
    'HR Support': "kindly contact anyone in the HR department you need hr support.",
    'Access': "Reset your password using the 'Forgot Password' link and clear browser cache/cookies.",
    'Administrative rights': "Submit a formal access request through the IT help desk portal with manager approval.",
    'Miscellaneous': "This is a Miscellaneous problem you can contact th IT support",
    'Storage': "that is a storage problem you can buy a new hard instead" ,
    'Purchase': "Submit a purchase requisition through the procurement system with proper budget approval.",
    'Internal Project': "Contact the project lead or IT project coordinator for specific access and resources.",
    'Other': "Contact IT support at extension 5555 or submit a ticket through the help desk portal."
}

            gru_solution = solution_map.get(predicted_category, 'Contact IT support for assistance')
        
        st.success("✅ Analysis Complete!")
        
        # Display Results in Tabs
        tab1, tab2, tab3 = st.tabs(["🔍 CBR Results", "🤖 GRU Prediction", "📊 Comparison"])
        
        with tab1:
            st.markdown("### Case-Based Reasoning Solutions")
            
            # Similarity chart
            st.plotly_chart(plot_similarity_scores(cbr_results), use_container_width=True)
            
            # Display similar cases
            for i, result in enumerate(cbr_results, 1):
                st.markdown(f"""
                <div class="solution-card">
                    <h4>Solution {i} - Similarity: {result['similarity']*100:.1f}%</h4>
                    <p><strong>Issue:</strong> {result['description']}</p>
                    <p><strong>Solution:</strong> {result['solution']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    if st.button(f"👍 Helpful", key=f"cbr_pos_{i}"):
                        feedback = {
        'model': 'CBR',
        'case': i,
        'query': user_query,
        'suggested_solution': result['solution'],
        'feedback': 'positive',
        'timestamp': datetime.now().isoformat()
    }
                        save_feedback_to_file(feedback)
                        st.success("Feedback recorded!")
                with col_f2:
                    if st.button(f"👎 Not Helpful", key=f"cbr_neg_{i}"):
                        feedback = {
        'model': 'CBR',
        'case': i,
        'query': user_query,
        'suggested_solution': result['solution'],
        'feedback': 'negative',
        'timestamp': datetime.now().isoformat()
    }
                        save_feedback_to_file(feedback)
                        st.info("Feedback recorded!")
        
        with tab2:
            st.markdown("### GRU Deep Learning Prediction")
            
            col_g1, col_g2 = st.columns([1, 1])
            
            with col_g1:
                st.plotly_chart(plot_confidence_gauge(confidence), use_container_width=True)
            
            with col_g2:
                st.markdown(f"""
                <div class="solution-card">
                    <h4>AI Prediction</h4>
                    <p><strong>Category:</strong> {predicted_category}</p>
                    <p><strong>Solution:</strong> {gru_solution}</p>
                    <p><strong>Confidence:</strong> {confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    if st.button("👍 Helpful", key="gru_pos"):
                        feedback = {
        'model': 'GRU',
        'case': i,
        'query': user_query,
        'suggested_solution': result['solution'],
        'feedback': 'negative',
        'timestamp': datetime.now().isoformat()
    }
                        save_feedback_to_file(feedback)
                        st.success("Feedback recorded!")

                with col_r2:
                    if st.button("👎 Not Helpful", key="gru_neg"):
                        feedback = {
        'model': 'GRU',
        'case': i,
        'query': user_query,
        'suggested_solution': result['solution'],
        'feedback': 'negative',
        'timestamp': datetime.now().isoformat()
    }
                        save_feedback_to_file(feedback)
                        st.info("Feedback recorded!")
        
        with tab3:
            st.markdown("### Model Comparison")
            
            comparison_df = pd.DataFrame({
                'Model': ['CBR', 'GRU'],
                'Top Solution': [cbr_results[0]['solution'], gru_solution],
                'Confidence': [f"{cbr_results[0]['similarity']*100:.1f}%", 
                              f"{confidence*100:.1f}%"],
                'Method': ['Similarity Search', 'Deep Learning']
            })
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.markdown("### 🎯 Combined Recommendation")
            
            if cbr_results[0]['similarity'] > 0.7 and confidence > 0.7:
                st.success("✅ **High Confidence**: Both models agree on similar solutions!")
            elif cbr_results[0]['similarity'] > 0.6 or confidence > 0.6:
                st.warning("⚠️ **Moderate Confidence**: Review both solutions carefully.")
            else:
                st.info("💡 **Low Confidence**: Consider contacting IT support directly.")
            
            # Option to add new case
            st.markdown("---")
            st.markdown("### ➕ Add New Case to Knowledge Base")
            
            with st.expander("Add this query as a new case"):
                new_solution = st.text_input("Enter the solution that worked:")
                if st.button("💾 Save to Case Base"):
                    if new_solution:
                        new_case = {
                            'case_id': len(case_base) + 1,
                            'description': user_query,
                            'solution': new_solution,
                            'topic_category': predicted_category
                        }
                        cbr.add_case(new_case)
                        #actually this method not the best we used it just for interpretability
                        #we used above more optimized method which is the JSON file solution instead of o(n^2) solution
                        cbr.save('cbr')
                        st.success("✅ Case added successfully!")
                        st.balloons()
                    else:
                        st.warning("Please enter a solution first.")

if __name__ == "__main__":
    main()