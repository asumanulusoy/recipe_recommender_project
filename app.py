import streamlit as st
import pandas as pd
from src.data_processor import DataProcessor
from src.recommender import Recommender

st.set_page_config(
    page_title="FastFood Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .recommendation-card h4 {
        color: #1f2937;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .recommendation-card p {
        color: #f8f9fa;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    .recommendation-card .restaurant {
        color: #2563eb;
        font-weight: 500;
    }
    .recommendation-card .metrics {
        color: #1f2937;
        font-weight: 500;
    }
    .recommendation-card .score {
        color: #059669;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
    st.session_state.data = st.session_state.data_processor.load_data('data/fastfood.csv')
    st.session_state.recommender = Recommender(st.session_state.data_processor)
    st.session_state.similarity_matrix = st.session_state.recommender.build_similarity_matrix()

def main():
    st.markdown('<h1 class="main-header">🍔 FastFood Recommender</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0; color: #f8f9fa;'>
            Discover your next favorite fast food item with our smart recommendation system! 
            Browse through various options, compare nutrition facts, and get personalized suggestions. 🍟
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        selected_restaurant = st.multiselect(
            "Select Restaurants",
            options=sorted(st.session_state.data['restaurant'].unique()),
            default=None,
            help="Choose one or more restaurants to filter items"
        )
        
        with st.expander("Advanced Filters"):
            healthy_option = st.checkbox(
                "Healthy Options Only",
                help="Show items with lower calories and higher protein"
            )

    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📋 Select Your Food Item")
        
        filtered_data = st.session_state.data
        if selected_restaurant:
            filtered_data = filtered_data[filtered_data['restaurant'].isin(selected_restaurant)]
        
        selected_item = st.selectbox(
            "Browse food items",
            options=filtered_data['item'].tolist(),
            help="Choose an item to see details and recommendations"
        )
        
        if selected_item:
            item_details = st.session_state.data_processor.get_item_details(selected_item)
            
            st.markdown("### 🔍 Item Details")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Restaurant", item_details['restaurant'])
                st.metric("Calories", f"{item_details['calories']}")
            with cols[1]:
                st.metric("Protein", f"{item_details['protein']}g")
                st.metric("Total Fat", f"{item_details['total_fat']}g")
            

    with col2:
        if selected_item:
            st.markdown("### 🌟 Recommended For You")
            recommendations = st.session_state.recommender.get_recommendations(
                selected_item,
                healthy_option
            )
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{i}. {rec['item']}</h4>
                            <p class="restaurant">🏪 {rec['restaurant']}</p>
                            <p class="metrics">🔥 {rec['calories']} calories | 💪 {rec['protein']}g protein</p>
                            <p class="score">Match Score: {rec['similarity_score']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No similar items found based on your criteria.")
  
if __name__ == "__main__":
    main()
