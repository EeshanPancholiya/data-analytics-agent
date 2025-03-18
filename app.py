from json import loads
import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import plotly.express as px


# Set the current working directory to "desktop/BI Agent"
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import the chart generator module
from BI_agent_module import app, GraphState

# Page configuration
st.set_page_config(
    page_title="Data Analytics Assistant",
    page_icon="\U0001F4CA",
    layout="wide"
)

# Add title and description
st.title("Data Analytics Assistant")
st.markdown("""
This tool will generate SQL queries, retrieve data, and create appropriate charts automatically.
""")

# Add a separator
st.markdown("---")

# Sidebar with database info
with st.sidebar:
    st.header("Database Information")
    st.markdown("""
    
    **Tables**:
    - Categories: Product categories
    - Customers: Customer information
    - Employees: Employee details
    - OrderDetails: Line items for orders
    - Orders: Customer orders
    - Products: Product information
    - Shippers: Shipping companies
    - Suppliers: Product suppliers
    """)
    
    st.markdown("---")
    
    st.header("Example Questions")
    st.markdown("""
    - What are the top 5 best-selling products?
    - Show me monthly sales for 2018 as a line chart
    - Which customers spent the most money? Make a bar chart
    - What is the revenue share of top 5 selling products? Can you make a donut chart for it?
    """)

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display all previous queries and results in a continuous scroll layout
for idx, result in enumerate(st.session_state.conversation):
    st.markdown(f"### Query {idx + 1}: {result.get('question', 'Unknown Query')}")
    
    # Display the SQL query first
    st.subheader("SQL Query")
    st.code(result.get('sql_query', 'No SQL query generated'), language="sql")
    
    # Display Data Results
    st.subheader("Data Results")
    if result.get('data'):
        df = pd.DataFrame(result.get('data'))
        st.dataframe(df.head(20) if len(df) > 20 else df)
        if len(df) > 20:
            st.caption(f"Showing 20 of {len(df)} rows")
    
    # Show visualization if available
    if result.get('condition') == 'Yes':
        st.subheader(f"Visualization ({result.get('chart_type', 'chart').capitalize()})")
        
        try:
            # Recreate the chart from the data
                df = pd.DataFrame(result.get('data'))
                chart_type = result.get('chart_type', 'scatter')
                
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
                
                if not categorical_cols:
                    categorical_cols = ["index"]
                    df["index"] = df.index
                    
                if numeric_cols and categorical_cols:
                    if chart_type == "pie":
                        fig = px.pie(df, values=numeric_cols[0], names=categorical_cols[0])
                    elif chart_type == "donut":
                        fig = px.pie(df, values=numeric_cols[0], names=categorical_cols[0], hole=0.4)
                    elif hasattr(px, chart_type):
                        fig = getattr(px, chart_type)(df, x=categorical_cols[0], y=numeric_cols[0])
                    else:
                        fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
                        
                    fig.update_layout(height=500, margin=dict(l=50, r=50, t=50, b=50))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"A {chart_type} chart was generated based on your query.")
        except Exception as e:
            st.error(f"Error displaying chart: {str(e)}")
            
    elif result.get('condition') == 'No':
        st.subheader("Data Insight")
        st.info("Your query did not require a visualization. Review the data results for insights.")

# # Add a footer
# st.markdown("---")
# st.caption("Powered by LangGraph, LangChain, and Plotly")

# Move the question input form to the bottom of the page
st.markdown("---")
st.header("Ask a Question")
with st.form(key="question_form"):
    user_question = st.text_area(
        "",
        placeholder="Example: What are the top 5 best-selling products? Show me a bar chart.",
        height=100
    )
    submit_button = st.form_submit_button(label="Generate Insights")

# Process when the form is submitted
if submit_button:
    if user_question:
        # Show a spinner while processing
        with st.spinner("Analyzing your question, generating SQL, and creating visualizations..."):
            # Initialize the state with user's question
            initial_state = GraphState(
                question=user_question,
                sql_query="",
                data={},
                chart_type="",
                should_generate_chart=False,
                condition=""
            )
            
            # Run the workflow with the initial state
            try:
                result = app.invoke(initial_state)
                
                # Store the result in session state
                st.session_state.conversation.append(result)
                st.experimental_rerun()
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try again with a different question.")
    else:
        st.warning("Please enter a question to generate insights.")
