from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langgraph.graph import END, StateGraph 

import pandas as pd
import sqlalchemy as sql
import os
import yaml
from pprint import pprint
from typing import TypedDict
from IPython.display import Image
import pandas as pd
import plotly.express as px
import sqlparse
from typing import TypedDict
import streamlit as st
from sql_agent.utils import extract_sql_code 
import sqlglot
from sqlalchemy.exc import SQLAlchemyError

os.environ["OPENAI_API_KEY"] = 'your key'

OPENAI_LLM = ChatOpenAI(
    model = "gpt-4o-mini"
)

llm = OPENAI_LLM

# mayo_recipe = llm.invoke("what is the recipe for mayo")
# pprint(mayo_recipe.content)


PATH_DB = "sqlite:///Database/northwind.db"

sql_engine = sql.create_engine(PATH_DB)

db = SQLDatabase.from_uri(PATH_DB)

sql_generator = create_sql_query_chain(
llm = llm,
db = db,
k = int(1e7)
)


class GraphState(TypedDict):
    question: str
    sql_query: str
    data: dict
    chart_type: str
    should_generate_chart: bool
    condition: str  # Added for conditional routing

def preprocess_routing(state):
    print("----Preprocess Routing----")
    question = state.get("question", "").lower()
    
    # Check for explicit keywords that strongly indicate chart requirements
    chart_keywords = ["trend", "distribution", "graph", "chart", "plot", "visualization", 
                     "compare", "comparison", "over time", "pattern", "correlation", 
                     "histogram", "bar chart", "line chart", "pie chart", "scatter plot"]
    
    # First check if any chart keywords are present in the question
    if any(keyword in question for keyword in chart_keywords):
        state["should_generate_chart"] = True
        print(f"Chart generation triggered by keyword detection in: '{question}'")
    else:
        # If no obvious keywords, use LLM for more nuanced decision
        prompt = f"""
        Determine if the following question would benefit from a chart visualization.
        Question: '{question}'
        
        Consider these factors:
        - Does it ask about trends, patterns, distributions, or comparisons?
        - Is the user trying to understand relationships between data points?
        - Would a visual representation make the answer clearer?
        - Is the question about how values change over time or categories?
        
        Answer with only 'yes' or 'no'.
        """
        
        response = llm.invoke(prompt)
        state["should_generate_chart"] = str(response).strip().lower() == "yes"
        # print(f"LLM chart determination for '{question}': {str(response).strip()}")
    
    return state

def generate_sql(state):
    print("----Generate SQL----")
    question = state.get("question")
    sql_query = sql_generator.invoke({"question": question})
    sql_query = extract_sql_code(sql_query)
    state["sql_query"] = sql_query  # Store the SQL query in state
    return state  # Return the full state

def convert_dataframe(state):
    print("----Convert Dataframe----")
    sql_query = state.get("sql_query")
    with sql_engine.connect() as conn:
        result = conn.execute(sql.text(sql_query))
    conn.close()
    rows = result.fetchall()
    columns = result.keys()
    df = pd.DataFrame(rows, columns=columns)
    state["data"] = df.to_dict()  # Store dataframe as dict in state
    
    # Set condition for the conditional routing
    state["condition"] = "Yes" if state.get("should_generate_chart", False) else "No"
    
    return state  # Return the full state

def instruct_chart_generator(state):
    print("----Instruct Chart Generator----")
    df = pd.DataFrame(state["data"])
    question = state.get("question", "")
    
    # Ask LLM for chart type recommendation
    response = llm.invoke(f"Determine the best chart type for the following question and data: '{question}'. Provide only the chart type name (e.g., line, bar, scatter, pie).")
    
    # Extract just the text content from the response object
    if hasattr(response, 'content'):
        chart_type = str(response.content).strip().lower()
    else:
        # If it's already a string or has some other structure
        chart_type = str(response).strip().lower()
    
    # Clean up the chart type by removing punctuation and extracting just the chart name
    import re
    chart_type = re.sub(r'[^\w\s]', '', chart_type)  # Remove punctuation
    
    # Extract just the chart type from possible longer responses
    chart_mapping = {
        'line': 'line',
        'bar': 'bar',
        'scatter': 'scatter',
        'pie': 'pie',
        'histogram': 'histogram',
        'box': 'box',
        'violin': 'violin',
        'area': 'area',
        'heatmap': 'heatmap',
        'donut': 'donut'
    }
    
    # Find the first match in the response
    matched_chart = next((chart for keyword, chart in chart_mapping.items() 
                         if keyword in chart_type), 'scatter')
    
    print(f"Chart type determined: '{matched_chart}' (from response: '{chart_type}')")
    state["chart_type"] = matched_chart
    
    return state

def generate_chart(state):
    # Convert data dictionary back to DataFrame
    df = pd.DataFrame(state["data"])
    chart_type = state.get("chart_type", "scatter")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not numeric_cols:
        state["chart_error"] = "No numeric columns available for visualization."
        return state  # Return state even if there's an error

    # Ensure we have at least one categorical column
    if not categorical_cols:
        categorical_cols = ["index"]
        df["index"] = df.index

    try:
        # Create the figure based on chart type
        if chart_type == "pie":
            fig = px.pie(df, values=numeric_cols[0], names=categorical_cols[0], 
                         title=f"Pie Chart of {numeric_cols[0]} by {categorical_cols[0]}")
        elif chart_type == "donut":
            fig = px.pie(df, values=numeric_cols[0], names=categorical_cols[0], 
                         title=f"Donut Chart of {numeric_cols[0]} by {categorical_cols[0]}",
                         hole=0.4)  
        elif hasattr(px, chart_type):
            fig = getattr(px, chart_type)(df, x=categorical_cols[0], y=numeric_cols[0], 
                                          title=f"{chart_type.capitalize()} of {numeric_cols[0]} by {categorical_cols[0]}")
        else:
            fig = px.scatter(df, x=categorical_cols[0], y=numeric_cols[0], title="Scatter Plot")
            state["chart_warning"] = f"Chart type '{chart_type}' not available. Defaulted to scatter plot."

        # Customize figure layout
        fig.update_layout(
            height=500,
            width=800,
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly_white"
        )
        
        state["chart_fig"] = fig
        
    except Exception as e:
        state["chart_error"] = f"Error creating chart: {str(e)}"

    return state

def state_printer(state):
    print("----Final State----")
    print(f"question: {state.get('question', '')}")
    print(f"SQL Query: {state.get('sql_query', '')}")
    print(f"Data: {pd.DataFrame(state.get('data', {}))}")
    return state  # Return the state

def create_workflow():

    # Define the workflow
    workflow = StateGraph(GraphState)
    # Add nodes
    workflow.add_node("preprocess_routing", preprocess_routing)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("convert_dataframe", convert_dataframe)
    workflow.add_node("instruct_chart_generator", instruct_chart_generator)
    workflow.add_node("generate_chart", generate_chart)
    workflow.add_node("state_printer", state_printer)
    
    # Set the entry point
    workflow.set_entry_point("preprocess_routing")
    
    # Define edges ensuring no cycles
    workflow.add_edge("preprocess_routing", "generate_sql")
    workflow.add_edge("generate_sql", "convert_dataframe")
    
    # **Conditional branching based on dataframe conversion results**
    workflow.add_conditional_edges(
        "convert_dataframe",
        lambda x: x["condition"],
        {
            "Yes": "instruct_chart_generator",
            "No": "state_printer",
        },
    )
    
    # Define edges only in one direction
    workflow.add_edge("instruct_chart_generator", "generate_chart")
    workflow.add_edge("generate_chart", "state_printer")
    workflow.add_edge("state_printer", END)
    
    # Compile the workflow
    return workflow.compile()
    # # Visualize the DAG
    # Image(app.get_graph().draw_mermaid_png())

app = create_workflow()

__all__ = ['app', 'GraphState']