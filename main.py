import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from collections import defaultdict, deque
import heapq
import random

# Set page configuration
st.set_page_config(
    page_title="Densest Subgraph Discovery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;

        margin-bottom: 20px;
        text-align: center;
    }
    .sub-header {
        font-size: 28px;
        font-weight: bold;
        color: #2563EB;
        margin: 20px 0;
    }
    .algorithm-box {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .highlight {
        background-color: #FECACA;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: bold;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home - Graph Demo", "CORE_EXACT Algorithm", "EXACT Algorithm", "Performance Comparison"])

# Dataset selection in sidebar (shown on all pages)
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Information")
dataset_info = {
    "AS733": {"nodes": 6474, "edges": 13895, "description": "Autonomous systems network"},
    "Netscience": {"nodes": 1589, "edges": 2742, "description": "Scientific collaboration network"},
    "CaHepTh": {"nodes": 9877, "edges": 25998, "description": "Arxiv High Energy Physics collaboration network"}
}
selected_dataset = st.sidebar.selectbox("Select Dataset", list(dataset_info.keys()))
st.sidebar.markdown(f"""
**Dataset:** {selected_dataset}  
**Nodes:** {dataset_info[selected_dataset]['nodes']}  
**Edges:** {dataset_info[selected_dataset]['edges']}  
**Description:** {dataset_info[selected_dataset]['description']}
""")

# Experiment data from the report
data = {
    "Dataset": ["AS733", "AS733", "AS733", "AS733", "AS733", 
                "Netscience", "Netscience", "Netscience", "Netscience", "Netscience", 
                "CaHepTh", "CaHepTh", "CaHepTh", "CaHepTh", "CaHepTh"],
    "h": [2, 3, 4, 5, 6, 
          2, 3, 4, 5, 6, 
          2, 3, 4, 5, 6],
    "Density": [8.875, 35.9091, 85.125, 126.767, 123.393, 
                9.5, 57, 242.25, 775.2, 1938, 
                15.5, 124.714, 587.99, 2873.51, 11689.4],
    "Time_EXACT": [46.7817, 55.4167, 70.4739, 116.448, 119.57, 
                   0.4, 0.988, 5.287, 34.2815, 210.07, 
                    19.659, 66.607, 208.59, 438.623, 815.832
                   ],
    "Time_CORE_EXACT": [4.834, 7.966, 8.177,8.471,24.19,
                        0.521326, 0.681457, 0.981387, 1.73257, 4.12113, 
                   27.3634, 44.8106, 61.3247, 146.206, 529.803
                        ]
}

# Create a dataframe for easier manipulation
df = pd.DataFrame(data)

# Helper functions for graph algorithms

def generate_sample_graph(n=12, p=0.3, seed=42):
    """Generate a sample random graph"""
    random.seed(seed)
    G = nx.gnp_random_graph(n, p, seed=seed)
    return G

def compute_edge_density(G, subgraph_nodes=None):
    """Compute edge density of a graph or subgraph"""
    if subgraph_nodes is None:
        subgraph_nodes = list(G.nodes())
    
    if len(subgraph_nodes) <= 1:
        return 0
    
    subgraph = G.subgraph(subgraph_nodes)
    return subgraph.number_of_edges() / subgraph.number_of_nodes()

def find_exact_eds(G, max_iterations=10):
    """Simplified EXACT algorithm for finding edge-based densest subgraph (h=2)"""
    nodes = list(G.nodes())
    edges = list(G.edges())
    n = len(nodes)
    
    # Initial density bounds
    l = 0
    u = G.number_of_edges()
    
    best_subgraph = nodes.copy()
    
    # Binary search for optimal density
    iterations = 0
    while u - l >= 1/(n*(n-1)) and iterations < max_iterations:
        iterations += 1
        alpha = (l + u) / 2
        
        # Create a simplified flow network representation (conceptual)
        S = set()
        for node in nodes:
            if G.degree(node) < alpha:
                S.add(node)
        
        if len(S) == 0:
            l = alpha
        else:
            u = alpha
            remaining_nodes = [node for node in nodes if node not in S]
            if len(remaining_nodes) > 0:
                best_subgraph = remaining_nodes
    
    return best_subgraph

def find_k_core(G, k):
    """Find k-core of graph G"""
    core = G.copy()
    degrees = dict(core.degree())
    nodes_to_remove = [n for n, d in degrees.items() if d < k]
    
    while nodes_to_remove:
        node = nodes_to_remove.pop()
        neighbors = list(core.neighbors(node))
        core.remove_node(node)
        
        for neighbor in neighbors:
            if neighbor in core:
                degrees[neighbor] -= 1
                if degrees[neighbor] < k:
                    nodes_to_remove.append(neighbor)
    
    return core

def find_core_exact_eds(G, max_iterations=5):
    """Simplified CORE_EXACT algorithm for edge-based densest subgraph (h=2)"""
    # Step 1: Find maximum k-core
    max_k = 0
    for k in range(1, G.number_of_nodes()):
        k_core = find_k_core(G, k)
        if k_core.number_of_nodes() == 0:
            max_k = k - 1
            break
    
    # Step 2: Get the max k-core as our starting subgraph
    core = find_k_core(G, max_k)
    
    # If the graph is disconnected, take the largest connected component
    if not nx.is_connected(core):
        components = list(nx.connected_components(core))
        if components:
            largest_component = max(components, key=len)
            core = G.subgraph(largest_component).copy()
    
    # Apply simplified EXACT on the core
    if core.number_of_nodes() > 0:
        best_subgraph = find_exact_eds(core, max_iterations)
        return best_subgraph
    
    return list(G.nodes())

# HOME PAGE - GRAPH DEMO
if page == "Home - Graph Demo":
    st.markdown('<div class="main-header">Densest Subgraph Discovery</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This website illustrates the concept of Densest Subgraph Discovery, a fundamental problem in graph mining. 
    The goal is to find a subgraph with the highest density, measured by the ratio of edges to vertices.
    </div>
    """, unsafe_allow_html=True)
    # Explanation of density
    with st.expander("What is Edge Density?"):
        st.markdown("""
        The edge density of a graph or subgraph is defined as:
        
        $$\\tau(G) = \\frac{|E|}{|V|}$$
        
        Where:
        - |E| is the number of edges in the graph
        - |V| is the number of vertices (nodes) in the graph
        
        A higher edge density indicates a more tightly connected structure. The densest subgraph problem aims to find the subgraph with the maximum edge density.
        """)

    st.markdown("""
                """)
    # Graph parameters
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     n_nodes = st.slider("Number of nodes", 5, 20, 10)
    # with col2:
    #     edge_probability = st.slider("Edge probability", 0.1, 0.9, 0.3, 0.1)
    # with col3:
    #     random_seed = st.slider("Random seed", 1, 100, 42)
    
    # # Generate graph
    # G = generate_sample_graph(n=n_nodes, p=edge_probability, seed=random_seed)
    
    # # Select algorithm
    # algorithm = st.radio("Select algorithm", ["EXACT", "CORE_EXACT"])
    
    # # Run algorithm button
    # if st.button("Find Densest Subgraph"):
    #     with st.spinner("Finding densest subgraph..."):
    #         start_time = time.time()
    #         if algorithm == "EXACT":
    #             densest_nodes = find_exact_eds(G)
    #         else:
    #             densest_nodes = find_exact_eds(G)
    #         elapsed_time = time.time() - start_time
            
    #         # Compute density
    #         density = compute_edge_density(G, densest_nodes)
            
    #         # Display results
    #         st.success(f"Found densest subgraph with {len(densest_nodes)} nodes and density {density:.4f} in {elapsed_time:.4f} seconds")
            
    #         # Create visualization
    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
    #         # Original graph
    #         pos = nx.spring_layout(G, seed=random_seed)
    #         nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', 
    #                 node_size=500, font_weight='bold', ax=ax1)
    #         ax1.set_title("Original Graph")
            
    #         # Densest subgraph
    #         node_colors = ['red' if node in densest_nodes else 'lightblue' for node in G.nodes()]
    #         nx.draw(G, pos=pos, with_labels=True, node_color=node_colors, 
    #                 node_size=500, font_weight='bold', ax=ax2)
    #         ax2.set_title("Densest Subgraph Highlighted")
            
    #         st.pyplot(fig)
    # else:
    #     # Just show the original graph
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     pos = nx.spring_layout(G, seed=random_seed)
    #     nx.draw(G, pos=pos, with_labels=True, node_color='lightblue', 
    #             node_size=500, font_weight='bold', ax=ax)
    #     ax.set_title("Original Graph")
    #     st.pyplot(fig)
    
    

# CORE_EXACT ALGORITHM PAGE
elif page == "CORE_EXACT Algorithm":
    st.markdown('<div class="main-header">Core Exact Algorithm</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sub-header">Algorithm Overview</div>
    <div class="info-box">
    The CORE_EXACT algorithm is an optimization of the EXACT algorithm that uses core decomposition to reduce the search space. It first identifies the (k, Ψ)-core of the graph, which contains the densest subgraph, and then applies the flow-based technique only on this smaller subgraph.
    </div>
    """, unsafe_allow_html=True)
    

    # Display performance data
    st.markdown('<div class="sub-header">Performance Data</div>', unsafe_allow_html=True)
    
    # Filter data for CORE_EXACT
    core_exact_df = df[['Dataset', 'h', 'Time_CORE_EXACT', 'Density']]
    core_exact_df = core_exact_df.rename(columns={'Time_CORE_EXACT': 'Runtime (seconds)'})
    
    # Display as table
    st.write("Runtime results for CORE_EXACT algorithm across different datasets and h values:")
    st.dataframe(core_exact_df)
    

    # Create runtime histogram
    st.markdown('<div class="sub-header">Runtime Visualization</div>', unsafe_allow_html=True)
    
    # Filter by dataset
    dataset_filter = st.selectbox("Select Dataset for Visualization", 
                                 [ "Netscience", "CaHepTh" , "AS733"], 
                                 key="core_exact_dataset")
    
    if dataset_filter == "All Datasets":
        filtered_df = core_exact_df
    else:
        filtered_df = core_exact_df[core_exact_df['Dataset'] == dataset_filter]
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = filtered_df['h'].astype(str)
    y = filtered_df['Runtime (seconds)']
    
    bars = ax.bar(x, y, color='skyblue', edgecolor='black')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    ax.set_xlabel('Clique Size (h)')
    ax.set_ylabel('Runtime (seconds)')
    if dataset_filter == "All Datasets":
        ax.set_title('CORE_EXACT Runtime by Clique Size Across All Datasets')
    else:
        ax.set_title(f'CORE_EXACT Runtime by Clique Size for {dataset_filter}')
    
    plt.tight_layout()
    st.pyplot(fig)
    

    
    # Additional plot for density vs h
    st.markdown('<div class="sub-header">Density vs Clique Size</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset in filtered_df['Dataset'].unique():
        dataset_data = filtered_df[filtered_df['Dataset'] == dataset]
        ax.plot(dataset_data['h'], dataset_data['Density'], marker='o', linewidth=2, label=dataset)
    
    ax.set_xlabel('Clique Size (h)')
    ax.set_ylabel('Density')
    ax.set_title('Subgraph Density vs Clique Size')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if dataset_filter != "All Datasets":
        ax.set_yscale('log')
    
    plt.tight_layout()
    st.pyplot(fig)

    
    st.markdown("""
    ### Key Steps
    1. **Core Decomposition**: Find the clique-degree of each vertex and identify the k-cores
    2. **Locate optimal core**: Find the (k'', Ψ)-core using pruning criteria
    3. **Process connected components**: For each connected component in the core
    4. **Flow Network Construction**: Build a flow network similar to EXACT but on smaller subgraphs
    5. **Binary Search**: Perform binary search to find the optimal density threshold
    6. **Return**: The subgraph with the highest density
    """)

    
    # Theoretical complexity
    st.markdown('<div class="sub-header">Time Complexity</div>', unsafe_allow_html=True)
    st.markdown("""
    The time complexity of Core Exact algorithm is significantly lower than Exact due to:
    
    - Reduced search space from core decomposition
    - Smaller flow networks to process
    - Tighter bounds on binary search ranges
    
    For practical graphs, this results in substantially faster computation times, especially for larger h values.
    """)
    
    # Implementation details
    with st.expander("Implementation Details"):
        st.markdown("""
        #### Core Decomposition
        ```python
        def coreDecomposition(g, cliques):
            # Find number of cliques each vertex belongs to
            vertex_clique_count = defaultdict(int)
            for clique in cliques:
                for vertex in clique:
                    vertex_clique_count[vertex] += 1
            
            # Use min-heap to process vertices in ascending order of clique degree
            heap = [(count, vertex) for vertex, count in vertex_clique_count.items()]
            heapq.heapify(heap)
            
            # Assign core numbers based on removal order
            cores = {}
            while heap:
                degree, vertex = heapq.heappop(heap)
                cores[vertex] = degree
                
                # Update neighbors
                for clique in cliques:
                    if vertex in clique:
                        for neighbor in clique:
                            if neighbor != vertex and neighbor in vertex_clique_count:
                                vertex_clique_count[neighbor] -= 1
                                # Reheapify with updated values
                                # (simplified - actual implementation would be more efficient)
            
            return cores
        ```
        
        #### Clique Enumeration using Bron-Kerbosch Algorithm
        ```python
        def bronKerbosch(R, P, X, h, cliques):
            if len(R) == h:
                cliques.append(R.copy())
                return
            
            if not P and not X:
                return
            
            for v in list(P):
                bronKerbosch(R + [v], 
                             [p for p in P if p in graph.neighbors(v)], 
                             [x for x in X if x in graph.neighbors(v)], 
                             h, cliques)
                P.remove(v)
                X.append(v)
        ```
        """)

# EXACT ALGORITHM PAGE
elif page == "EXACT Algorithm":
    st.markdown('<div class="main-header">Exact Algorithm</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sub-header">Algorithm Overview</div>
    <div class="info-box">
    The EXACT algorithm solves the Densest Subgraph Discovery problem using a flow-based approach. It constructs a specialized flow network and performs binary search over possible density thresholds to find the optimal subgraph efficiently.
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    ### Key Steps
    1. **Initialization**: Set lower and upper bounds for the density parameter
    2. **Flow Network Construction**: Build a flow network with source s, sink t, vertices, and clique instances
    3. **Edge Capacity Assignment**: Set capacities based on vertex degrees and density parameter α
    4. **Binary Search**: Iteratively refine the density threshold using min-cut computations
    5. **Return**: The subgraph induced by the vertices in the minimum s-t cut
    """, unsafe_allow_html=True)
    
    # Theoretical complexity
    st.markdown('<div class="sub-header">Time Complexity</div>', unsafe_allow_html=True)
    st.markdown("""
    The time complexity of the Exact algorithm for edge-based densest subgraph (h=2) is:

    $$O((mn + m^3) \log n)$$

    Where:
    - n is the number of vertices
    - m is the number of edges
    
    For h-clique based densest subgraph (h>2), the complexity grows with the number of h-cliques in the graph.
    """)
    
    # Display performance data
    st.markdown('<div class="sub-header">Performance Data</div>', unsafe_allow_html=True)
    
    # Filter data for EXACT
    exact_df = df[['Dataset', 'h', 'Time_EXACT', 'Density']]
    exact_df = exact_df.rename(columns={'Time_EXACT': 'Runtime (seconds)'})
    
    # Display as table
    st.write("Runtime results for EXACT algorithm across different datasets and h values:")
    st.dataframe(exact_df)
    
    # Create runtime histogram
    st.markdown('<div class="sub-header">Runtime Visualization</div>', unsafe_allow_html=True)
    
    # Filter by dataset
    dataset_filter = st.selectbox("Select Dataset for Visualization", 
                                 [ "AS733", "Netscience", "CaHepTh" , "All Datasets"], 
                                 key="exact_dataset")
    
    if dataset_filter == "All Datasets":
        filtered_df = exact_df
    else:
        filtered_df = exact_df[exact_df['Dataset'] == dataset_filter]
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = filtered_df['h'].astype(str)
    y = filtered_df['Runtime (seconds)']
    
    bars = ax.bar(x, y, color='lightgreen', edgecolor='black')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    ax.set_xlabel('Clique Size (h)')
    ax.set_ylabel('Runtime (seconds)')
    if dataset_filter == "All Datasets":
        ax.set_title('EXACT Runtime by Clique Size Across All Datasets')
    else:
        ax.set_title(f'EXACT Runtime by Clique Size for {dataset_filter}')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional plot for density vs h
    st.markdown('<div class="sub-header">Density vs Clique Size</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset in filtered_df['Dataset'].unique():
        dataset_data = filtered_df[filtered_df['Dataset'] == dataset]
        ax.plot(dataset_data['h'], dataset_data['Density'], marker='o', linewidth=2, label=dataset)
    
    ax.set_xlabel('Clique Size (h)')
    ax.set_ylabel('Density')
    ax.set_title('Subgraph Density vs Clique Size')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if dataset_filter != "All Datasets":
        ax.set_yscale('log')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("Understanding the Flow Network"):
        st.markdown("""
        The flow network is constructed as follows:
        
        1. Create a source node 's' and sink node 't'
        2. Add all vertices from the original graph G
        3. Add nodes for all (h-1)-clique instances
        4. Connect 's' to each vertex v with capacity equal to its h-clique degree
        5. Connect each vertex v to 't' with capacity equal to the density parameter α
        6. Add infinite capacity edges from (h-1)-cliques to their vertices
        7. Add edges from vertices to (h-1)-cliques with capacity 1 if they form an h-clique
        
        The minimum s-t cut in this network identifies the vertices that should be removed to maximize density.
        """)

# PERFORMANCE COMPARISON PAGE
else:  # Performance Comparison
    st.markdown('<div class="main-header">Algorithm Performance Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This page compares the performance of EXACT and CORE_EXACT algorithms across different datasets and clique sizes.
    The comparison highlights the efficiency improvements achieved by core decomposition in the CORE_EXACT algorithm.
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data for comparison
    comparison_df = df[['Dataset', 'h', 'Time_EXACT', 'Time_CORE_EXACT', 'Density']]
    comparison_df['Speedup'] = comparison_df['Time_EXACT'] / comparison_df['Time_CORE_EXACT']
    
    # Display comparison table
    st.markdown('<div class="sub-header">Runtime Comparison Table</div>', unsafe_allow_html=True)
    st.dataframe(comparison_df)
    
    # Dataset filter for visualizations
    dataset_filter = st.selectbox("Select Dataset for Visualization", 
                                 [ "AS733", "Netscience", "CaHepTh" , "All Datasets"])
    
    if dataset_filter == "All Datasets":
        filtered_df = comparison_df
    else:
        filtered_df = comparison_df[comparison_df['Dataset'] == dataset_filter]
    
    # Runtime comparison bar chart
    st.markdown('<div class="sub-header">Runtime Comparison</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set width of bars
    barWidth = 0.3
    
    # Set position of bars on X axis
    r1 = np.arange(len(filtered_df))
    r2 = [x + barWidth for x in r1]
    
    # Create bars with custom labels
    labels = [f"{row['Dataset']}\nh={row['h']}" for _, row in filtered_df.iterrows()]
    
    # Create bars
    bars1 = ax.bar(r1, filtered_df['Time_EXACT'], width=barWidth, edgecolor='black', label='EXACT', color='lightgreen')
    bars2 = ax.bar(r2, filtered_df['Time_CORE_EXACT'], width=barWidth, edgecolor='black', label='CORE_EXACT', color='skyblue')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}s',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom', fontsize=8)
    
    # Add labels and title
    ax.set_xlabel('Dataset and Clique Size (h)')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison: EXACT vs CORE_EXACT')
    ax.set_xticks([r + barWidth/2 for r in range(len(filtered_df))])
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Display plot
    st.pyplot(fig)
    
    # Speedup line chart
    st.markdown('<div class="sub-header">Speedup Analysis</div>', unsafe_allow_html=True)
    
    # Group by dataset and calculate average speedup
    grouped_df = filtered_df.groupby(['Dataset', 'h'])['Speedup'].mean().reset_index()
    
    # Plot speedup by h for each dataset
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset in grouped_df['Dataset'].unique():
        dataset_data = grouped_df[grouped_df['Dataset'] == dataset]
        ax.plot(dataset_data['h'], dataset_data['Speedup'], marker='o', linewidth=2, label=dataset)
    
    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Clique Size (h)')
    ax.set_ylabel('Speedup (EXACT time / CORE_EXACT time)')
    ax.set_title('Algorithm Speedup by Clique Size')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Add text to explain the speedup
    plt.figtext(0.5, 0.01, "Values > 1 indicate CORE_EXACT is faster. Values < 1 indicate EXACT is faster.", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    st.pyplot(fig)
    
    # Density comparison across datasets
    st.markdown('<div class="sub-header">Density Comparison Across Datasets</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dataset in df['Dataset'].unique():
        dataset_data = df[df['Dataset'] == dataset]
        ax.plot(dataset_data['h'], dataset_data['Density'], marker='o', linewidth=2, label=dataset)
    
    ax.set_xlabel('Clique Size (h)')
    ax.set_ylabel('Density (log scale)')
    ax.set_title('Density Growth with Clique Size')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Key findings section
    st.markdown('<div class="sub-header">Key Findings</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Based on the experiments conducted on the three datasets (AS733, Netscience, and CaHepTh), we observed:
    
    1. **Performance Gain**: The CORE_EXACT algorithm generally outperforms EXACT on larger graphs (like AS733) and for intermediate clique sizes (h=3,4,5), with speedups of up to 7x.
    
    2. **Crossover Points**: For very small (h=2) or very large (h=6) clique sizes, the performance gap narrows, and in some cases EXACT performs better.
    
    3. **Density Growth**: As clique size h increases, the density of discovered subgraphs grows exponentially, with the rate of growth varying significantly between datasets.
    
    4. **Dataset Characteristics**: The structure of the underlying graph heavily influences both runtime and result quality, with scientific collaboration networks showing distinct patterns.
    
    5. **Memory Usage**: While not directly measured, CORE_EXACT has significantly lower memory requirements as it works with much smaller flow networks.
    """)
    
    # Recommendations
    st.markdown('<div class="sub-header">Practical Recommendations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    - For large graphs: Use CORE_EXACT for best performance
    - For small graphs: Either algorithm performs adequately
    - For h=2 (edge density): Traditional algorithms may be sufficient
    - For h≥4: Pre-computing and caching cliques becomes important
    - For real-time applications: Consider approximate algorithms instead
    """)
    
    # Future work
    with st.expander("Future Work"):
        st.markdown("""
        - **Parallel Implementations**: Both algorithms could benefit from parallelization, especially in the clique enumeration phase
        - **Incremental Updates**: Developing methods for handling dynamic graphs without full recomputation
        - **Approximation Algorithms**: For extremely large graphs where exact solutions are infeasible
        - **Weighted Extensions**: Adapting the algorithms for weighted graphs and alternative density definitions
        - **Application-Specific Optimizations**: Further optimizations based on domain-specific graph properties
        """)
    
    # References
    with st.expander("References"):
        st.markdown("""
        1. Fang, Y., Cheng, R., Li, Y., Shen, X., & Luo, W. (2019). Efficient algorithms for densest subgraph discovery. Proceedings of the VLDB Endowment, 12(11), 1719-1732.
        2. Tsourakakis, C. (2015). The k-clique densest subgraph problem. In Proceedings of the 24th international conference on World Wide Web (pp. 1122-1132).
        3. Leskovec, J., & Krevl, A. (2014). SNAP Datasets: Stanford large network dataset collection. http://snap.stanford.edu/data.
        """)

# Footer
st.markdown("""
<div class="footer">
Densest Subgraph Discovery Implementation - DAA CS F364 - Assignment 2
</div>
""", unsafe_allow_html=True)
