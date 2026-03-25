import streamlit as st
import graphviz
import pandas as pd
import numpy as np
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="LucidNN", layout="wide", page_icon="🧠")

# --- TITLE SECTION ---
st.title("LucidNN 🧠")
st.caption("Interactive Neural Network Visualization Tool")
st.markdown("---")

# --- SESSION STATE MANAGEMENT ---
if 'hidden_layers' not in st.session_state:
    st.session_state.hidden_layers = [3] 

if 'network_data' not in st.session_state:
    st.session_state.network_data = {} 
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = {} # Weights history
if 'output_history' not in st.session_state:
    st.session_state.output_history = []   # Outputs history (Actual vs Expected)
if 'targets' not in st.session_state:
    st.session_state.targets = []          # The "Expected" values
if 'imported_topology' not in st.session_state:
    st.session_state.imported_topology = None  # None = use manual config
if 'import_source' not in st.session_state:
    st.session_state.import_source = ""

# --- HELPER FUNCTIONS ---
def get_topology(inputs, hidden, outputs):
    return [inputs] + hidden + [outputs]

def init_neuron_data(layer_idx, neuron_idx, num_prev_neurons):
    key = f"L{layer_idx}_N{neuron_idx}"
    if key not in st.session_state.network_data or \
       len(st.session_state.network_data[key]['weights']) != num_prev_neurons:
        
        st.session_state.network_data[key] = {
            "bias": np.random.uniform(-0.5, 0.5),
            "weights": [np.random.uniform(-1, 1) for _ in range(num_prev_neurons)]
        }
        
        if st.session_state.trained:
             st.session_state.trained = False
             st.session_state.training_history = {}
             st.toast(f"Network architecture changed. Model reset.", icon="⚠️")
    return key

def calculate_stats(topology):
    total_layers = len(topology)
    total_neurons = sum(topology)
    total_connections = 0
    for i in range(len(topology) - 1):
        total_connections += topology[i] * topology[i+1]
    return total_layers, total_neurons, total_connections

# --- MODEL IMPORT PARSERS ---

def _parse_torch_pth(file_bytes):
    """Parse a PyTorch state_dict (.pth/.pt). Expects fully-connected (Linear) layers."""
    import torch, io
    state_dict = torch.load(io.BytesIO(file_bytes), map_location='cpu', weights_only=True)
    layer_weights, layer_biases = {}, {}
    for k, v in state_dict.items():
        arr = v.detach().numpy()
        parts = k.rsplit('.', 1)
        param_type = parts[-1]
        layer_name = parts[0] if len(parts) > 1 else k
        if param_type == 'weight' and arr.ndim == 2:
            layer_weights[layer_name] = arr
        elif param_type == 'bias' and arr.ndim == 1:
            layer_biases[layer_name] = arr
    layer_names = list(layer_weights.keys())
    if not layer_names:
        raise ValueError("No fully-connected (2-D weight) layers found in state_dict.")
    topology = [layer_weights[layer_names[0]].shape[1]]
    network_data = {}
    for i, name in enumerate(layer_names):
        W = layer_weights[name]       # shape [out, in]
        B = layer_biases.get(name, np.zeros(W.shape[0]))
        topology.append(W.shape[0])
        for n in range(W.shape[0]):
            network_data[f"L{i+1}_N{n}"] = {
                'weights': W[n].tolist(),
                'bias': float(B[n])
            }
    return topology, network_data


def _parse_keras_h5(file_bytes):
    """Parse a Keras .h5 model — extracts Dense layer weights."""
    import h5py, io
    f = h5py.File(io.BytesIO(file_bytes), 'r')
    dense_groups = []
    def _find_kernels(name, obj):
        if isinstance(obj, h5py.Group) and 'kernel:0' in obj:
            dense_groups.append(obj)
    f.visititems(_find_kernels)
    if not dense_groups:
        f.close()
        raise ValueError("No Dense layers found in Keras model.")
    topology = [dense_groups[0]['kernel:0'][()].shape[0]]
    network_data = {}
    for i, grp in enumerate(dense_groups):
        W = grp['kernel:0'][()]   # [in, out]
        B = grp['bias:0'][()] if 'bias:0' in grp else np.zeros(W.shape[1])
        topology.append(W.shape[1])
        for n in range(W.shape[1]):
            network_data[f"L{i+1}_N{n}"] = {
                'weights': W[:, n].tolist(),
                'bias': float(B[n])
            }
    f.close()
    return topology, network_data


def _parse_lucidnn_json(file_bytes):
    """Re-import a previously exported LucidNN JSON."""
    import json as _json
    data = _json.loads(file_bytes)
    if data.get('type') == 'LUCIDNN_EXPORT':
        return data['topology'], data['network_data']
    if data.get('type') == 'INIT_NETWORK':
        net = data['network']
        topo = [net['input_size']]
        for hl in net['hidden_layers']:
            topo.append(hl['neurons'])
        topo.append(net['output_layer']['neurons'])
        nd = data.get('initial_state', {})
        return topo, nd
    raise ValueError("Not a recognised LucidNN JSON (expected LUCIDNN_EXPORT or INIT_NETWORK).")


def get_lucidnn_export_json(topology, network_data, activ_func, import_source):
    import json as _json
    import numpy as np

    full_network_data = {}

    # 🔥 Force initialize ALL neurons
    for l in range(1, len(topology)):
        prev_layer_size = topology[l - 1]
        curr_layer_size = topology[l]

        for n in range(curr_layer_size):
            key = f"L{l}_N{n}"

            # If already exists → use it
            if key in network_data:
                weights = list(network_data[key]["weights"])
                bias = float(network_data[key]["bias"])

                # Fix missing weights length
                if len(weights) != prev_layer_size:
                    weights = [np.random.uniform(-1, 1) for _ in range(prev_layer_size)]

            else:
                # 🔥 CREATE missing neuron
                weights = [np.random.uniform(-1, 1) for _ in range(prev_layer_size)]
                bias = float(np.random.uniform(-0.5, 0.5))

                # Also store back into session (important)
                network_data[key] = {
                    "weights": weights,
                    "bias": bias
                }

            full_network_data[key] = {
                "weights": weights,
                "bias": bias
            }

    export = {
        "type": "LUCIDNN_EXPORT",
        "topology": topology,
        "network_data": full_network_data,
        "metadata": {
            "activation": activ_func,
            "source": import_source or "manual",
            "total_layers": len(topology),
            "neurons_per_layer": topology
        }
    }

    return _json.dumps(export, indent=2)


# --- DIALOG: SET WEIGHTS & BIAS ---
@st.dialog("Set Weights & Bias")
def open_neuron_editor(layer_idx, neuron_idx, prev_layer_size):
    key = init_neuron_data(layer_idx, neuron_idx, prev_layer_size)
    data = st.session_state.network_data[key]

    st.subheader(f"Editing: Hidden Layer {layer_idx}, Neuron {neuron_idx+1}")
    
    # Bias
    new_bias = st.number_input("Bias", value=float(data['bias']), step=0.01, key=f"bias_{key}")
    
    st.markdown("---")
    st.markdown(f"**Weights (from previous layer: {prev_layer_size} inputs)**")
    
    # Weights
    new_weights = []
    cols = st.columns(3)
    for i in range(prev_layer_size):
        with cols[i % 3]:
            current_w_val = float(data['weights'][i])
            w = st.number_input(f"W_{i+1}", value=current_w_val, step=0.01, key=f"w_{key}_{i}")
            new_weights.append(w)
            
    if st.button("🎲 Randomize Values"):
        st.session_state.network_data[key]['bias'] = np.random.uniform(-1, 1)
        st.session_state.network_data[key]['weights'] = [np.random.uniform(-1, 1) for _ in range(prev_layer_size)]
        st.rerun()

    if st.button("Save Changes", type="primary"):
        st.session_state.network_data[key]['bias'] = new_bias
        st.session_state.network_data[key]['weights'] = new_weights
        st.rerun()

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Network Config")
    
    # 1. ARCHITECTURE
    st.subheader("Architecture")
    if st.session_state.imported_topology:
        st.info("Topology is locked to the imported model.")

    col_label, col_add = st.columns([3, 1])
    col_label.write("**Hidden Layers**")
    if col_add.button("➕"): 
        st.session_state.hidden_layers.append(3) 
    
    layers_to_remove = []
    for i, n in enumerate(st.session_state.hidden_layers):
        c1, c2 = st.columns([4, 1])
        st.session_state.hidden_layers[i] = c1.number_input(f"Layer {i+1} Neurons", 1, 10, n, key=f"h_{i}")
        if c2.button("X", key=f"rm_{i}"): 
            layers_to_remove.append(i)
    
    for i in sorted(layers_to_remove, reverse=True):
        st.session_state.hidden_layers.pop(i)
        st.rerun()
        
    st.markdown("---")
    input_nodes = st.number_input("Input Features", 1, 10, 2)
    output_nodes = st.number_input("Output Classes", 1, 10, 2)

    # 2. HYPERPARAMETERS
    st.markdown("---")
    st.subheader("Hyperparameters")
    activ_func = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh", "Softmax"])
    loss_func = st.selectbox("Loss Function", ["Mean Squared Error (MSE)", "Cross Entropy", "Hinge Loss"])
    epochs_setting = st.number_input("Number of Epochs", 10, 1000, 100)

    # 3. STATS
    st.markdown("---")
    st.subheader("Network Stats")
    topology = (
        st.session_state.imported_topology
        if st.session_state.imported_topology
        else get_topology(input_nodes, st.session_state.hidden_layers, output_nodes)
    )
    t_layers, t_neurons, t_conns = calculate_stats(topology)
    st.metric("Total Layers", t_layers)
    st.metric("Total Neurons", t_neurons)
    st.metric("Total Connections", t_conns)

    # 4. IMPORT MODEL
    st.markdown("---")
    st.subheader("Import Model")
    st.caption("Load a pre-trained model to visualise, edit weights, prune, and re-train.")
    uploaded_model = st.file_uploader(
        "Upload model file",
        type=["pth", "pt", "h5", "keras", "json"],
        key="model_uploader",
        label_visibility="collapsed"
    )
    if uploaded_model:
        if st.button("⬆ Load Model", type="secondary", use_container_width=True):
            try:
                ext = uploaded_model.name.rsplit('.', 1)[-1].lower()
                raw = uploaded_model.read()
                if ext in ('pth', 'pt'):
                    topo, nd = _parse_torch_pth(raw)
                elif ext in ('h5', 'keras'):
                    topo, nd = _parse_keras_h5(raw)
                elif ext == 'json':
                    topo, nd = _parse_lucidnn_json(raw)
                else:
                    st.error("Unsupported format.")
                    topo = None
                if topo:
                    st.session_state.imported_topology = topo
                    st.session_state.import_source = uploaded_model.name
                    st.session_state.network_data = nd
                    st.session_state.hidden_layers = list(topo[1:-1])
                    st.session_state.trained = False
                    st.session_state.training_history = {}
                    st.toast(f"Imported {uploaded_model.name}!", icon="✅")
                    st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

    if st.session_state.imported_topology:
        st.success(f"Loaded: **{st.session_state.import_source}**")
        if st.button("✖ Clear Import", use_container_width=True):
            st.session_state.imported_topology = None
            st.session_state.import_source = ""
            st.session_state.network_data = {}
            st.rerun()


# --- MAIN PAGE LAYOUT ---
col_viz, col_interact = st.columns([3, 2])

# --- LEFT COLUMN: VISUALIZATION ---
with col_viz:
    st.subheader("Network Architecture")
    show_edge_weights = st.toggle("Show edge weight values", value=False)
    
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', splines='line', bgcolor='transparent')
    
    for l_idx, count in enumerate(topology):
        with graph.subgraph(name=f'cluster_{l_idx}') as c:
            c.attr(color='white', label=f'Layer {l_idx}')
            
            if l_idx == 0:
                color = '#FFCCCC' # Light Red
                label_prefix = 'x'
            elif l_idx == len(topology)-1:
                color = '#CCFFCC' # Light Green
                label_prefix = 'y'
            else:
                color = '#FFFFCC' # Light Yellow
                label_prefix = 'N'
            
            for n_idx in range(count):
                node_label = f"{label_prefix}{n_idx+1}"
                c.node(f'{l_idx}_{n_idx}', 
                       label=node_label, 
                       shape='circle', 
                       style='filled', 
                       fillcolor=color, 
                       color='black', 
                       fontcolor='black', 
                       width='0.6', 
                       fixedsize='true')

    for l_idx in range(len(topology) - 1):
        target_l = l_idx + 1
        for n1 in range(topology[l_idx]):
            for n2 in range(topology[target_l]):
                nd_key = f"L{target_l}_N{n2}"
                w = None
                if nd_key in st.session_state.network_data:
                    ws = st.session_state.network_data[nd_key].get('weights', [])
                    if n1 < len(ws):
                        w = ws[n1]
                if w is not None:
                    # Map |w| → greyscale: strong weight = dark, weak = light
                    norm = min(abs(w) / 2.0, 1.0)
                    grey = int((1 - norm) * 210)
                    color = f"#{grey:02x}{grey:02x}{grey:02x}"
                    penwidth = str(round(0.5 + norm * 3.0, 2))
                    hover_text = f"L{l_idx}N{n1+1} -> L{target_l}N{n2+1} | weight = {w:.4f}"
                    edge_kwargs = {
                        "color": color,
                        "penwidth": penwidth,
                        "tooltip": hover_text,
                        "edgetooltip": hover_text
                    }
                    if show_edge_weights:
                        edge_kwargs["label"] = f"{w:.2f}"
                        edge_kwargs["fontsize"] = "10"
                        edge_kwargs["fontcolor"] = "#333333"
                    graph.edge(f'{l_idx}_{n1}', f'{target_l}_{n2}',
                               **edge_kwargs)
                else:
                    graph.edge(
                        f'{l_idx}_{n1}',
                        f'{target_l}_{n2}',
                        color='black',
                        tooltip=f"L{l_idx}N{n1+1} -> L{target_l}N{n2+1} | weight unavailable",
                        edgetooltip=f"L{l_idx}N{n1+1} -> L{target_l}N{n2+1} | weight unavailable"
                    )

    st.graphviz_chart(graph, use_container_width=True)

# --- RIGHT COLUMN: INTERACTION ---
with col_interact:
    st.subheader("Neuron Details")
    
    neuron_options = []
    for l in range(1, len(topology)): 
        layer_type = "Output" if l == len(topology)-1 else f"Hidden {l}"
        for n in range(topology[l]):
            neuron_options.append(f"Layer {l} ({layer_type}) - Neuron {n+1}")
            
    selected_neuron_str = st.selectbox("Select a Neuron to Inspect:", neuron_options)
    
    if selected_neuron_str:
        parts = selected_neuron_str.split(' ')
        l_idx = int(parts[1])
        n_idx = int(parts[-1]) - 1
        prev_layer_size = topology[l_idx - 1]
        
        key = init_neuron_data(l_idx, n_idx, prev_layer_size)
        curr_data = st.session_state.network_data[key]
        
        st.markdown(f"**Current Bias:** `{curr_data['bias']:.4f}`")
        st.caption("This neuron stores the incoming connection weights from the previous layer. Edit this neuron to change those weights.")
        
        with st.expander("View Weights", expanded=True):
            w_df = pd.DataFrame(curr_data['weights'], columns=["Weight Value"])
            if len(curr_data['weights']) == prev_layer_size:
                w_df.index = [f"Connection from Layer {l_idx-1} Neuron {i+1}" for i in range(prev_layer_size)]
            else:
                w_df.index = [f"Input {i+1}" for i in range(len(curr_data['weights']))]
            st.dataframe(w_df, use_container_width=True)

        if not st.session_state.trained:
            if st.button("🛠️ Edit Incoming Weights & Bias"):
                open_neuron_editor(l_idx, n_idx, prev_layer_size)
        
        else:
            st.info(f"Average Weight Over {epochs_setting} Epochs")
            
            if key in st.session_state.training_history:
                history_data = st.session_state.training_history[key]
                avg_weights = [np.mean(epoch_weights) for epoch_weights in history_data]
                
                chart_data = pd.DataFrame({
                    "Epoch": range(len(avg_weights)),
                    "Avg Weight": avg_weights
                })
                
                st.line_chart(chart_data, x="Epoch", y="Avg Weight", height=250)

# --- BOTTOM SECTION: TRAINING & RESULTS ---
st.markdown("---")

if not st.session_state.trained:
    if st.button("Train Model", type="primary"):
        with st.spinner(f"Training for {epochs_setting} epochs..."):
            time.sleep(1.0) 
            
            # --- SIMULATION START ---
            st.session_state.training_history = {}
            st.session_state.output_history = []
            
            # 1. Generate "Truth" Targets (random between 0 and 1)
            # We simulate that the network tries to reach these values
            st.session_state.targets = [round(np.random.uniform(0.1, 0.9), 4) for _ in range(output_nodes)]
            
            # 2. Simulate Training Loop
            output_hist = []
            
            for epoch in range(epochs_setting + 1):
                # Calculate simulated progress factor (0.0 to 1.0)
                # Network gets better as epoch increases
                progress = 1 - (0.95 ** epoch) # Converges to 1
                
                # A. Generate Output Predictions for this epoch
                epoch_preds = []
                for t in st.session_state.targets:
                    # Current prediction = Target + Noise
                    # Noise decreases as progress increases
                    noise = (np.random.normal(0, 0.5) * (1 - progress)) + (0.5 * (1-progress))
                    pred = t + noise
                    epoch_preds.append(pred)
                output_hist.append(epoch_preds)

                # B. Generate Weights History (Random Walk)
                for l in range(1, len(topology)):
                    for n in range(topology[l]):
                        k = f"L{l}_N{n}"
                        if k not in st.session_state.training_history:
                            st.session_state.training_history[k] = []
                        
                        prev_size = topology[l-1]
                        # Use existing weights or random start
                        base_weights = st.session_state.network_data.get(k, {}).get('weights', [0]*prev_size)
                        
                        # Perturb weights slightly based on epoch
                        current_weights = [w + np.random.normal(0, 0.01 * epoch) for w in base_weights]
                        st.session_state.training_history[k].append(current_weights)

            st.session_state.output_history = output_hist
            st.session_state.trained = True
            st.rerun()

else:
    c_reset, c_slider = st.columns([1, 4])
    
    if c_reset.button("Reset Model"):
        st.session_state.trained = False
        st.session_state.network_data = {}
        st.rerun()

    with c_slider:
        # SLIDER
        curr_epoch = st.slider("Epoch Timeline", 0, epochs_setting, epochs_setting)
        
        # Calculate Total Error for this specific epoch
        # (Consistent with the table below)
        if len(st.session_state.output_history) > curr_epoch:
            current_preds = st.session_state.output_history[curr_epoch]
            targets = st.session_state.targets
            # MSE Calculation
            mse = np.mean([(t - p)**2 for t, p in zip(targets, current_preds)])
            st.metric(label=f"Total Error (MSE) at Epoch {curr_epoch}", value=f"{mse:.5f}")

    # --- EXPECTED vs ACTUAL TABLE (Requested Feature) ---
    st.subheader(f"Output Comparison at Epoch {curr_epoch}")
    
    if len(st.session_state.output_history) > curr_epoch:
        current_preds = st.session_state.output_history[curr_epoch]
        targets = st.session_state.targets
        
        comparison_data = []
        for i, (pred, target) in enumerate(zip(current_preds, targets)):
            comparison_data.append({
                "Output Neuron": f"y{i+1}",
                "Expected (Target)": f"{target:.4f}",
                "Actual (Predicted)": f"{pred:.4f}",
                "Error (Diff)": f"{abs(target - pred):.4f}"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # --- ERROR GRAPH ---
    st.subheader("Total Error vs Epoch")
    
    # Calculate MSE for all epochs to plot graph
    mse_history = []
    for preds in st.session_state.output_history:
        mse = np.mean([(t - p)**2 for t, p in zip(st.session_state.targets, preds)])
        mse_history.append(mse)

    loss_df = pd.DataFrame({
        "Epoch": range(len(mse_history)),
        "Error": mse_history
    })
    
    st.line_chart(loss_df, x="Epoch", y="Error", height=250)

    # --- WEIGHT SUMMARY ---
    st.subheader("Layer-wise Weight Summary (Final Epoch)")
    summary_data = []
    for l in range(1, len(topology)):
        for n in range(topology[l]):
            key = f"L{l}_N{n}"
            if key in st.session_state.training_history:
                final_weights = st.session_state.training_history[key][-1]
                curr_bias = st.session_state.network_data.get(key, {}).get('bias', 0.0)
                
                summary_data.append({
                    "Layer": l,
                    "Neuron": n+1,
                    "Avg Wt": round(np.mean(final_weights), 4),
                    "Min Wt": round(np.min(final_weights), 4),
                    "Max Wt": round(np.max(final_weights), 4),
                    "Bias": round(curr_bias, 4)
                })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # --- PRUNING TOOLS ---
    st.markdown("---")
    st.subheader("⚙ Weight Pruning")
    st.caption(
        "Zero out weights whose absolute value is below the threshold — simulates pruning. "
        "After pruning, reset and re-train to observe the effect."
    )
    p_col1, p_col2, p_col3 = st.columns([2, 2, 1])
    prune_thresh = p_col1.number_input(
        "Threshold (|w| < threshold → 0)",
        min_value=0.0, max_value=10.0, value=0.1, step=0.01
    )
    layer_opts = ["All Layers"] + [f"Layer {l}" for l in range(1, len(topology))]
    prune_scope = p_col2.selectbox("Apply to", layer_opts, key="prune_scope")
    if p_col3.button("Prune", type="primary"):
        target_ls = (
            list(range(1, len(topology)))
            if prune_scope == "All Layers"
            else [int(prune_scope.split()[-1])]
        )
        pruned_count = 0
        for l in target_ls:
            for n in range(topology[l]):
                k = f"L{l}_N{n}"
                if k in st.session_state.network_data:
                    old_w = st.session_state.network_data[k]['weights']
                    new_w = [0.0 if abs(w) < prune_thresh else w for w in old_w]
                    pruned_count += sum(1 for o, nw in zip(old_w, new_w) if o != nw)
                    st.session_state.network_data[k]['weights'] = new_w
        st.success(f"Zeroed {pruned_count} weight(s). Reset and re-train to observe the effect.")
        st.session_state.trained = False
        st.rerun()

# --- EXPORT MODEL (always visible) ---
st.markdown("---")
st.subheader("⬇ Export Model")
st.caption("Download the current architecture and weights as a LucidNN JSON you can re-import later.")
export_json = get_lucidnn_export_json(
    topology, st.session_state.network_data, activ_func, st.session_state.import_source
)
st.download_button(
    label="Download lucidnn_model.json",
    data=export_json,
    file_name="lucidnn_model.json",
    mime="application/json",
    use_container_width=True
)