import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

# --- Path Configuration & Logging Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "error_log.txt") 

# Basic logging configuration (MOVED TO VERY TOP)
logging.basicConfig(
    level=logging.DEBUG, # Use DEBUG level to capture all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'), # 'w' to overwrite log file each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("--- SCRIPT EXECUTION STARTED ---")
logger.info(f"Attempting to log to: {log_file}")

# --- Constants ---
MODEL_PATH_REL = "../../../Papers/McGinn-GSP/GSP-main/GSP-main/Main/Models/MMFF_GCN.pkl"
EXCEL_PATH_REL = "../95Molecules-Muller_Corrected.xlsx"
OUTPUT_DIR_NAME = "Maginn_Sigma_Profile" 
SIGMA_RANGE = (-0.025, 0.025, 51) # Start, End, Number of points
UNIQUE_ATOM_TYPES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 42, 43, 44, 45,
    59, 61, 63, 64, 65, 66, 70, 71, 72, 74
]
ID_COLUMN_NAME = 'Amine' 
DEFAULT_SMILES_COLUMN_NAME = 'SMILES' 

# Construct absolute paths
MODEL_PATH = os.path.abspath(os.path.join(script_dir, MODEL_PATH_REL))
EXCEL_PATH = os.path.abspath(os.path.join(script_dir, EXCEL_PATH_REL))
OUTPUT_DIR = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR_NAME))

logger.debug(f"Script running in directory: {script_dir}")
logger.debug(f"Model path resolved to: {MODEL_PATH}")
logger.debug(f"Excel path resolved to: {EXCEL_PATH}")
logger.debug(f"Output directory resolved to: {OUTPUT_DIR}")


# --- Early exit for testing logging ---
# logger.info("Logging test: This message should be in the log file.")
# print("Printed to stdout: Logging test complete. Exiting.")
# import sys
# sys.exit(0) # Exit cleanly after attempting to log.


import tensorflow as tf
from spektral import transforms
from spektral.data import Dataset, Graph, BatchLoader
from spektral.layers import GCNConv, GlobalSumPool
from rdkit import Chem
from rdkit.Chem import AllChem


# --- Model Definition ---
class GCNModel(tf.keras.Model):
    """Graph Convolutional Network model for sigma profile prediction."""
    def __init__(self, conv_channels: List[int], l2_coeff: float = 0.01):
        super().__init__()
        self.layers_stack = []
        reg = tf.keras.regularizers.L2(l2_coeff)
        initializer = 'he_uniform'

        for channels in filter(None, conv_channels): 
            self.layers_stack.append(
                GCNConv(channels, activation='relu',
                        kernel_initializer=initializer, kernel_regularizer=reg, use_bias=False)
            )
        self.layers_stack.extend([
            tf.keras.layers.Dense(SIGMA_RANGE[2], activation='relu', 
                                  kernel_initializer=initializer, kernel_regularizer=reg, use_bias=False),
            GlobalSumPool()
        ])

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, a = inputs
        for layer in self.layers_stack[:-2]: 
            x = layer([x, a])
        x = self.layers_stack[-2](x) 
        return self.layers_stack[-1](x)


# --- Dataset Definition ---
class MoleculeDataset(Dataset):
    def __init__(self, graph: Graph, **kwargs):
        self.graph = graph
        super().__init__(**kwargs)

    def read(self) -> List[Graph]:
        return [self.graph]


# --- Core Functions ---
def load_gcn_model() -> Optional[GCNModel]:
    logger.info(f"Attempting to load GCN model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.critical(f"Model file not found: {MODEL_PATH}. Cannot proceed.")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            weights = pickle.load(f)
        logger.info(f"Successfully loaded model weights from {MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Critical error loading model weights from {MODEL_PATH}: {e}", exc_info=True)
        return None

    try:
        architecture = [w.shape[1] for w in weights if w.ndim == 2][:3] 
        if not architecture :
             logger.critical(f"Could not determine GCN architecture from weights. Ensure weights file is correct.")
             return None
        logger.debug(f"Inferred GCN architecture (channels): {architecture}")
        model = GCNModel(architecture)
        num_node_features = weights[0].shape[0]
        model.build([(None, num_node_features), (None, None)]) 
        model.set_weights(weights)
        logger.info("GCN Model built and weights set successfully.")
        return model
    except Exception as e:
        logger.critical(f"Critical error building GCN model or setting weights: {e}", exc_info=True)
        return None

def process_molecule(smiles: str) -> Optional[Chem.Mol]:
    logger.debug(f"Processing SMILES: {smiles}")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"RDKit could not parse SMILES: '{smiles}'.")
            return None
        mol = AllChem.AddHs(mol)
        embed_params = AllChem.ETKDGv3()
        embed_params.useRandomCoords = False
        embed_success = AllChem.EmbedMolecule(mol, embed_params)
        if embed_success == -1:
            logger.warning(f"Initial embedding failed for '{smiles}'. Trying with random coordinates.")
            embed_params.useRandomCoords = True
            embed_success = AllChem.EmbedMolecule(mol, embed_params)
            if embed_success == -1:
                logger.error(f"Failed to generate 3D conformation for '{smiles}'.")
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e: 
            logger.warning(f"Could not optimize molecule '{smiles}' using MMFF94: {e}.")
        return mol
    except Exception as e: 
        logger.error(f"Unexpected error in process_molecule for SMILES '{smiles}': {e}", exc_info=True)
        return None

def create_graph_features(mol: Chem.Mol, smiles_for_logging: str = "N/A") -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    logger.debug(f"Creating graph features for SMILES: {smiles_for_logging}")
    try:
        prop = AllChem.MMFFGetMoleculeProperties(mol)
        if prop is None: 
            logger.error(f"MMFFGetMoleculeProperties returned None for SMILES: {smiles_for_logging}.")
            return None
        atom_types = []
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            try:
                mmff_atom_type = prop.GetMMFFAtomType(atom_idx)
                atom_types.append(mmff_atom_type)
            except Exception as e: 
                logger.error(f"Could not get MMFF atom type for atom {atom_idx} (SMILES: {smiles_for_logging}): {e}.")
                return None
        x = np.zeros((len(atom_types), len(UNIQUE_ATOM_TYPES)), dtype=np.float32)
        for i, a_type in enumerate(atom_types):
            if a_type not in UNIQUE_ATOM_TYPES:
                logger.error(f"Unsupported MMFF atom type: {a_type} (SMILES: {smiles_for_logging}).")
                return None
            x[i, UNIQUE_ATOM_TYPES.index(a_type)] = 1.0
        adj = Chem.GetAdjacencyMatrix(mol).astype(np.float32)
        y = np.zeros((len(atom_types), 1), dtype=np.float32)
        return x, adj, y
    except Exception as e:
        logger.error(f"Unexpected error in create_graph_features for SMILES '{smiles_for_logging}': {e}", exc_info=True)
        return None

def predict_sigma_profile(model: GCNModel, graph: Graph, smiles_for_logging: str = "N/A") -> Optional[np.ndarray]:
    logger.debug(f"Predicting sigma profile for SMILES: {smiles_for_logging}")
    try:
        dataset = MoleculeDataset(graph)
        dataset.apply(transforms.GCNFilter())
        loader = BatchLoader(dataset, batch_size=1, shuffle=False)
        prediction = model.predict(loader.load(), steps=loader.steps_per_epoch)
        return prediction
    except Exception as e:
        logger.error(f"Error during sigma profile prediction for SMILES '{smiles_for_logging}': {e}", exc_info=True)
        return None

def save_sigma_profile(prediction: np.ndarray, identifier: str, smiles_for_logging: str = "N/A") -> None:
    logger.debug(f"Saving sigma profile for ID '{identifier}' (SMILES: {smiles_for_logging})")
    try:
        sigma_values = np.linspace(*SIGMA_RANGE)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, f"{identifier}.txt")
        np.savetxt(filename, np.column_stack([sigma_values, prediction.flatten()]), delimiter='\t', fmt='%.6f')
        logger.info(f"Saved sigma profile: {filename}")
    except Exception as e:
        logger.error(f"Error saving sigma profile for ID '{identifier}' (SMILES: {smiles_for_logging}'): {e}", exc_info=True)

# --- Main Execution ---
def main():
    logger.info("--- Main function started ---")
    
    model = load_gcn_model()
    if model is None: return

    logger.info(f"Reading Excel file: {EXCEL_PATH}")
    if not os.path.exists(EXCEL_PATH):
        logger.critical(f"Excel file not found: {EXCEL_PATH}.")
        return
    try:
        data = pd.read_excel(EXCEL_PATH, sheet_name=0, engine='openpyxl') 
        logger.info(f"Loaded {len(data)} rows from Excel.")
    except Exception as e:
        logger.critical(f"Critical error reading Excel: {EXCEL_PATH}: {e}", exc_info=True)
        return

    if ID_COLUMN_NAME not in data.columns:
        logger.critical(f"ID column '{ID_COLUMN_NAME}' not found. Available: {data.columns.tolist()}.")
        return
    logger.info(f"Using ID column: '{ID_COLUMN_NAME}'.")

    smiles_col_to_use = DEFAULT_SMILES_COLUMN_NAME
    possible_smiles_cols = [DEFAULT_SMILES_COLUMN_NAME, 'Smiles', 'smiles', 'canonical_smiles', 'Canonical SMILES']
    if smiles_col_to_use not in data.columns:
        logger.warning(f"Default SMILES column '{smiles_col_to_use}' not found. Trying alternatives: {possible_smiles_cols[1:]}")
        found_alt_smiles = False
        for col in possible_smiles_cols[1:]:
            if col in data.columns:
                smiles_col_to_use = col
                logger.info(f"Using alternative SMILES column: '{smiles_col_to_use}'.")
                found_alt_smiles = True
                break
        if not found_alt_smiles:
            logger.critical(f"No SMILES column found. Tried: {possible_smiles_cols}. Available: {data.columns.tolist()}.")
            return
    else:
        logger.info(f"Using SMILES column: '{smiles_col_to_use}'.")

    logger.info(f"Processing {len(data)} molecules...")
    processed_count = 0
    error_count = 0

    for index, row in data.iterrows():
        current_molecule_id_raw = row.get(ID_COLUMN_NAME, f"MISSING_ID_ROW_{index}")
        current_smiles = row.get(smiles_col_to_use, "")
        current_molecule_id_str = str(current_molecule_id_raw) if pd.notna(current_molecule_id_raw) and current_molecule_id_raw != "" else ""
        safe_filename_id = "".join(c for c in current_molecule_id_str if c.isalnum() or c in ['.', '_', '-'])
        if not safe_filename_id:
            safe_filename_id = f"unidentified_molecule_row_{index}"
            if current_molecule_id_str:
                 logger.warning(f"Original ID '{current_molecule_id_str}' (row {index}) sanitized to empty. Using filename '{safe_filename_id}'.")
            else:
                 logger.error(f"Missing/empty ID for row {index}. Using filename '{safe_filename_id}'.")
        
        logger.info(f"Processing {index + 1}/{len(data)}: Orig_ID='{current_molecule_id_raw}', File_ID='{safe_filename_id}', SMILES='{current_smiles}'")

        if not current_smiles or pd.isna(current_smiles):
            logger.error(f"Skipping: Empty/invalid SMILES for ID '{current_molecule_id_raw}' (row {index}).")
            error_count += 1
            continue
        
        try:
            mol = process_molecule(current_smiles)
            if mol is None: error_count += 1; continue
            graph_features_tuple = create_graph_features(mol, smiles_for_logging=current_smiles)
            if graph_features_tuple is None: error_count += 1; continue
            x_nodes, adj_matrix, y_dummy = graph_features_tuple
            graph = Graph(x=x_nodes, a=adj_matrix, y=y_dummy)
            prediction_array = predict_sigma_profile(model, graph, smiles_for_logging=current_smiles)
            if prediction_array is None: error_count += 1; continue
            save_sigma_profile(prediction_array.flatten(), safe_filename_id, smiles_for_logging=current_smiles) 
            processed_count += 1
        except Exception as e: 
            error_count += 1
            logger.critical(f"CRITICAL UNHANDLED EXCEPTION: Orig_ID='{current_molecule_id_raw}', SMILES='{current_smiles}', Row='{index}', File_ID='{safe_filename_id}'. Error: {str(e)}", exc_info=True)
        
    logger.info(f"--- Processing Summary ---")
    logger.info(f"Total molecules: {len(data)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped/failed: {error_count}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output in: {OUTPUT_DIR}")
    logger.info("--- End of Script ---")

if __name__ == "__main__":
    # The very first action should be initializing the logger if not done globally already
    # However, in this script, logger is global and initialized at the top.
    # For robustness, one might add a try-except around main() to log fatal errors
    # that occur outside the main loop's try-except.
    try:
        main()
    except Exception as e:
        logger.critical(f"A CRITICAL error occurred outside the main processing loop: {e}", exc_info=True)
        # Optional: print to stdout as well if logging to file might be the issue
        print(f"A CRITICAL error occurred outside the main processing loop: {e}")
    finally:
        logging.shutdown() # Ensure all handlers are closed properly
        print("Script execution finished. Check error_log.txt for details.")
