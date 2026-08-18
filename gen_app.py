import streamlit as st
import pandas as pd
import mols2grid
import uuid
from pathlib import Path
from gen_mols import mols_gen

# Runs once per session; survives reruns
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

# Create a per-session folder
session_dir = Path('sessions') / st.session_state.session_id
session_dir.mkdir(parents=True, exist_ok=True)


#from gen_mols import get_mols

df_beads = pd.read_csv('dim_by_dg.csv')

beads = {}
for i,j in enumerate(df_beads['bead'].tolist()):
    beads[j] = df_beads['dg'][i]

#values_x = df_beads.to_dict()

VALUE_OPTIONS = beads

MIN_ATOMS = 5
MAX_ATOMS = 9
ATOM_CHOICES = list(range(MIN_ATOMS, MAX_ATOMS + 1))  # 5, 6, 7, 8, 9

def run_generation(key,guidance,value_s,n_mols,n_atoms,session_dir,calculate_prop=False,sasa_calc=False):
    df = mols_gen(key,guidance,value_s,n_mols,n_atoms,session_dir,calculate_prop,sasa_calc)
    return df


st.set_page_config(page_title="Juniper DG", layout="centered")
st.title("Molecule generator of Juniper DG")  # PLACEHOLDER title


# 1. Value: pick from the dictionary or enter a float manually
st.header("Bead id")
value_mode = st.radio(
    "Do you want to select a bead combination or enter a value of DG?",
    ["Choose from list", "Enter manually"],
    horizontal=True,
)

if value_mode == "Choose from list":
    chosen_key = st.selectbox("Select a value", list(VALUE_OPTIONS.keys()))
    value = VALUE_OPTIONS[chosen_key]
else:
    value = st.number_input("Enter a value", value=1.0, step=0.1, format="%.4f")
    chosen_key = 'no_bead'

st.caption(f"Current value: {value}")


# 2. Number of atoms: pick a preset, enter manually (5 to 9), or nothing yet
st.header("Number of atoms")
SENTINEL = "Select an option"
LEAVE_BLANK = "Leave blank"
atom_mode = st.selectbox(
    "Number of atoms",
    [SENTINEL] + [str(a) for a in ATOM_CHOICES] + ["Enter manually", LEAVE_BLANK],
)
 
n_atoms = None
atoms_ready = False
if atom_mode == SENTINEL:
    # PLACEHOLDER message: write here whatever you want to prompt yourself with.
    st.info("NOTE: The molecules to be generated need to be between 5 to 9 heavy atoms."
            " If it is not selected, the number of atoms would be derived from the "   
            "probability on the training dataset.")
elif atom_mode == LEAVE_BLANK:
    # Explicitly blank: n_atoms stays None and None is passed to your code.
    atoms_ready = True
    st.caption("Number of atoms: blank (None will be passed)")
elif atom_mode == "Enter manually":
    n_atoms = st.number_input(
        "Enter number of atoms",
        min_value=MIN_ATOMS,
        max_value=MAX_ATOMS,
        value=MIN_ATOMS,
        step=1,
    )
    atoms_ready = True
else:
    n_atoms = int(atom_mode)
    atoms_ready = True

# 3. Number of molecules to generate
st.header("Number of molecules")
n_molecules = st.number_input(
    "How many molecules would you like to generate?",
    min_value=0,
    value=10,
    step=1,
)

# 4. Conditional weight
st.header("Conditional weight")
cond_w = st.number_input(
    "Condition value between conditional and unconditional probabilities. If 1, it samples the conditional distribution",
    min_value=0.1,
    value=1.0,
    step=0.1,
)

# 5. Property calc
st.header("Calculation of properties with RdKit")
st.info(
    "Basic properties of the generated molecules will be computed with RDKit.\n\n"
    "Properties are:\n"
    "- Number of heavy atoms\n"
    "- Molecular weight\n"
    "- Octanol/Water partition coefficient (logP)\n"
    "- Topological Polar Surface Area (TPSA)\n"
    "- Octanol/Water free energy\n\n"
    "Optional: Solvent Accessible Surface Area"
)
prop_calc = st.toggle("Property calculation")
if prop_calc:
   sasa_calc = st.checkbox("SASA")
else:
   sasa_calc = False
# 4 and 5. Run the other code, read the result, offer a CSV download
st.header("Generate")

can_run = atoms_ready
if st.button("Generate", disabled=not can_run):
   with st.spinner("Running..."):
        result_df,file_name = run_generation(chosen_key,value,cond_w,int(n_molecules),n_atoms,session_dir,
                                   prop_calc,sasa_calc)
        st.session_state["result_df"] = result_df
        st.session_state["result_name"] = f"{file_name}"

if "result_df" in st.session_state:
     st.success("Done.")
     result_df = st.session_state["result_df"]
     # Generate mols2grid HTML
     raw_html = mols2grid.display(result_df, smiles_col="smiles")

     # Display in Streamlit
     st.components.v1.html(raw_html._repr_html_(), height=600, scrolling=True)
     st.dataframe(result_df, width='stretch')
     csv_bytes = result_df.to_csv(index=False).encode("utf-8")
     st.download_button(
        "Download result as CSV",
         data=csv_bytes,
         file_name=st.session_state.get("result_name", f"juniper_mols_s{cond_w}_{chosen_key}_nmols{n_molecules}.csv"),
         mime="text/csv",
     )