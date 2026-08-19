import spaces  # must be the very first import so ZeroGPU can patch CUDA init in torch

import gradio as gr
import pandas as pd
import mols2grid
import uuid
from pathlib import Path
import html as html_lib
from gen_mols import mols_gen

df_beads = pd.read_csv('dim_by_dg.csv')
VALUE_OPTIONS = {j: df_beads['dg'][i] for i, j in enumerate(df_beads['bead'].tolist())}

MIN_ATOMS = 5
MAX_ATOMS = 9
ATOM_CHOICES = [str(a) for a in range(MIN_ATOMS, MAX_ATOMS + 1)]  # 5, 6, 7, 8, 9
SENTINEL = "Select an option"
LEAVE_BLANK = "Distribution"
ATOM_MODE_CHOICES = [SENTINEL] + ATOM_CHOICES + [LEAVE_BLANK]

ATOM_INFO = (
    "NOTE: The molecules to be generated need to be between 5 to 9 heavy atoms. "
    "If Distribution is selected, the number of atoms would be derived from the "
    "probability on the training dataset."
)

PROP_INFO = (
    "Basic properties of the generated molecules will be computed with RDKit.\n\n"
    "Properties are:\n"
    "- Number of heavy atoms\n"
    "- Molecular weight\n"
    "- Octanol/Water partition coefficient (logP)\n"
    "- Topological Polar Surface Area (TPSA)\n"
    "- Octanol/Water free energy\n\n"
    "Optional: Solvent Accessible Surface Area"
)


def new_session_dir() -> str:
    session_dir = Path('sessions') / uuid.uuid4().hex
    session_dir.mkdir(parents=True, exist_ok=True)
    return str(session_dir)


def resolve_value_and_key(value_mode, chosen_key, manual_value):
    if value_mode == "Choose from list":
        return chosen_key, VALUE_OPTIONS[chosen_key]
    return "no_bead", manual_value


def resolve_atoms(atom_mode):
    if atom_mode in (SENTINEL, LEAVE_BLANK):
        return None
    return int(atom_mode)


def estimate_duration(value_mode, chosen_key, manual_value, atom_mode,
                       n_molecules, cond_w, calc_prop, sasa_calc, session_dir):
    """ZeroGPU needs an estimate (seconds) of how long the GPU will be held.
    Sampling cost scales roughly with how many molecules are requested."""
    n_mols = int(n_molecules) if n_molecules else 100
    return min(300, 30 + n_mols)


def render_grid(result_df):
    # render to a raw HTML document (outside Jupyter this returns HTML, not a widget)
    grid_doc = mols2grid.MolGrid(result_df, smiles_col="smiles").render(template="interactive")
    # embed in our own iframe so its scripts execute in their own browsing context
    srcdoc = html_lib.escape(grid_doc)   # escapes quotes too — valid for the srcdoc attribute
    return (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%; height:600px; border:none;"></iframe>'
    )

@spaces.GPU(duration=estimate_duration)
def run_generation(value_mode, chosen_key, manual_value, atom_mode,
                    n_molecules, cond_w, calc_prop, sasa_calc, session_dir):
    key, guidance = resolve_value_and_key(value_mode, chosen_key, manual_value)
    n_atoms = resolve_atoms(atom_mode)

    df, file_mols_gen = mols_gen(
        key, guidance, cond_w, int(n_molecules), n_atoms, session_dir,
        calculate_prop=calc_prop, sasa_calc=sasa_calc,
    )

    grid_html = render_grid(df)

    csv_path = str(Path(session_dir) / f"juniper_mols_s{cond_w}_{key}_nmols{int(n_molecules)}.csv")
    df.to_csv(csv_path, index=False)

    return df, grid_html, csv_path


with gr.Blocks(title="Juniper DG") as demo:
    session_dir_state = gr.State()

    gr.Markdown("# Molecule generator of Juniper DG")

    gr.Markdown("## Bead id")
    value_mode = gr.Radio(
        ["Choose from list", "Enter manually"],
        value="Choose from list",
        label="Do you want to select a bead combination or enter a value of DG?",
    )
    chosen_key = gr.Dropdown(list(VALUE_OPTIONS.keys()), value=list(VALUE_OPTIONS.keys())[0],
                              label="Select a value", visible=True)
    manual_value = gr.Number(value=1.0, step=0.1, label="Enter a value", visible=False)
    current_value_caption = gr.Markdown(f"Current value: {list(VALUE_OPTIONS.values())[0]}")

    gr.Markdown("## Number of atoms")
    atom_mode = gr.Dropdown(ATOM_MODE_CHOICES, value=SENTINEL, label="Number of atoms")
    atom_info = gr.Markdown(ATOM_INFO)

    gr.Markdown("## Number of molecules")
    n_molecules = gr.Number(value=10, minimum=0, step=1,
                             label="How many molecules would you like to generate?")

    gr.Markdown("## Conditional weight")
    cond_w = gr.Number(
        value=1.0, minimum=0.1, step=0.1,
        label=("Condition value between conditional and unconditional probabilities. "
               "If 1, it samples the conditional distribution"),
    )

    gr.Markdown("## Calculation of properties with RDKit")
    gr.Markdown(PROP_INFO)
    calc_prop = gr.Checkbox(label="Property calculation", value=False)
    sasa_calc = gr.Checkbox(label="SASA", value=False, visible=False)

    gr.Markdown("## Generate")
    generate_btn = gr.Button("Generate", variant="primary", interactive=False)
    status = gr.Markdown()
    result_table = gr.Dataframe(label="Results", visible=False)
    result_grid = gr.HTML(visible=False)
    download_file = gr.File(label="Download result as CSV", visible=False)

    def toggle_value_mode(mode):
        return gr.update(visible=mode == "Choose from list"), gr.update(visible=mode == "Enter manually")

    value_mode.change(toggle_value_mode, inputs=value_mode, outputs=[chosen_key, manual_value])

    def show_current_value(mode, key, manual):
        value = VALUE_OPTIONS[key] if mode == "Choose from list" else manual
        return f"Current value: {value}"

    for component in (value_mode, chosen_key, manual_value):
        component.change(show_current_value, inputs=[value_mode, chosen_key, manual_value],
                          outputs=current_value_caption)

    def toggle_atom_mode(mode):
        return gr.update(interactive=mode != SENTINEL)

    atom_mode.change(toggle_atom_mode, inputs=atom_mode, outputs=generate_btn)

    def toggle_sasa(checked):
        return gr.update(visible=checked, value=checked and None or False)

    calc_prop.change(toggle_sasa, inputs=calc_prop, outputs=sasa_calc)

    def on_generate(value_mode_v, chosen_key_v, manual_value_v, atom_mode_v,
                     n_molecules_v, cond_w_v, calc_prop_v, sasa_calc_v, session_dir_v):
        if sasa_calc_v and not calc_prop_v:
            raise gr.Error("You need to enable 'Property calculation' to compute SASA.")
        if session_dir_v is None:
            session_dir_v = new_session_dir()

        df, grid_html, csv_path = run_generation(
            value_mode_v, chosen_key_v, manual_value_v, atom_mode_v,
            n_molecules_v, cond_w_v, calc_prop_v, sasa_calc_v, session_dir_v,
        )
        return (
            "Done.",
            gr.update(value=df, visible=True),
            gr.update(value=grid_html, visible=True),
            gr.update(value=csv_path, visible=True),
            session_dir_v,
        )

    generate_btn.click(
        on_generate,
        inputs=[value_mode, chosen_key, manual_value, atom_mode,
                n_molecules, cond_w, calc_prop, sasa_calc, session_dir_state],
        outputs=[status, result_table, result_grid, download_file, session_dir_state],
    )

if __name__ == "__main__":
    demo.launch()
