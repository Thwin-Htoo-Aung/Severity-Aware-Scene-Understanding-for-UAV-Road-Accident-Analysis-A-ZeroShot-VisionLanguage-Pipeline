import copy
from datetime import datetime
import os
import click
import yaml


from vista.evaluate import run
from vista.stats import analyze_yolo_dataset, ALL_SECTIONS
from vista.utils.logger import get_logger
from vista.utils.utils import load_yaml
from vista.utils.grid import create_experiment
from vista.utils.run import ParallelRun


OUT_FOLDER = "out"

FUNCTIONS = {
    "evaluate": run,
    "train": run,  # for now, train and evaluate are the same function, but they could be different in the future
    "run": run,  # alias for evaluate
}
FUNCTION_SLURMS = {
    "evaluate": "hpc/launch_run",
    "computational": "hpc/launch_computational",
}
FUNCTIONS_CONDOR = {
    "evaluate": "hpc/condor",
    "computational": "hpc/condor",
}
SCHEDULERS = {
    "slurm": FUNCTION_SLURMS,
    "condor": FUNCTIONS_CONDOR,
}


@click.group()
def cli():
    """Run a refinement or a grid"""
    pass


def manage_multiprocess_run(run_parameters, run_name, logger, job_parallelism=None):
    """
    Manage the multiprocess run.
    This function is used to launch the run in parallel or sequentially.
    """
    if "dataloader" in run_parameters and "num_processes" in run_parameters["dataloader"]:
        multi_runs = [
                copy.deepcopy(run_parameters) for _ in range(run_parameters["dataloader"]["num_processes"])
            ]
        if job_parallelism is not None and job_parallelism > 1:
            run_names = [
                f"{run_name}/job_{str(i//job_parallelism).zfill(3)}/p_{str(i%job_parallelism).zfill(3)}" for i in range(run_parameters["dataloader"]["num_processes"])
            ]
        else:    
            run_names = [
                f"{run_name}/p_{str(i).zfill(3)}" for i in range(run_parameters["dataloader"]["num_processes"])
            ]
        
        logger.info(f"Running {len(multi_runs)} processes in parallel")
        for i, run_parameters in enumerate(multi_runs):
            run_name = f"{run_names[i]}"
            multi_runs[i]["dataloader"]["process_id"] = i
            os.makedirs(run_name, exist_ok=True)
    else:
        multi_runs = [run_parameters]
        run_names = [run_name]
        logger.info("Running in single process mode")

    return multi_runs, run_names


@cli.command("grid")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a grid search",
)
@click.option(
    "--parallel",
    default=False,
    is_flag=True,
    help="Run the grid in parallel",
)
@click.option(
    "--only_create",
    default=False,
    is_flag=True,
    help="Only create the slurm scripts",
)
@click.option(
    "--function",
    default="evaluate",
    help="Name of the function to run, either 'evaluate' only for now",
)
@click.option(
    "--job_parallelism",
    default=None,
    type=int,
    help="If --parallel is provided, and dataloader.num_processes is provided in the parameters, this will set the number of processes to use in a single job, should be divisor of dataloader.num_processes",
)
@click.option(
    "--scheduler",
    default="slurm",
    help="Scheduler to use, either 'slurm' or 'condor'",
)
@click.option(
    "--out_folder",
    default=OUT_FOLDER,
    help="Folder where the grid results will be saved",
)
def grid(parameters, parallel, only_create=False, function="evaluate", job_parallelism=None, scheduler="slurm", out_folder=OUT_FOLDER):
    assert function in FUNCTIONS, f"Function {function} not recognized, available functions: {list(FUNCTIONS.keys())}"
    
    run_function = FUNCTIONS[function]
    slurm_script = SCHEDULERS[scheduler][function]
    
    assert os.path.exists(slurm_script), f"Slurm script {slurm_script} does not exist"
    
    parameters = load_yaml(parameters)
    grid_name = parameters["grid"]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    grid_name = f"{current_time}_{grid_name}"
    log_folder = os.path.join(out_folder, grid_name)
    
    runs_parameters = create_experiment(parameters)
    
    os.makedirs(log_folder)
    with open(os.path.join(log_folder, "hyperparams.yaml"), "w") as f:
        yaml.dump(parameters, f)

    grid_logger = get_logger("Grid", f"{log_folder}/grid.log")
    grid_logger.info(f"Running {len(runs_parameters)} runs")
    for i, run_parameters in enumerate(runs_parameters):
        run_name = f"{log_folder}/run_{i}"
        os.makedirs(run_name, exist_ok=True)
        multi_runs, run_names = manage_multiprocess_run(
            run_parameters, run_name, grid_logger, job_parallelism
        )
        if parallel:
            job_parallelism = job_parallelism if job_parallelism is not None else 1
            jobs = len(multi_runs) // job_parallelism
            for j in range(jobs):
                subrun_parameters = multi_runs[j*(job_parallelism):(j+1)*(job_parallelism)]
                subrun_name = run_names[j*(job_parallelism):(j+1)*(job_parallelism)]
                if len(subrun_parameters) == 1:
                    subrun_name = subrun_name[0]
                    subrun_parameters = subrun_parameters[0]
                run = ParallelRun(
                    subrun_parameters,
                    multi_gpu=False,
                    logger=grid_logger,
                    run_name=subrun_name,
                    slurm_script=slurm_script,
                    scheduler=scheduler,
                )
                run.launch(
                    only_create=only_create,
                    script_args=[
                        "--disable_log_params",
                        "--disable_log_on_file",
                        "--function",
                        function,
                    ],
                )
        else:
            for k, (subrun_parameters, subrun_name) in enumerate(
                zip(multi_runs, run_names)
            ):
                if len(multi_runs) > 1:
                    grid_logger.info(
                        f"Running subrun {k+1}/{len(multi_runs)} in run {i+1}/{len(runs_parameters)}"
                    )
                else:
                    grid_logger.info(f"Running run {i+1}/{len(runs_parameters)}")
                run_function(subrun_parameters, run_name=subrun_name)


@cli.command("run")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
@click.option("--run_name", default=None, help="Name of the run")
@click.option(
    "--disable_log_params",
    default=False,
    is_flag=True,
    help="Disable Log the parameters",
)
@click.option(
    "--disable_log_on_file", default=False, is_flag=True, help="Disable Log on file"
)
@click.option(
    "--run_name",
    default=None,
    help="Name of the run, if not provided, it will be generated based on the current time",
)
@click.option(
    "--function",
    default="evaluate",
    help="Name of the function to run, either 'evaluate' or 'computational'",
)
def run(
    parameters,
    run_name=None,
    disable_log_params=False,
    disable_log_on_file=False,
    function="evaluate",
):
    assert function in FUNCTIONS, f"Function {function} not recognized, available functions: {list(FUNCTIONS.keys())}"
    run_function = FUNCTIONS[function]
    
    parameters = load_yaml(parameters)
    run_function(
        parameters,
        run_name,
        not disable_log_params,
        not disable_log_on_file,
    )

@cli.command("stats")
@click.argument("yaml_path")
@click.option("--output_dir", "-o", default="dataset_stats", show_default=True,
              help="Directory where plots and reports are saved.")
@click.option("--splits", default="train,val,test", show_default=True,
              help="Comma-separated list of splits to analyse.")
@click.option("--sections", default=None,
              help=(
                  "Comma-separated list of sections to run.  Omit to run all.  "
                  f"Available: {','.join(ALL_SECTIONS)}"
              ))
@click.option("--read_image_sizes", is_flag=True, default=False,
              help="Open every image to read pixel dimensions (slow; requires Pillow).")
@click.option("--no_json", is_flag=True, default=False, help="Skip saving stats.json.")
@click.option("--no_csv",  is_flag=True, default=False, help="Skip saving annotations.csv.")
@click.option("--figsize", default="10,6", show_default=True,
              help="Figure size as 'width,height' in inches.")
@click.option("--dpi", default=150, show_default=True, help="DPI for SVG rasterisation.")
@click.option("--style", default="seaborn-v0_8-whitegrid", show_default=True,
              help="Matplotlib style name.")
@click.option("--heatmap_bins", default=64, show_default=True,
              help="Grid resolution for spatial heatmaps.")
@click.option("--heatmap_cmap", default="hot", show_default=True,
              help="Matplotlib colourmap for heatmaps.")
@click.option("--hist_bins", default=40, show_default=True,
              help="Number of bins for histogram plots.")
@click.option("--bar_orientation", default="horizontal", show_default=True,
              type=click.Choice(["horizontal", "vertical"]),
              help="Orientation for bar charts.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress progress output.")
def stats(
    yaml_path,
    output_dir,
    splits,
    sections,
    read_image_sizes,
    no_json,
    no_csv,
    figsize,
    dpi,
    style,
    heatmap_bins,
    heatmap_cmap,
    hist_bins,
    bar_orientation,
    quiet,
):
    """Compute and save dataset statistics for a YOLO-format dataset YAML."""
    splits_list   = [s.strip() for s in splits.split(",") if s.strip()]
    sections_list = [s.strip() for s in sections.split(",") if s.strip()] if sections else None
    fw, fh = (float(v) for v in figsize.split(","))

    analyze_yolo_dataset(
        yaml_path=yaml_path,
        output_dir=output_dir,
        splits=splits_list,
        sections=sections_list,
        read_image_sizes=read_image_sizes,
        save_json=not no_json,
        save_csv=not no_csv,
        verbose=not quiet,
        figsize=(fw, fh),
        dpi=dpi,
        style=style,
        heatmap_bins=heatmap_bins,
        heatmap_cmap=heatmap_cmap,
        hist_bins=hist_bins,
        bar_orientation=bar_orientation,
    )


if __name__ == "__main__":
    cli()
