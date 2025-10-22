# Copyright (c) Meta Platforms, Inc. and affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import itertools
import json
import os
import time

import numpy as np

np.random.seed(42)  # For deterministic sampling

from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
sns.set_style("white")


# Apps
SUPPORTED_APPS = {
    "Calendar",
    "Messaging",
    "Search",
    "Videos",
    "Notes",
    "Maps",
    "Music",
}

# Input data files
INPUT_PATHS = {
    "generative": "ob2/structured_goals.json",
    "digital": "ob2/digital_state_v2.4.4.json",
}

# Model classes
SMALL_MODELS = ["internvl_2b", "qwen_3b"]
MEDIUM_MODELS = ["llama3v", "internvl_8b", "qwen_7b"]
LARGE_MODELS = ["llama4v", "internvl_78b", "qwen_72b"]
MODELS = SMALL_MODELS + MEDIUM_MODELS + LARGE_MODELS

# Modalities and subsets
INPUT_MODALITIES = ["V", "VA", "VD", "VL", "VADL", "VD*", "VL*"]
SUBSETS = ["V", "VA", "VD", "VL"]
FULL_SUBSETS = SUBSETS + ["Full"]

# Raw result paths
RAW_RESULT_PATHS = {}
for model in MODELS:
    RAW_RESULT_PATHS[model] = {}
    for input_modality in INPUT_MODALITIES:
        RAW_RESULT_PATHS[model][input_modality] = (
            f"assets/raw_predictions/generative_{model}_{input_modality}.json".replace(
                "*", "star"
            )
        )

# Plotting resources
PLOT_SUBSETS = {
    "VA": "$S_{VA}$",
    "VD": "$S_{VD}$",
    "VL": "$S_{VL}$",
}
PLOT_INPUT_MODALITIES = [
    "V",
    "VA",
    "VD",
    "VD*",
    "VL",
    "VL*",
    "VADL",
]  # Re-ordered for plotting
MODEL_CLASSES = {"small": "Small", "medium": "Medium", "large": "Large"}


# Function to create subplots for each data subset
def plot_subplots(data, savedir=None, font_size=30):
    # Create a figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 6), sharey=True)
    fig.tight_layout()

    # Iterate over each data subset and corresponding subplot axis
    for plot_subset, ax in zip(PLOT_SUBSETS, axes):
        # Generate a dataframe for the current plot_subset
        dict_extracted = {
            input_modality: data[input_modality][plot_subset]
            for input_modality in PLOT_INPUT_MODALITIES
        }
        list_filtered = [
            ["Input Modality", "Model Class", "Accuracy", "Error_low", "Error_high"]
        ]
        supported_input_modalities = []
        for input_modality in dict_extracted:
            for model_class in dict_extracted[input_modality]:
                accuracy = dict_extracted[input_modality][model_class]["mean"]
                if (input_modality == "V") or (
                    len(set(plot_subset) - set(input_modality)) == 0
                ):
                    if input_modality not in supported_input_modalities:
                        supported_input_modalities.append(input_modality)
                    list_filtered.append(
                        [
                            input_modality,
                            MODEL_CLASSES[model_class],
                            accuracy,
                            accuracy
                            - dict_extracted[input_modality][model_class]["ci_lower"],
                            dict_extracted[input_modality][model_class]["ci_upper"]
                            - accuracy,
                        ]
                    )
        df_filtered = pd.DataFrame(list_filtered[1:], columns=list_filtered[0])

        # Create the bar plot on the current axis
        bars = sns.barplot(
            data=df_filtered,
            x="Model Class",
            y="Accuracy",
            hue="Input Modality",
            errorbar=None,
            ax=ax,
            palette=[(0.09411764705882353, 0.4666666666666667, 0.9490196078431372)]
            * len(supported_input_modalities),  # Set all bars to same color
        )
        ax.set_ylim(0, 0.6)
        ax.tick_params(axis="x", labelsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size)

        # Error bars
        x_positions = [
            (
                i
                + (j - (len(supported_input_modalities) - 1) / 2)
                * 1.0
                / (len(supported_input_modalities) + 1)
            )
            for j in range(len(supported_input_modalities))
            for i in range(len(MODEL_CLASSES))
        ]
        y_positions = [
            df_filtered[
                (df_filtered["Model Class"] == model_class)
                & (df_filtered["Input Modality"] == input_modality)
            ]["Accuracy"].iloc[0]
            for input_modality in supported_input_modalities
            for model_class in MODEL_CLASSES.values()
        ]
        ax.errorbar(
            x=x_positions,
            y=y_positions,
            yerr=[df_filtered["Error_low"], df_filtered["Error_high"]],
            fmt="none",
            c="black",
            capsize=2,
            elinewidth=1,
        )

        # Add hatches and colors to bars
        colors = sns.color_palette("crest")
        for p, (input_modality, model_class) in zip(
            bars.patches,
            itertools.product(supported_input_modalities, MODEL_CLASSES.keys()),
        ):
            if set(plot_subset).difference(set(input_modality)) == set():
                p.set_hatch("//")
            if model_class == "small":
                p.set_facecolor(colors[0])
            elif model_class == "medium":
                p.set_facecolor(colors[2])
            else:
                p.set_facecolor(colors[5])

        # Annotate bars with input modality labels
        for p, label in zip(
            bars.patches,
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(item, len(MODEL_CLASSES))
                    for item in supported_input_modalities
                )
            ),
        ):
            ax.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + 0.03,
                label,
                fontsize=font_size - 10,
                ha="center",
                va="bottom",
            )

        # Set title and labels for each subplot
        ax.set_title(
            f"Subset = {PLOT_SUBSETS[plot_subset]}",
            fontsize=font_size + 4,
        )
        ax.set_xlabel(None)
        ax.set_ylabel("Generative LLM-Judge score", fontsize=font_size)

        # Add horizontal gridlines
        ax.yaxis.grid(True, linestyle="--", alpha=0.75)

        # Remove individual legends
        ax.get_legend().remove()

    # Put a joint x-label
    fig.text(0.5, -0.05, "Model Class", fontsize=font_size, ha="center")

    # Save the figure
    if savedir is not None:
        savefile = os.path.join(savedir, f"subsets.png")
        plt.savefig(
            savefile,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Plot saved to {savefile}")


def bootstrap_ci(
    data, num_bootstraps: int = 1000, ci: int = 95
) -> Tuple[float, float, float]:
    """
    Calculate a bootstrapped confidence interval around the mean.

    Args:
        data (array-like): The input data.
        num_bootstraps (int): The number of bootstrap samples to generate (Default: 1000).
        ci (float): The desired confidence level (Default: 95).

    Returns:
        tuple: A tuple with the mean, the lower and upper bounds of the confidence interval.
    """
    # Convert data to a NumPy array
    data = np.array(data)
    # Calculate the mean of the original data
    mean = np.mean(data)
    # Initialize an array to store the means of the bootstrap samples
    bootstrap_means = np.zeros(num_bootstraps)
    # Generate bootstrap samples and calculate their means
    for i in range(num_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    # Calculate the confidence interval
    ci_lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    ci_upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return mean, ci_lower, ci_upper


def ukey(row: Dict[str, Any]) -> str:
    """Get the unique key for each row"""
    return row["meta"]["client_tag"] + "_" + row["meta"]["scenario"]["shortname"]


def form_subsets(
    input_list: List[Dict[str, Any]],
    longitudinal_input: Dict[str, Any],
    digital_input: Dict[str, Any],
) -> Dict[str, Any]:
    # Form modality subsets
    subsets = {subset: set() for subset in FULL_SUBSETS}
    seen_keys = set()
    for row in input_list:
        unique_key = ukey(row)

        # Check for existing
        if unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)

        # Add to full subset
        subsets["Full"].add(unique_key)

        # Add to other appropriate subset
        cue_modalities = []
        digital = False
        longitudinal = False
        ## Check for digital cues with supported apps
        for cid, cue in enumerate(row["x"]["cues"]):
            cue_modalities += cue["modality"]
            if "digital" in cue["modality"]:
                digital_cue = digital_input[unique_key]["x"]["cues"][cid]
                assert digital_cue["modality"] == cue["modality"]
                if "app" in digital_cue and digital_cue["app"] in SUPPORTED_APPS:
                    digital = True
        ## Check for longitudinal cues
        if unique_key in longitudinal_input:
            if "longitudinal_history" in longitudinal_input[unique_key]:
                curr_history = longitudinal_input[unique_key]["longitudinal_history"]
                support_types = [h["history_type"] for h in curr_history]
                if "setup" in support_types:
                    longitudinal = True
        cue_modalities = set(cue_modalities)
        if "" in cue_modalities:
            cue_modalities.remove("")
        if ("digital" in cue_modalities) and not digital:
            cue_modalities.remove("digital")
        if longitudinal:
            cue_modalities.add("longitudinal")
        if cue_modalities == {"vision"}:
            subsets["V"].add(unique_key)
        elif cue_modalities == {"vision", "audio"}:
            subsets["VA"].add(unique_key)
        elif cue_modalities == {"vision", "digital"}:
            subsets["VD"].add(unique_key)
        elif cue_modalities == {"vision", "longitudinal"}:
            subsets["VL"].add(unique_key)
    return subsets


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outdir",
        help="Directory to save output files (Output is not saved if not provided)",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)

    # Load input files
    t0 = time.time()
    with open(INPUT_PATHS["generative"], "r") as f:
        generative_input = longitudinal_input = json.load(f)
        # Re-index with unique key for longitudinal
        new_longitudinal_input = {ukey(row): row for row in longitudinal_input}
        assert len(longitudinal_input) == len(new_longitudinal_input)
        longitudinal_input = new_longitudinal_input
    with open(INPUT_PATHS["digital"], "r") as f:
        digital_input = json.load(f)
        # Re-index with unique key
        new_digital_input = {ukey(row): row for (_, row) in digital_input.items()}
        assert len(digital_input) == len(new_digital_input)
        digital_input = new_digital_input
    print(f"Loaded input files in {time.time() - t0:.2f} seconds")

    # Form different modality subsets
    t0 = time.time()
    gen_subsets = form_subsets(generative_input, longitudinal_input, digital_input)
    print(
        f"Formed subsets of size Full={len(gen_subsets['Full'])}, V={len(gen_subsets['V'])}, VA={len(gen_subsets['VA'])}, VD={len(gen_subsets['VD'])}, VL={len(gen_subsets['VL'])}"
    )
    print(f"Formed different modality subsets in {time.time() - t0:.2f} seconds")

    # Form subsets for results
    results = {}
    for model, paths in RAW_RESULT_PATHS.items():
        results[model] = {}
        for in_subset in INPUT_MODALITIES:
            t0 = time.time()
            # Load raw results
            try:
                with open(RAW_RESULT_PATHS[model][in_subset], "r") as f:
                    raw_results = json.load(f)
                results[model][in_subset] = {
                    out_subset: [] for out_subset in FULL_SUBSETS
                }
            except Exception as e:
                print(
                    f"ERROR: File not found for {model=}, {in_subset=}: {RAW_RESULT_PATHS[model][in_subset]}"
                )
                continue
            # Iterate over raw results
            for row in raw_results:
                unique_key = ukey(row)
                # Update full subset
                if unique_key in gen_subsets["Full"]:
                    results[model][in_subset]["Full"].append(row["metrics"]["correct"])
                # Update other appropriate subset
                found = False
                for out_subset in SUBSETS:
                    if unique_key in gen_subsets[out_subset]:
                        found = True
                        break
                if not found:
                    continue
                # Update results
                results[model][in_subset][out_subset].append(row["metrics"]["correct"])
            print(f"Processed {model=}, {in_subset=} in {time.time() - t0:.2f} seconds")
    print("Formed results buckets for different modality subsets")

    # Get results for VADL on Full subset for every model
    acc_full = {}
    in_subset = "VADL"
    out_subset = "Full"
    t0 = time.time()
    for model in MODELS:
        try:
            total = len(results[model][in_subset][out_subset])
            mean, ci_lower, ci_upper = bootstrap_ci(
                results[model][in_subset][out_subset]
            )
        except:
            total, mean, ci_lower, ci_upper = 0, 0, 0, 0
        # Store results with appropriate formatting
        acc_full[model] = {
            "total": round(total, 4),
            "mean": round(mean, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }
    print(
        f"Processed all models for {in_subset=}, {out_subset=} in {time.time() - t0:.2f} seconds"
    )

    # Get results averaged by model size
    avg_acc = {}
    for in_subset in INPUT_MODALITIES:
        avg_acc[in_subset] = {}
        for out_subset in FULL_SUBSETS:
            t0 = time.time()
            # Get accuracies for each model class
            ## Small models
            m_results = []
            for model in SMALL_MODELS:
                if in_subset in results[model]:
                    m_results += results[model][in_subset][out_subset]
            small_total = len(m_results)
            if small_total == 0:
                small_mean, small_ci_lower, small_ci_upper = 0, 0, 0
            else:
                small_mean, small_ci_lower, small_ci_upper = bootstrap_ci(m_results)

            ## Medium models
            m_results = []
            for model in MEDIUM_MODELS:
                if in_subset in results[model]:
                    m_results += results[model][in_subset][out_subset]
            medium_total = len(m_results)
            if medium_total == 0:
                medium_mean, medium_ci_lower, medium_ci_upper = 0, 0, 0
            else:
                medium_mean, medium_ci_lower, medium_ci_upper = bootstrap_ci(m_results)

            ## Large models
            m_results = []
            for model in LARGE_MODELS:
                if in_subset in results[model]:
                    m_results += results[model][in_subset][out_subset]
            large_total = len(m_results)
            if large_total == 0:
                large_mean, large_ci_lower, large_ci_upper = 0, 0, 0
            else:
                large_mean, large_ci_lower, large_ci_upper = bootstrap_ci(m_results)

            # Store results with appropriate formatting
            avg_acc[in_subset][out_subset] = {
                "small": {
                    "total": round(small_total, 4),
                    "mean": round(small_mean, 4),
                    "ci_lower": round(small_ci_lower, 4),
                    "ci_upper": round(small_ci_upper, 4),
                },
                "medium": {
                    "total": round(medium_total, 4),
                    "mean": round(medium_mean, 4),
                    "ci_lower": round(medium_ci_lower, 4),
                    "ci_upper": round(medium_ci_upper, 4),
                },
                "large": {
                    "total": round(large_total, 4),
                    "mean": round(large_mean, 4),
                    "ci_lower": round(large_ci_lower, 4),
                    "ci_upper": round(large_ci_upper, 4),
                },
            }

            print(
                f"Processed {in_subset=}, {out_subset=} in {time.time() - t0:.2f} seconds"
            )

    # Dump results to file
    if args.outdir is not None:
        ## Full results on models
        output_file = os.path.join(args.outdir, "full_evals.json")
        with open(output_file, "w") as f:
            json.dump(acc_full, f, indent=4)
        print(f"Full model results written to: {output_file}")

        ## Results on subsets averaged across model classes
        output_file = os.path.join(args.outdir, "subset_evals.json")
        with open(output_file, "w") as f:
            json.dump(avg_acc, f, indent=4)
        print(f"Subset results written to: {output_file}")

    # Print results
    print("#### MODEL RESULTS on generative task ####")
    for model, vals in acc_full.items():
        print(f"{model}: {vals['mean']} ({vals['ci_lower']}, {vals['ci_upper']})")

    # Create the model classes plot
    plot_subplots(avg_acc, args.outdir)
