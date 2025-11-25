from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.sparse.csgraph import shortest_path
import umap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, f_oneway
import scipy.stats as st
import os


class Visualize:
    def __init__(self):
        pass

    def plotProbing(
        self,
        plot,
        folder,
        perspective,
        id,
        iter,
        f1_score_all,
        val_acc_all,
        test_acc_mlp,
        f1_score_mlp,
        f1_score_Y_shuffled,
        val_acc_Y_shuffled,
        test_acc_mlp_Y_shuffled,
        f1_score_mlp_Y_shuffled,
    ):
        if plot == "heatmap":
            self.plot_heatmap(
                f1_score_all,
                "Linear Probe f1 score",
                save_path=f"{folder}/{perspective}_f1_score_{id}",
            )
            self.plot_heatmap(
                val_acc_all,
                "Linear Probe Acc.",
                save_path=f"{folder}/{perspective}_lr_acc_{id}",
            )
            # plot_heatmap(roc_auc_all,
            #             "ROC AUC Val",
            #             save_path= f"{folder}/{perspective}_val_auc_{id}" )
            self.plot_heatmap(
                test_acc_mlp,
                "Non-linear Probe acc",
                save_path=f"{folder}/{perspective}_mlp_acc_{id}",
            )
            self.plot_heatmap(
                f1_score_mlp,
                "Non-linear Probe f1 score",
                save_path=f"{folder}/{perspective}_f1_score_mlp_{id}",
            )
        elif plot == "linegraph":
            return
            if id != "AH":
                test_data_list = [val_acc_all, test_acc_mlp]
                train_data_list = [f1_score_all, f1_score_mlp]
                train_data_list_control = [
                    f1_score_Y_shuffled,
                    f1_score_mlp_Y_shuffled,
                ]
                test_data_list_control = [val_acc_Y_shuffled, test_acc_mlp_Y_shuffled]
                plt_title = f"{id} per Layer"
                save_path = f"{folder}/Linegraph_{id}_{str(iter)}"
                self.plot_all_linegraph(
                    train_data_list,
                    test_data_list,
                    train_data_list_control,
                    test_data_list_control,
                    plt_title,
                    save_path=save_path,
                )
            else:
                self.plot_linegraph(
                    val_acc_all,
                    f"LR acc-{id} per Layer",
                    save_path=f"{folder}/Linegraph_val_acc_{id}",
                )
                self.plot_linegraph(
                    test_acc_mlp,
                    f"MLP acc-{id} per Layer",
                    save_path=f"{folder}/Linegraph_test_acc_{id}",
                )
                self.plot_linegraph(
                    f1_score_mlp,
                    f"MLP f1 score-{id} per Layer",
                    save_path=f"{folder}/Linegraph_train_acc_{id}",
                )

    def plot_heatmap(self, ht, name, save_path=None):
        # Increase global font size for all text elements
        plt.rcParams.update({"font.size": 22})

        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(15, 12))

        # Create a heatmap using seaborn
        sns.heatmap(
            ht,
            ax=ax,
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"drawedges": False, "shrink": 0.5},
            cbar=True,
            square=True,
            annot=True,
            annot_kws={"size": 6},
            fmt=".2f",
        )

        # Customize the colorbar
        cbar = ax.collections[0].colorbar
        cbar.outline.set_linewidth(2)  # Set colorbar outline width

        # Set the ticks for x and y axes with specified interval
        ax.set_xticks(np.arange(0.5, ht.shape[1], 5))
        ax.set_yticks(np.arange(0.5, ht.shape[0], 5))

        # Set the tick labels for x and y axes with specified interval and keep x-axis labels horizontal
        ax.set_xticklabels(np.arange(0, ht.shape[1], 5), rotation=0)
        ax.set_yticklabels(np.arange(0, ht.shape[0], 5))

        # Set axis labels and title with increased padding and font size
        ax.set_xlabel("Head", fontsize=24, labelpad=20)
        ax.set_ylabel("Layer", fontsize=24, labelpad=20)
        # ax.set_title(name, fontsize=28)

        # Reinstate axis lines with specified linewidth
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_visible(True)
            ax.spines[axis].set_linewidth(2)

        # Optionally save the figure as a PDF with vectorized content
        if save_path:
            # print(save_path + '.png')
            plt.savefig(save_path + ".png", format="png", bbox_inches="tight")

        # Clear the current figure's memory to prevent resource leaks
        plt.close(fig)

    def plot_linegraph(self, ht, pt_title, save_path=None):
        num_layers = len(ht)
        num_heads = len(ht[0])
        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(10, 7))

        for head in range(num_heads):
            head_acc = [ht[layer][head] for layer in range(num_layers)]
            ax.plot(range(num_layers), head_acc, label=f"Head {head}", marker="s")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title(pt_title)
        plt.grid()
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
        # ax.grid(True)
        # ax.tight_layout()
        # Optionally save the figure as a PDF with vectorized content
        if save_path:
            # print(save_path + '.png')
            plt.savefig(save_path + ".png", format="png", bbox_inches="tight")

        # Clear the current figure's memory to prevent resource leaks
        plt.close(fig)

    def plot_all_linegraph(
        self,
        train_data_list,
        test_data_list,
        train_data_list_control,
        test_data_list_control,
        pt_title,
        save_path=None,
    ):
        data_label_list = ["L", "NL"]
        ht = train_data_list[0]
        num_layers = len(ht)
        num_heads = len(ht[0])
        # Create a figure and a single subplot
        fig, ax = plt.subplots(figsize=(10, 8))
        # plot random guess acc at 0.5
        y = [0.5] * num_layers
        ax.plot(
            range(num_layers),
            y,
            linestyle="dashed",
            label=f"random",
            color="k",
            linewidth=1,
        )

        # for head in range(num_heads):
        #     head_acc = [ht[layer][head] for layer in range(num_layers)]
        #     ax.plot(range(num_layers), head_acc, label=f"Head {head}", marker = "s")
        data_color_list = ["tab:blue", "tab:orange", "tab:green"]
        temp_control_label_list = ["control-L", "control-NL"]
        temp_control_color = ["#000000", "#808080"]
        if True:
            # viz. train data
            save_path = f"{save_path}_train_test"
        for i in range(len(train_data_list)):
            if True:
                # viz. train data
                ax.plot(
                    range(num_layers),
                    train_data_list[i],
                    linestyle="dashed",
                    label=f"f1_{data_label_list[i]}",
                    color=data_color_list[i],
                )
            # viz. test data
            ax.plot(
                range(num_layers),
                test_data_list[i],
                marker="s",
                label=f"acc_{data_label_list[i]}",
                color=data_color_list[i],
            )
            ax.plot(
                range(num_layers),
                test_data_list_control[i],
                label=f"{temp_control_label_list[i]}",
                color=temp_control_color[i],
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        # ax.set_xlim(0, 15)
        ax.set_title(pt_title)
        plt.grid()
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=8)
        # ax.grid(True)
        # ax.tight_layout()
        # Optionally save the figure as a PDF with vectorized content
        if save_path:
            # print(save_path + '.png')
            plt.savefig(save_path + ".png", format="png", bbox_inches="tight")

        # Clear the current figure's memory to prevent resource leaks
        plt.close(fig)

    def dimentionalityReduction(self, proj_key, X_value, y_labels):
        return_proj = ""
        sil_score = ""
        inter, intra = "", ""
        if proj_key == "PCA":
            x_pca_proj = PCA(n_components=2).fit_transform(
                X_value
            )  # reduced attention heads, from(240, 64) to(240, 2)
            sil_score = silhouette_score(x_pca_proj, y_labels)
            intra, inter = self.class_distances(x_pca_proj, y_labels)

            return_proj = x_pca_proj
        elif proj_key == "TSNE":
            x_tsne_proj = TSNE(n_components=2, perplexity=30).fit_transform(X_value)
            return_proj = x_tsne_proj
        elif proj_key == "UMAP":
            umap_model = umap.UMAP(n_components=2, n_neighbors=15)
            x_umap_proj = umap_model.fit_transform(X_value)
            # G = umap_model.graph_
            # D_umap = shortest_path(G, directed=False)
            # sil_score = silhouette_score(D_umap, y_labels, metric="precomputed")
            sil_score = silhouette_score(x_umap_proj, y_labels)
            return_proj = x_umap_proj
            intra, inter = self.class_distances(x_umap_proj, y_labels)

        return return_proj, sil_score, intra, inter

    def plot_individual_DR(
        self,
        proj_key,
        dataset_name,
        X_value,
        label_y,
        belief_types,
        plt_title,
        filename,
        save_path=None,
    ):
        # True-correct false belief identification, blue
        # False-incorrect false belief identification, red
        y_labels = ["1st_belief", "2nd_belief", "world"]
        y_colors = ["blue", "green"]
        x_projected_val, sil_score, intra, inter = self.dimentionalityReduction(
            proj_key, X_value, label_y
        )
        y = []
        if proj_key == "PCA":
            x_label = "PC1"
            y_label = "PC2"
        elif proj_key == "TSNE":
            x_label = "t-SNE1"
            y_label = "t-SNE2"
        elif proj_key == "UMAP":
            x_label = "UMAP1"
            y_label = "UMAP2"
        if True:
            for i in range(len(label_y)):
                if label_y[i] == 1:
                    y.append("blue")
                else:
                    y.append("red")
            legend_elements = [
                Patch(color="blue", label="correct belief"),
                Patch(color="red", label="incorrect belief"),
            ]
        else:
            for i in range(len(label_y)):
                if label_y[i]:
                    if belief_types[i].startswith("1st"):
                        y.append("blue")
                    else:
                        y.append("green")
                else:
                    if belief_types[i].startswith("1st"):
                        y.append("red")
                    else:
                        y.append("brown")
            legend_elements = [
                Patch(color="blue", label="correct FB_1storder"),
                Patch(color="red", label="incorrect FB_1storder"),
                Patch(color="green", label="correct FB_2ndorder"),
                Patch(color="brown", label="incorrect FB_2ndorder"),
            ]

        plt.figure(figsize=(9, 7))
        plt.scatter(x_projected_val[:, 0], x_projected_val[:, 1], c=y, alpha=0.6)
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        # Reduce tick label font size
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
        plt.title(plt_title, fontsize=14)
        # Optionally save the figure as a .png
        if save_path:
            # print(save_path)
            filename = f"{save_path}/{filename}.png"
            plt.savefig(filename, format="png", bbox_inches="tight")
        else:
            plt.show()
        # Clear the current figure's memory to prevent resource leaks
        plt.close()
        return sil_score, intra, inter

    def class_distances(self, X_data, y_labels):
        labels = np.array(y_labels)
        unique = np.unique(labels)

        intra = {}
        inter = {}
        distance_arry = pairwise_distances(X_data)

        for cls in unique:
            idx = np.where(labels == cls)[0]
            other_idx = np.where(labels != cls)[0]

            # Intra-class
            d_intra = distance_arry[np.ix_(idx, idx)]
            intra[cls] = np.mean(d_intra[np.triu_indices_from(d_intra, k=1)])

            # Inter-class
            d_inter = distance_arry[np.ix_(idx, other_idx)]
            inter[cls] = np.mean(d_inter)

        return intra, inter

    def plot_all_layers_DR(
        self,
        all_attn_head_mtx,
        Y_label,
        belief_types,
        id,
        dataset_name,
        probe_results_path,
        model_name_folder,
        prompt_type,
        filepath,
    ):
        num_layer = all_attn_head_mtx.shape[1]
        if id == "AH":
            num_attn_heads = all_attn_head_mtx.shape[2]
            plt_title = f"Layer <l>, Attention Head <ah>"
            filename = f"<vis>_Layer<l>_AH<ah>"
        else:
            num_attn_heads = 1
            plt_title = f"Layer <l>"
            filename = f"<vis>_Layer<l>"
        total_plots = num_layer * num_attn_heads
        vis_list = ["PCA", "TSNE", "UMAP"]
        separability_dict = {
            "layers": list(range(num_layer)),
            "pca_sil_score": [],
            "umap_sil_score": [],
            "pca_intra": [],
            "pca_inter": [],
            "umap_intra": [],
            "umap_inter": [],
        }
        for vis in vis_list:
            save_path = f"{probe_results_path}/{dataset_name}/{model_name_folder}_{prompt_type}/{vis}_{id}"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            print(f"{vis}...")
            for layer_idx in tqdm(range(num_layer)):  # each layer
                for attn_head_idx in range(num_attn_heads):  # each AH
                    if len(all_attn_head_mtx.shape) == 4:
                        X_plot = all_attn_head_mtx[:, layer_idx, attn_head_idx, :]
                    else:
                        X_plot = all_attn_head_mtx[:, layer_idx, :]
                    filename_updated = (
                        filename.replace("<vis>", vis)
                        .replace("<l>", str(layer_idx))
                        .replace("<ah>", str(attn_head_idx))
                    )
                    plt_title_updated = plt_title.replace(
                        "<l>", str(layer_idx)
                    ).replace("<ah>", str(attn_head_idx))
                    sil_score, intra, inter = self.plot_individual_DR(
                        vis,
                        dataset_name,
                        X_plot,
                        Y_label,
                        belief_types,
                        plt_title_updated,
                        filename_updated,
                        save_path,
                    )
                    if vis == "PCA":
                        separability_dict["pca_sil_score"].append(sil_score)
                        separability_dict["pca_intra"].append(intra)
                        separability_dict["pca_inter"].append(inter)
                    elif vis == "UMAP":
                        separability_dict["umap_sil_score"].append(sil_score)
                        separability_dict["umap_intra"].append(intra)
                        separability_dict["umap_inter"].append(inter)

        sep_df = pd.DataFrame.from_dict(separability_dict)
        print(sep_df)
        sep_df.to_csv(f"{filepath}/model_seperability.csv")

    def resultComparedTest(self):
        # compare result of sublayers(mhsa, ffn, hs)
        # compare results of prompts(qa, mcq, fb, comp, tf, tfr)
        # comapre results of sa-datatype and sm-datatype
        pass

    def pairedTTest(
        self,
        df,
        root_path,
        n_layers=None,
        title="Layer-wise Belief Representation: MHSA vs FFN vs Hidden State",
    ):
        """
        Visualize MHSA, FFN, and HS accuracies across layers for multiple prompt and probe types.

        Required df columns:
        ['layer_number', 'layer_type', 'prompt_type', 'probe_type', 'accuracy']
        layer_type must include: ['self-attn', 'ffn', 'hidden-state']
        """

        prompts = df["prompt_type"].unique()
        probes = df["probe_type"].unique()
        n_rows, n_cols = len(prompts), len(probes)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])
        results = []
        for i, prompt in enumerate(prompts):
            for j, probe in enumerate(probes):
                ax = axes[i, j]
                sub = df[(df["prompt_type"] == prompt) & (df["probe_type"] == probe)]
                pivot = sub.pivot_table(
                    index="layers_number", columns="layers_tm", values="accuracy"
                )
                heatmap_disp = sub.pivot_table(
                    index="layers_tm", columns="layers_number", values="accuracy"
                )
                pivot = pivot.sort_index()
                if n_layers:
                    pivot = pivot.head(n_layers)

                # Extract each
                mhsa = pivot.get("SAL", pd.Series())
                ffn = pivot.get("MLP", pd.Series())
                hs = pivot.get("HS", pd.Series())

                # Compute Repeated Measures ANOVA
                try:
                    f_stat, p_val = f_oneway(mhsa, ffn, hs)
                except Exception:
                    f_stat, p_val = np.nan, np.nan

                # Pairwise t-tests
                def safe_t(a, b):
                    if len(a) == len(b):
                        t, p = ttest_rel(a, b)
                        return round(t, 3), round(p, 5)
                        # return f"t={t:.2f}, p={p:.4e}"
                    else:
                        return np.nan, np.nan

                # stats_summary_list.append(stats_summary)
                t1, p1 = safe_t(mhsa, ffn)
                t2, p2 = safe_t(mhsa, hs)
                t3, p3 = safe_t(ffn, hs)

                # Plot
                sns.heatmap(
                    heatmap_disp,
                    annot=False,
                    fmt=".2f",
                    cmap="RdBu_r",
                    vmin=0.0,
                    vmax=1.0,
                    cbar_kws={"label": "Probe Accuracy"},
                    ax=ax,
                )

                ax.set_title(f"{prompt} ({probe})", fontsize=10)
                ax.set_xlabel("Sublayer Type")
                ax.set_ylabel("Layer Number")

                results.append(
                    {
                        "Prompt Type": prompt,
                        "Probe Type": probe,
                        "ANOVA F": round(f_stat, 3),
                        "ANOVA p": round(p_val, 5),
                        "t(MHSA vs FFN)": t1,
                        "p(MHSA vs FFN)": p1,
                        "t(MHSA vs HS)": t2,
                        "p(MHSA vs HS)": p2,
                        "t(FFN vs HS)": t3,
                        "p(FFN vs HS)": p3,
                    }
                )

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        addTimestamp = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
        filename = f"{root_path}/repeated_anova_{addTimestamp}.png"
        plt.savefig(filename, format="png", bbox_inches="tight")
        plt.close()
        result_df = pd.DataFrame(results)
        print(result_df)

    def compare_prompt(self, path, filename, ext):
        df_dataset = pd.read_csv(f"{path}/{filename}")
        df_dataset_mlp = df_dataset[df_dataset["probe_type"] == "nonlinear-MLP"]
        df_dataset_lr = df_dataset[df_dataset["probe_type"] == "linear-LR"]
        # addTimestamp = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
        table = df_dataset_mlp.pivot_table(
            index="layers_number",
            columns=["prompt_type", "layers_tm"],
            values="roc_auc",
        )

        table = table.sort_index(axis=1, level=[0, 1])
        # print(table.to_string())
        table.to_csv(f"{path}/auc/probe_results_table_SVM_{ext}_auc.csv")

        table = df_dataset_lr.pivot_table(
            index="layers_number",
            columns=["prompt_type", "layers_tm"],
            values="roc_auc",
        )

        table = table.sort_index(axis=1, level=[0, 1])
        # print(table.to_string())
        table.to_csv(f"{path}/auc/probe_results_table_LR_{ext}_auc.csv")

    def compare_prompt_repeat(self, path, filename, ext):
        df_dataset = pd.read_csv(f"{path}/{filename}")
        df_dataset_mlp = df_dataset[df_dataset["probe_type"] == "nonlinear-MLP"]
        df_dataset_lr = df_dataset[df_dataset["probe_type"] == "linear-LR"]
        # addTimestamp = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
        table = df_dataset_mlp.pivot_table(
            index="layers_number",
            columns=["prompt_type", "layers_tm"],
            values="f1_scores",
        )

        table = table.sort_index(axis=1, level=[0, 1])
        # print(table.to_string())
        table.to_csv(f"{path}/f1score/probe_results_table_SVM_{ext}_f1score.csv")

        table = df_dataset_lr.pivot_table(
            index="layers_number",
            columns=["prompt_type", "layers_tm"],
            values="f1_scores",
        )

        table = table.sort_index(axis=1, level=[0, 1])
        # print(table.to_string())
        table.to_csv(f"{path}/f1score/probe_results_table_LR_{ext}_f1score.csv")

        # prompts = df_dataset["prompt_type"].unique()
        # layers = df_dataset["layers_tm"].unique()
        # results = {"prompt_type": prompts.tolist()}
        # for layer in layers:
        #     results[f"{layer}_mean_acc"] = []
        #     results[f"{layer}_ci"] = []
        #     for prompt in prompts:
        #         # probetype_check = "nonlinear" in df_dataset["probe_type"]
        #         acc = df_dataset[
        #             (df_dataset["prompt_type"] == prompt)
        #             & (df_dataset["layers_tm"] == layer)
        #             & (df_dataset["probe_type"] == "nonlinear-MLP")
        #         ]["accuracy"].values
        #         mean_acc = np.mean(acc)
        #         ci = st.t.interval(
        #             0.95,  # 95% confidence
        #             len(acc) - 1,  # degrees of freedom
        #             loc=mean_acc,  # sample mean
        #             scale=st.sem(acc),  # standard error
        #         )
        #         results[f"{layer}_mean_acc"].append(f"{mean_acc:.3f}")
        #         results[f"{layer}_ci"].append(f"{ci[0]:.3f}, {ci[1]:.3f}")
        #         # results.append((prompt, mean_acc, ci))
        # # print(results)
        # df_results = pd.DataFrame(results).set_index("prompt_type")
        # print(df_results)
        # new_filename = f"{path}\Prompt_compare_NL_{addTimestamp}.csv"
        # df_results.to_csv(new_filename)

        # Display results
        # for prompt, mean_acc, ci in results:
        # print(f"{prompt}: mean={mean_acc:.3f}, 95% CI=({ci[0]:.3f}, {ci[1]:.3f})")

    def multi_prompt_lineplot(
        self, filepath, filename, model_name, data_type, num_layers
    ):
        df_dataset = pd.read_csv(f"{filepath}/{filename}")
        prompt_color_list = [
            "tab:red",
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:cyan",
            "tab:pink",
        ]
        prompt_list = [
            "qa_prompt",
            "comp_prompt",
            "mc_prompt",
            "fb_prompt",
            "tf_prompt",
            "tfr_prompt",
        ]
        layer_tm_list = ["SAL", "MLP", "HS"]
        layer_tm = layer_tm_list[1]
        num_layers = num_layers
        fig_nl, ax_nl = plt.subplots(figsize=(10, 8))
        # fig_auc, ax_auc = plt.subplots(figsize=(10, 8))
        # non-linear
        temp_filter_df = []
        for i in range(len(prompt_list)):
            temp_filter_df = df_dataset[
                (df_dataset["probe_type"] == "nonlinear-MLP")
                & (df_dataset["prompt_type"] == f"{prompt_list[i]}")
                & (df_dataset["layers_tm"] == f"{layer_tm}")
            ]
            x_data_acc_nl = temp_filter_df["accuracy"].to_list()
            x_data_f1_nl = temp_filter_df["f1_scores"].to_list()
            x_data_auc_nl = temp_filter_df["roc_auc"].to_list()
            ax_nl.plot(
                range(num_layers),
                x_data_acc_nl,
                label=f"{prompt_list[i]}",
                color=prompt_color_list[i],
            )
            ax_nl.plot(
                range(num_layers),
                x_data_f1_nl,
                linestyle="dashed",
                color=prompt_color_list[i],
            )
            # ax_auc.plot(
            #     range(num_layers),
            #     x_data_auc_nl,
            #     label=f"{prompt_list[i]}",
            #     color=prompt_color_list[i],
            # )
        ax_nl.set_xlabel("Layer")
        ax_nl.set_ylabel("Accuracy")
        ax_nl.set_ylim(0.0, 1.0)
        ax_nl.set_title(f"{model_name}")
        plt.grid(True)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=8)
        plt.savefig(
            f"{filepath}/result_plot/{model_name}_{data_type}_{layer_tm}_NL.png",
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig_nl)

        # linear
        fig_l, ax_l = plt.subplots(figsize=(10, 8))
        temp_filter_df = []
        for i in range(len(prompt_list)):
            temp_filter_df = df_dataset[
                (df_dataset["probe_type"] == "linear-LR")
                & (df_dataset["prompt_type"] == f"{prompt_list[i]}")
                & (df_dataset["layers_tm"] == f"{layer_tm}")
            ]
            x_data_acc_l = temp_filter_df["accuracy"].to_list()
            x_data_f1_l = temp_filter_df["f1_scores"].to_list()
            x_data_auc_l = temp_filter_df["roc_auc"].to_list()
            ax_l.plot(
                range(1, num_layers + 1),
                x_data_acc_l,
                label=f"{prompt_list[i]}",
                color=prompt_color_list[i],
            )
            ax_l.plot(
                range(1, num_layers + 1),
                x_data_f1_l,
                linestyle="dashed",
                color=prompt_color_list[i],
            )
            # ax_auc.plot(
            #     range(1, num_layers + 1),
            #     x_data_auc_l,
            #     linestyle="dashed",
            #     color=prompt_color_list[i],
            # )

        ax_l.set_xlabel("Layer")
        ax_l.set_ylabel("Accuracy")
        ax_l.set_ylim(0.0, 1.0)
        ax_l.set_title(f"{model_name}")
        plt.grid(True)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=8)
        plt.savefig(
            f"{filepath}/result_plot/{model_name}_{data_type}_{layer_tm}_L.png",
            format="png",
            bbox_inches="tight",
        )
        plt.close(fig_l)
        # ax_auc.set_xlabel("Layer")
        # ax_auc.set_ylabel("ROC-AUC")
        # ax_auc.set_ylim(0.0, 1.0)
        # ax_auc.set_title(f"{model_name}")
        # plt.grid()
        # plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize=8)
        # plt.savefig(
        #     f"{filepath}/{model_name}_{data_type}_ROC_AUC.png",
        #     format="png",
        #     bbox_inches="tight",
        # )
        # plt.close(fig_auc)
