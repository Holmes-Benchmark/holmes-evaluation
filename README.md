<div align="center">
<img style="vertical-align:middle" height="300" src="logo.svg" />
</div>

<h4 align="center">
    <p>
        <a href="https://holmes-benchmark.github.io/">Project</a> |
        <a href="https://holmes-explorer.streamlit.app/">Explorer 🔎</a> |
        <a href="https://holmes-leaderboard.streamlit.app/">Leaderboard 🚀</a>
    <p>
</h4>

This repository is part of the [Holmes 🔎 benchmark](https://holmes-benchmark.github.io).
It provides code to evaluate a language model on Holmes 🔎, _the comprehensive option_, or FlashHolmes ⚡, _the efficient variant_.


# 🔥 How does it work?


## ⚙️ Setting up the environment
To evaluate your desired language model on Holmes 🔎 or FlashHolmes ⚡ make sure to install all the requirements:
* Please make sure that you use python `3.10`
* Install the required packages using `pip install -r requirements.txt`
* If you want to load a language models in `four_bit`, install `bitsandbytes`. In case you trouble with installing, we rely on the version [`0.40.0`](https://github.com/TimDettmers/bitsandbytes/tree/0.40.0) and build the library locally. Make sure `python3 -m bitsandbytes` runs without errors.

## 🗃️ Getting the data
Don't worry about parsing linguistic corpora and compose probing datasets, we already did that for.
You can find the instructions to download Holmes 🔎 ([here](data/holmes/README.md)) and FlashHolmes ⚡ ([here](data/flash-holmes/README.md)).

## 🔎 Investigate your language model
After making sure all things are set up, the evaluation can start. 
For the ease of use, you only need to run the investigation script (`src/investigate.py`) and provide the following essential commands:
* `--model_name` the huggingface tag of the model to investigate, for example [`google/ul2`](https://huggingface.co/google/ul2).
* `--version` the specific benchmark version to evaluate on. This corresponds to the [data](data) folder, either ([`holmes`](data/holmes)) for Holmes 🔎 ([here](data/holmes/README.md)) or ([`flash-holmes`](data/flash-holmes)) for FlashHolmes ⚡.
* `--parallel_probing` add this flag parameter if you are in a hurry and want to parallelize stuff.

<details>
<summary>Additional parameters you may need.</summary>

* `--dump_folder` (default `./dumps`) the folder to save the encoded probing datasets. 
* `--force_encoding` add this flag parameter if you want to replace the dumped encodings of the probing dataset. Otherwise, we skip probing datasets when they are already encoded.
* `--model_precision` (default `full`) this specifies the precision to use when loading the language model, either `full`, `half`, or `four_bit`. Make sure to install `bitsandbytes` when you want to use `four_bit`.
* `--encoding_batch_size` (default `10`) the batch size when we encode the probing datasets. Lower this if you encounter out-of-memory errors on the GPU.
* `--in_filter` (default ``) define a string filter to only consider probing datasets matching this filter. For example, when setting to `rst`, we only consider probing datasets like `rst-edu-depth`
* `--control_task_types` (default `none`) whether to apply specific control tasks ([Hewitt et al., 2019](https://aclanthology.org/D19-1275/): `none` no control task is applied, `perm` input words will be shuffled randomly, `rand-weights` run the probes with random language model weights, and `randomoization` run the probes with randomized labels.
* `--run_probe` (default `True`) run the default linear probe. 
* `--run_mdl_probe` (default `False`) run the probe including minimal description length as in [Voita and Titov, 2020](https://aclanthology.org/2020.emnlp-main.14/)
* `--num_hidden_layers` (default `0`) hidden layers to consider within the probe. For example, with `0,1` we evaluate the probes once with none (linear model) and once with one intermediate layer (MLP). 
* `--seeds` (default `0,1,2,3,4`) seeds to consider when probing. With `0,1,2,3,4`, we run every probe five time using these seeds. 
* `--results_folder` (default `./results`) the folder to save the probing results.
* `--force_probing` add this flag parameter if you want to re-probe and replace already evaluated probing datasets. Otherwise, we skip already probed datasets.
* `--dump_preds` use this flag parameter and we will dump instance level prediction of every probe for all probing datasets.
</details>

After running all probes an evaluation, you will find the aggregated results in the results folder. Either in `results_holmes.csv` for Holmes 🔎 or `results_flash-holmes.csv` for FlashHolmes ⚡.


# 📖 References

Hewitt, J., & Liang, P. (2019, November). Designing and Interpreting Probes with Control Tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 2733-2743).

Voita, E., & Titov, I. (2020). Information-theoretic probing with minimum description length. In EMNLP 2020-2020 Conference on Empirical Methods in Natural Language Processing, Proceedings of the Conference (pp. 183-196).