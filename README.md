<div align="center">
<img style="vertical-align:middle" height="300" src="logo.svg" />
</div>

This repository is part of the [Holmes 🔎 benchmark](https://holmes-benchmark.github.io).
It provides code to evaluate an language model on Holmes 🔎, the comprehensive option, or FlashHolmes ⚡, the efficient variant.


# 🔎 How does it work?
To evaluate your desired language model on Holmes 🔎 or FlashHolmes ⚡, follow these three steps: 
1. **Encoding** Encode inputs of the probing datasets with the language model - more detail [bellow](#encoding).
2. **Probing** Run to probes on the encoded datasets, see [probing](#probing). 
3. **Evaluating** Gather probing results to get an overview of the linguistic knowledge of your chosen language model, following [evaluating](#evaluating).

## 🔎 <a name="requirements"></a>Requirements
Before starting you need to setup all the requirements: 
* Please make sure that you use python `3.10`
* Install the required packages using `pip install -r requirements.txt`
* If you wish to use `four_bit` quantification install `bitsandbytes`. In case you trouble with installing, we rely on the version [`0.40.0`](https://github.com/TimDettmers/bitsandbytes/tree/0.40.0) and build the library locally. Make sure `python3 -m bitsandbytes` runs without errors.


@click.option('--config_file_path', type=str, default='../holmes-datasets/cwi/config-none.yaml')
@click.option('--encoding_batch_size', type=int, default=10)
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--encoding', type=str, default="full")
@click.option('--dump_folder', type=str, default="../dumps")
@click.option('--force', is_flag=True, default=False)

## 🔎 <a name="encoding"></a>First <u>Encoding</u>
## 🔎 <a name="probing"></a>Second <u>Probing</u>
## 🔎 <a name="evaluating"></a>Third <u>Evaluating</u>


For more details, check out our webpage https://holmes-benchmark.github.io or publication _HOLMES: Benchmark Linguistic Knowledge in Language Models_
