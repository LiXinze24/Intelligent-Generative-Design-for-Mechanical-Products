# Intelligent-Generative-Design-for-Mechanical-Products
<div align="center">
  

<h2> XXX (Spotlight 🤩) </h2>

<a href="https://arxiv.org/abs/2409.17106">
  <img src="https://img.shields.io/badge/Arxiv-3498db?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=2c3e50&borderRadius=10" alt="Arxiv" />
</a>
<a href="https://sadilkhan.github.io/text2cad-project/">
  <img src="https://img.shields.io/badge/Project-2ecc71?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Project" />
</a>
<a href="https://huggingface.co/datasets/SadilKhan/Text2CAD">
  <img src="https://img.shields.io/badge/Dataset-7D5BA6?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Dataset" />
</a>




</div>

# ⚙️ Abstract
Mechanical design integrates domain knowledge, analytical reasoning, and geometric modeling to transform functional requirements into feasible structures. Achieving this process in a truly end to end manner remains challenging, as current Large Language Model (LLM) based approaches still lack three-dimensional (3D) spatial perception, awareness of physical and manufacturing constraints. This work addresses the absence of a unified, scalable and knowledge-grounded intelligent framework for mechanical design by integrating historical design knowledge, multimodal design data and generative intelligence models into an Intelligent Generative Design (IGD) paradigm. Here we demonstrate that the proposed IGD framework effectively integrates general LLMs, the supervised fine-tuned LLM and task-specific neural networks (TNNs) within a multi-agent architecture to perform intent-driven design tasks of analysis decomposition, 3D modeling, 3D assembly and kinematic reasoning, thereby enabling mechanical product design that is directly driven by semantic requirements. The framework covers design workflow and can complete complex mechanical design tasks with high geometric and assembly accuracy and without human intervention. This shows that design intent expressed at the semantic level can directly drive the full mechanical design process, from requirement analysis through detailed modeling and assembly to motion analysis, and that this provides capabilities that were not possible with traditional methods or with approaches that rely on a single neural network modality. The IGD framework provides a viable path for combining generative intelligence models and engineering knowledge in practical workflows that are firmly rooted in real industrial practice, and can support intelligent product development in a wide range of industrial domains.


# ⚙️ Installation

## 🌍 Environment

- 🐧 Linux
- 🐍 Python >=3.9

## 📦 Dependencies

```bash
$ conda env create --file environment.yml
```

# ✅ Todo List

- [x] Release Data Preparation Code
- [x] Release Training Code
- [x] Release Inference Code

# 📊 Data Preparation

Download the DeepCAD data from [here](https://github.com/ChrisWu1997/DeepCAD?tab=readme-ov-file#data).

**Generate Vector Representation from DeepCAD Json**

_You can also download the processed cad vec from [here](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/cad_seq.zip)._

```bash
$ cd CadSeqProc
$  python3 json2vec.py --input_dir $DEEPCAD_JSON --split_json $TRAIN_TEST_VAL_JSON --output_dir $OUTPUT_DIR --max_workers $WORKERS --padding --deduplicate
```


**Download the text annotations from [here](https://huggingface.co/datasets/SadilKhan/Text2CAD). Download the preprocessed [training](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/text2cad_v1.0/train_data.pkl) and [validation](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/text2cad_v1.0/validation_data.pkl) data and place it in** `Cad_VLM/dataprep` folder.

# 🚀 Training

In the `Cad_VLM/config/trainer.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `prompt_path`: Path for the text annotation.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path` (Optional): For resuming training after some epochs.

</p>
</details> 

<br>

```bash
$ cd Cad_VLM
$ python3 train.py --config_path config/trainer.yaml
```


# 🤖 Inference

### For Test Dataset

In the `Cad_VLM/config/inference.yaml`, provide the following path. Download the checkpoint for v1.0 [here](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/text2cad_v1.0/Text2CAD_1.0.pth).

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `prompt_path`: Path for the text annotation.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path`: The path to model weights. 

</p>
</details> 

<br>

```bash
$ cd Cad_VLM
$ python3 test.py --config_path config/inference.yaml
```

### Run Evaluation

```bash
$ cd Evaluation
$ python3 eval_seq.py --input_path ./output.pkl --output_dir ./output
```

### For Random Text Prompts

In the `Cad_VLM/config/inference_user_input.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path`: The path to model weights.
- `prompt_file` (Optional): For single prompt ignore it, for multiple prompts provide a txt file.

</p>
</details> 
<br>

  #### For single prompt
  
  ```bash
  $ cd Cad_VLM
  $ python3 test_user_input.py --config_path config/inference_user_input.yaml --prompt "A rectangular prism with a hole in the middle."
  ```

  #### For Multiple prompts

  ```bash
  $ cd Cad_VLM
  $ python3 test_user_input.py --config_path config/inference_user_input.yaml
  ```

# 💻 Run Demo


In the `Cad_VLM/config/inference_user_input.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path`: The path to model weights.

</p>
</details> 
<br>

```bash
$ cd App
$ gradio app.py
```



# 👥 Contributors
Our project owes its success to the invaluable contributions of these remarkable individuals. We extend our heartfelt gratitude for their dedication and support.


<br>

# ✍🏻 Acknowledgement



# 📜 Citation

If you use this dataset in your work, please consider citing the following publications.


```

```
