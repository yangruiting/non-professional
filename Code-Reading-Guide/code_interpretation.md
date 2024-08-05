# Code Structure

｜—— `README.md`  Description of the project.

｜—— `requirements.txt`  The third-party libraries used in this project. (Installation command: ***python3 -m pip install -r requirements.txt***)

｜—— `pretrain.py`  The execution file of the project. It includes the entire process of the project, including loading, training and saving data and models. Other Python files are called by it.

｜—— `pretrain.sh` Project execution command.

｜—— `tokenization.py` Includes methods related to token processing, including tokenizer training, token generation, and padding and truncation of token sequences, etc.

｜—— `model.py` Our transformer-based model, methods and algorithms related to model building are all here.

｜—— `optimization.py` Methods related to deep learning optimization, including optimizers and learning rate schedulers.

｜—— `data` The data folder that needs to be prepared before training, including training data `train.txt` and evaluation data `eval.txt`, and `vocab.json` and `merges.txt` are used to load the tokenizer (assuming they have been generated through training before).

｜—— `result`  The output folder named by the user includes the saved models and the results of the training process.

｜—— `__ pycache __`   A folder automatically generated after the Python project is run. 

｜   cpython means that the Python interpreter is implemented in C language, and -39 means the version is 3.9.     

｜—— `runs`  Automatically created to store information related to model training, logging, or experiment tracking.

# The role of the library

**1. apex**

***why use it?***    

During deep learning training, the data type defaults to single-precision FP32. In order to speed up training time and reduce the memory occupied by network training while maintaining the model accuracy, a mixed-precision training method has emerged. APEX is an open source tool from NVIDIA that perfectly supports the PyTorch framework and is used to change data formats to reduce the model's memory usage. **apex.amp** (Automatic Mixed Precision) tests most operations of the model using the Float16 data type, and some special operations still use Float32.

***How do we use it?***

```
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
```
The opt-level parameter is the user's choice of data format for training. Specifically, 

O0: pure FP32; O1: mixed precision training; O2: almost FP16; O3: pure FP16.

**2. mmengine**

***why use it?*** 

FLOPS (floating-point operations per second), floating-point operations per second. It is often used to estimate the performance of a computer. The parameter value reflects the usage of display memory. **mmengine.analysis.get_model_complexity_info** to get the complexity of a model.
A good network model not only requires high accuracy, but also requires a small number of parameters and computational complexity to facilitate deployment.

***How do we use it?***
```
from mmengine.analysis import get_model_complexity_info
outputs = get_model_complexity_info(model, input_shape)
```

**3. tokenizers**

Hugging Face's Tokenizers library provides a fast and efficient way to process natural language text (i.e., tokenization) for subsequent machine learning model training and reasoning. This library provides a variety of pre-trained tokenizers, such as BPE, Byte-Pair Encoding (Byte-Level BPE), WordPiece, etc., which are widely used tokenization methods in modern NLP models (such as BERT, GPT-2, RoBERTa, etc.).

For more information, please refer to: [Hugging-Face tokenizers](https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/index).

**4. tqdm**

***why use it?*** 

tqdm is a Python progress bar library. tqdm loads an iterable object and displays the loading progress of the iterable object in real time in the form of a progress bar.

***How do we use it?***

```
from tqdm import tqdm, trange
train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
```
_train_dataloader_ is the loaded iterator, and _desc_ is the description text of the progress bar.

**5. scikit_learn**

Scikitlearn is an essential library for machine learning. It provides a variety of algorithms. We mainly use it to calculate metrics. 

For more information, please refer to: [scikitlearn 中文文档](https://scikitlearn.com.cn/)

# How does the project work?

In PyCharm IDE, you can use **view-Tool Windows-Structure** to see the code's structure. 

***Don't read the code from top to bottom！！！*** For mainstream programming languages ​​​​(such as C), it is crucial to ***get used to reading the code from the main function***. Although Python does not have a main function, `__name__ == "\__main__"` can be considered similar to it.

Open `pretrain.py`, logically, you should see the `main()` function first. 

**1.** Before Start

argparse is an argument and command parser. 
```
parser = argparse.ArgumentParser()  # Creating a parser
parser.add_argument("--model_dim", default=1, type=int, required=False)  # Adding a parameter to the parser
args = parser.parse_args()  # Parse the parameters and you can use them later
```
The logging module is used to record logs. When the program is running, you can output the running progress in the terminal and save it in the file. If you are not familiar with this module, you can check it out in [logging](https://docs.python.org/3/library/logging.html).

**2.** Load

We loaded the tokenizer and model using the following command.
```
tokenizer = DNATokenizer(args)
model = GPT2LMHeadModel(args)
```
DNATokenizer() and GPT2LMHeadModel() are classes written in `tokenization.py` and `model.py` respectively. They are called through the `from...import...` command and instantiated here.

Load the data and convert the sequence into token_ids using the following command.
```
train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
```
Now, you can check out the implementation of the `load_and_cache_examples()` function from the code above.

**3.** Train

DataLoader encapsulates the custom dataset into a Tensor of the same batch size according to the batch size and whether to shuffle it, for subsequent training.
```
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
```
Use the following command to extract the data:
```
epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
for step, batch in enumerate(epoch_iterator):
```

The data in each batch is input into the model, where att_mask consists of 0 and 1, and the att_mask value at the padding position is 0 and does not participate in learning.
```
inputs = batch[0]
labels = batch[0]
att_mask = batch[1]
outputs = model(inputs, labels=labels, attention_mask=att_mask)
```
After the loss is returned through the `model()`, a gradient update is performed using the optimizer.
```
loss.backward()
optimizer.step()
```
During the training process, when a certain number of steps are set, get the results on the evaluation dataset and save the model.

**4.** Eval

After training is complete, you can get the model's results on the evaluation dataset.
