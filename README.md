# Pytorch Seq2Seq Translation

A simple example on implementing a seq2seq attention-based encoder-decoder network using pytorch for translating french to english. Translations between any 2 languages will work as well. 

## Getting Started

### Prerequisites

The program requires the following dependencies.  

```
python 3.6
pytorch
numpy
CUDA (for using GPU)
matplotlib
```

### Installing

A step by step series of examples that tell you how to get a development env running

Download language data from [here](https://download.pytorch.org/tutorial/data.zip) and place them in current directory. 

Create new virtual environment (recommended)

```
conda create -n seq2seq2-pytorch python=3.6
```

Install required python packages

```
pip install -r requirements.txt
```

## Running

### Training and Testing

To simultaneously train and test the model, simply run `main.py`. 

Print and Plot intervals can be changed in the main function. Attentions of decoder output is also visualized during testing. 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The Pytorch Team
