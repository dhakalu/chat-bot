# Chatbot tutorial

In this tutorial we will create a chatbot that can answer questions about you.

## Tools we will be using

- **[conda](https://link)**: Python library to manage multiple python development environments.
- **[nltk](https://link)**: Python library used for natural language processing.
- **[numpy](https://link)**: python library used to manipulate the nd-arrays.
- **[PyTorch](https://link-to-pytorch)**: AI library used to create model and predict the [intent](http://linkto-initent) of user based on user [utterance](https://link-to-utterance).
- **[stuptools](<https://link-to-setuptools>)**: Python library used for package/dependencies management.

## Setting up the project

We have a folder named `ai` to hold all our code related to the model. The folder contains four folders:

- **__data__**: Holds text files for utterances and responses. For every file that exists in `utterances` there must be a corresponding file in the `responses` directory.
- **processors**: Logic to pre process the data.
- **ui**: Command line application that acts as the interface of the chat bot.

## Data Collection

Data collection is the first step in building any deep learning model. For this project we will be generating the data ourselves. To hold the raw data that we create we will create a directory named `__data__` under `ai` directory.

Under this directory we will create two directories: `utterances` and `responses`.

At this point we can start adding text files into `utterances` folder. Every file we add into this folder will contain one or more lines of utterances. Every utterances will match the intent that the file name represents.

For example, lets add a file named `greetings.txt` under utterances folder as below:

    hi
    hello
    sup
    whats up

Similarly, lets  add a file named `greetings.txt` under `responses` folder as shown below. Each line in this file will be replies that bot can make for `greetings` intent. One reply will be randomly picked to show in the UI.

    whats up?
    hi there, nice to meet you!
    Hi there!
    Hi there, nice to meet you!
    Hello!
    Greetings! How are you?

**Note**: data in *responses* directory is used only for ui. It is not used to feed the model. So what you put in this directory will not affect the model's performance.

Add as many intents as you would like your bot to answer questions about you.

## Data Processing

Next step at building a deep learning model is to process the data we collected in data collection step.

In this stage we will manipulate the data we collected using `numpy` and `nltk`.

### Read the File

### Tokenize and apply stemming

### Hot Bags

### DataLoader (pytorch)

## Building a Model

Next step is to architect the model that we want to build using the data we processed in previous step.

## Training the Model

Now we have defined the model we want to use to build our chatbot, we will start feeding it the data.

### Hyper Parameters

- INPUT_SIZE: numbers of the training data items
- HIDDEN_SIZE: number of hidden layers in the model
