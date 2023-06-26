# Fast AI Practical Deep Learning course

See if you get a response to [this](https://forums.fast.ai/t/run-fast-ai-on-paperspace-gradient-with-a-free-pro-subscription/101242/18)

## Important Links

- [the course lectures](https://course.fast.ai)
- [associated book](https://course.fast.ai/Resources/book.html)
- [your own Paper Space notebook for Fast.ai course](https://console.paperspace.com/evanemolo/projects/p49n3s3zjgc/notebooks)
- [ACIIDOC to MD](https://markdown2asciidoc.com)
- [Fast.ai's datasets](https://docs.fast.ai/data.external.html)
- [Fast.ai course forum](https://forums.fast.ai)
- [Github Fast.ai book (another source for clean notebooks, in case something breaks)](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb)
- [aiquizzes.com](https://aiquizzes.com)
- [VSCode + Paperspace](https://forums.fast.ai/t/beginner-setup/95289/109?u=evanemolo)

## [Lesson 1](https://course.fast.ai/Lessons/lesson1.html)

- Fast.ai is a library
  - sits on top of PyTorch
- Jupyter Notebooks will run Python in the browser
  - creates a virtual computer when running code.
  - `!` usage in the codeblocks indicated a bash command (ie `!pwd`)
- `DataBlock`
  - an abstraction for creating a model.
- docs.fast.ai
  /tutorials
- a model: multiply things, add them, negatives are set to 0, repeat.
- it's important to experiment
  - Kaggle notebooks
  - try the bird exercise, maybe introduce more categories.
- read chapter 1.

## Chapter 1

- [Full chapter 1](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb)
- [My Paperspace Chapter 1](https://console.paperspace.com/evanemolo/notebook/rjnw81udus20vir?file=%2F01_intro.ipynb)
  - copied from an original from teacher. do the same for subsequent chapters.

Capabilities of machine learning:

- Natural language processing (NLP):: Answering questions; speech recognition; summarizing documents; classifying documents; finding names, dates, etc. in documents; searching for articles mentioning a concept
- Computer vision:: Satellite and drone imagery interpretation (e.g., for disaster resilience); face recognition; image captioning; reading traffic signs; locating pedestrians and vehicles in autonomous vehicles
- Medicine:: Finding anomalies in radiology images, including CT, MRI, and X-ray images; counting features in pathology slides; measuring features in ultrasounds; diagnosing diabetic retinopathy
- Biology:: Folding proteins; classifying proteins; many genomics tasks, such as tumor-normal sequencing and classifying clinically actionable genetic mutations; cell classification; analyzing protein/protein interactions
- Image generation:: Colorizing images; increasing image resolution; removing noise from images; converting images to art in the style of famous artists
- Recommendation systems:: Web search; product recommendations; home page layout
- Playing games:: Chess, Go, most Atari video games, and many real-time strategy games
- Robotics:: Handling objects that are challenging to locate (e.g., transparent, shiny, lacking texture) or hard to pick up
- Other applications:: Financial and logistical forecasting, text to speech, and much more...

> There will be times when the journey will feel hard. Times where you feel stuck. Don't give up! Rewind through the book to find the last bit where you definitely weren't stuck, and then read slowly through from there to find the first thing that isn't clear. Then try some code experiments yourself, and Google around for more tutorials on whatever the issue you're stuck with is—often you'll find some different angle on the material might help it to click.

> So, what sorts of tasks make for good test cases? You could train your model to distinguish between Picasso and Monet paintings or to pick out pictures of your daughter instead of pictures of your son.

> You should assume that whatever specific libraries and software you learn today will be obsolete in a year or two.

> how do we know if this model is any good? In the last column of the table you can see the error rate, which is the proportion of images that were incorrectly identified.

> In deep learning, it really helps if you have the motivation to fix your model to get it to do better. That's when you start learning the relevant theory.

> the error rate serves as our metric—our measure of model quality, chosen to be intuitive and comprehensible.

this is referring to `error_rate` in the table.

> deep learning is just a modern area in the more general discipline of machine learning. To understand the essence of what you did when you trained your own classification model, you don't need to understand deep learning. It is enough to see how your model and your training process are examples of the concepts that apply to machine learning in general.

> Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a mechanism for altering the weight assignment so as to maximize the performance. We need not go into the details of such a procedure to see that it could be made entirely automatic and to see that a machine so programmed would "learn" from its experience.
> There are a number of powerful concepts embedded in this short statement:
> The idea of a "weight assignment"
> The fact that every weight assignment has some "actual performance"
> The requirement that there be an "automatic means" of testing that performance,
> The need for a "mechanism" (i.e., another automatic process) for improving the performance by changing the weight assignments

> **Weights** are just variables, and a weight assignment is a particular choice of values for those variables.

> the **model** is a special kind of program: it's one that can do many different things, depending on the weights.

> (By the way, what Samuel called "weights" are most generally referred to as model **parameters** these days, in case you have encountered that term. The term weights is reserved for a particular type of model parameter.)

> We can now see why he said that such a procedure could be made entirely automatic and... a machine so programmed would "learn" from its experience. Learning would become entirely automatic when the adjustment of the weights was also automatic—when instead of us improving a model by adjusting its weights manually, we relied on an automated mechanism that produced adjustments based on performance.

> Machine Learning: The training of programs developed by allowing a computer to learn from its experience, rather than through manually coding the individual steps.

### What is a neural network?

> The fact that neural networks are so flexible means that, in practice, they are often a suitable kind of model, and you can focus your effort on the process of training them—that is, of finding good weight assignments.

> In other words, to recap, a neural network is a particular kind of machine learning model, which fits right in to Samuel's original conception.

> Neural networks are special because they are highly flexible, which means they can solve an unusually wide range of problems just by finding the right weights. This is powerful, because stochastic gradient descent provides us a way to find those weight values automatically.

    One could imagine that you might need to find a new "mechanism" for automatically updating weights for every problem. This would be laborious. What we'd like here as well is a completely general way to
    update the weights of a neural network, to make it improve at any given task. Conveniently, this also exists!

    This is called **stochastic gradient descent (SGD).**

> Our inputs are the images. Our weights are the weights in the neural net. Our model is a neural net. Our results are the values that are calculated by the neural net, like "dog" or "cat."

> Putting this all together, and assuming that SGD is our mechanism for updating the weight assignments, we can see how our image classifier is a machine learning model, much like Samuel envisioned.

> Samuel was working in the 1960s, and since then terminology has changed. Here is the modern deep learning terminology for all the pieces we have discussed:
> The functional form of the model is called its architecture (but be careful—sometimes people use model as a synonym of architecture, so this can get confusing).
> The weights are called parameters.
> The predictions are calculated from the independent variable, which is the data not including the labels.
> The results of the model are called predictions.
> The measure of performance is called the loss.
> The loss depends not only on the predictions, but also the correct labels (also known as targets or the dependent variable); e.g., "dog" or "cat."

### Limitations inherent to machine learning

> From this picture we can now see some fundamental things about training a deep learning model:
> A model cannot be created without data.
> A model can only learn to operate on the patterns seen in the input data used to train it.
> This learning approach only creates predictions, not recommended actions.
> It's not enough to just have examples of input data; we need labels for that data too (e.g., pictures of dogs and cats aren't enough to train a model; we need a label for each one, saying which ones are dogs, and which are cats).

> Since these kinds of machine learning models can only make predictions (i.e., _attempt to replicate labels_), this can result in a significant gap between organizational goals and model capabilities

> Another critical insight comes from considering how a model interacts with its environment. This can create **feedback loops**

- i.e. the predications are biased because the more the model is used, the more biased the data becomes

### How Our Image Recognizer Works

```python
from fastai.vision.all import *

path = untar_data(URLs.PETS)/'images'

# The filenames start with an uppercase letter if the image is a cat, and a lowercase letter otherwise.
def is_cat(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

```

> There are various different classes for different kinds of deep learning datasets and problems—here we're using `ImageDataLoaders`

> The other important piece of information that we have to tell fastai is how to get the labels from the dataset. Computer vision datasets are normally structured in such a way that the label for an image is part of the filename, or path—most commonly the parent folder name. Here we're telling fastai to use the is_cat function we just defined.

> Finally, we define the **Transforms** that we need. A Transform contains code that is applied automatically during training; fastai includes many predefined Transforms, and adding new ones is as simple as creating a Python function. There are two kinds: `item_tfms` are applied to each item (in this case, each item is resized to a 224-pixel square), while `batch_tfms` are applied to a batch of items at a time using the GPU, so they're particularly fast

> Note: Classification and Regression: classification and regression have very specific meanings in machine learning. These are the two main types of model that we will be investigating in this book. A classification model is one which attempts to predict a class, or category. That is, it's predicting from a number of discrete possibilities, such as "dog" or "cat." A regression model is one which attempts to predict one or more numeric quantities, such as a temperature or a location. Sometimes people use the word regression to refer to a particular kind of model called a linear regression model; this is a bad practice, and we won't be using that terminology in this book!

> The most important parameter to mention here is `valid_pct=0.2`. This tells fastai to hold out 20% of the data and not use it for training the model at all. This 20% of the data is called the _validation set_; the remaining 80% is called the _training set_.

> The parameter `seed=42` sets the random seed to the same value every time we run this code, which means we get the same validation set every time we run it—this way, if we change our model and retrain it, we know that any differences are due to the changes to the model, not due to having a different random validation set.

> fastai will always show you your model's accuracy using only the validation set, never the training set.

> Even when your model has not fully memorized all your data, earlier on in training it may have memorized certain parts of it. As a result, the longer you train for, the better your accuracy will get on the training set; the validation set accuracy will also improve for a while, but eventually it will start getting worse as the model starts to memorize the training set, rather than finding generalizable underlying patterns in the data. When this happens, we say that the model is _overfitting_.

> **Overfitting is the single most important and challenging issue** when training for all machine learning practitioners, and all algorithms.

> We often see practitioners using over-fitting avoidance techniques even when they have enough data that they didn't need to do so, ending up with a model that may be less accurate than what they could have achieved.

> important: Validation Set: When you train a model, you must always have both a training set and a validation set, and must measure the accuracy of your model only on the validation set. If you train for too long, with not enough data, you will see the accuracy of your model start to get worse; this is called overfitting. fastai defaults valid_pct to 0.2, so even if you forget, fastai will create a validation set for you!

> Why a CNN (convolutional neural network)? It's the current state-of-the-art approach to creating computer vision models. We'll be learning all about how CNNs work in this book. Their structure is inspired by how the human vision system works.

> Most of the time, however, picking an architecture isn't a very important part of the deep learning process. It's something that academics love to talk about, but in practice it is unlikely to be something you need to spend much time on.

> The 34 in resnet34 refers to the number of layers in this variant of the architecture (other options are 18, 50, 101, and 152). Models using architectures with more layers take longer to train, and are more prone to overfitting

> What is a metric? A metric is a function that measures the quality of the model's predictions using the validation set, and will be printed at the end of each epoch. In this case, we're using `error_rate`, which is a function provided by fastai that does just what it says: tells you what percentage of images in the validation set are being classified incorrectly. Another common metric for classification is `accuracy` (which is just `1.0 - error_rate`). fastai provides many more, which will be discussed throughout this book.

> `vision_learner` also has a parameter `pretrained`, which defaults to True (so it's used in this case, even though we haven't specified it), which sets the weights in your model to values that have already been trained by experts to recognize a thousand different categories across 1.3 million photos (using the famous ImageNet dataset). A model that has weights that have already been trained on some other dataset is called a pretrained model. You should nearly always use a pretrained model, because it means that your model, before you've even shown it any of your data, is already very capable. And, as you'll see, in a deep learning model many of these capabilities are things you'll need, almost regardless of the details of your project. For instance, parts of pretrained models will handle edge, gradient, and color detection, which are needed for many tasks.

> **Using pretrained models is the most important method we have** to allow us to train more accurate models, more quickly, with less data, and less time and money.

> _Transfer learning_: Using a pretrained model for a task different to what it was originally trained for.

> Using a pretrained model for a task different to what it was originally trained for is known as transfer learning. Unfortunately, because transfer learning is so under-studied, few domains have pretrained models available. For instance, there are currently few pretrained models available in medicine, making transfer learning challenging to use in that domain. In addition, it is not yet well understood how to use transfer learning for tasks such as time series analysis.

> When you use the `fine_tune` method, fastai will use these tricks for you. There are a few parameters you can set (which we'll discuss later), but in the default form shown here, it does two steps:
>
> 1. Use one epoch to fit just those parts of the model necessary to get the new random head to work correctly with your dataset.
> 2. Use the number of epochs requested when calling the method to fit the entire model, updating the weights of the later layers (especially the head) faster than the earlier layers (which, as we'll see, generally don't require many changes from the pretrained weights).
>    The _head_ of a model is the part that is newly added to be specific to the new dataset. An _epoch_ is one complete pass through the dataset. After calling `fit`, the results after each epoch are printed, showing the epoch number, the training and validation set losses (the "measure of performance" used for training the model), and any _metrics_ you've requested (_error rate_, in this case).

### What Our Image Recognizer Learned

> if the human eye can recognize categories from the images, then a deep learning model should be able to do so too.

This is pretty interesting. This remark is in reference to malware's binary file being divided into 8-bit sequences which are then converted to decimal values. the decimal vector is reshaped and a gray-scale image is generated that represents the malware sample.

> In general, you'll find that a small number of general approaches in deep learning can go a long way, if you're a bit creative in how you represent your data! You shouldn't think of approaches like the ones described here as "hacky workarounds," because actually they often (as here) beat previously state-of-the-art results.

### Deep learning vocabulary

| Term             | Meaning                                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Label            | The data that we’re trying to predict, such as "dog" or "cat"                                                                                          |
| Architecture     | The _template_ of the model that we’re trying to fit; the actual mathematical function that we’re passing the input data and parameters to             |
| Model            | The combination of the architecture with a particular set of parameters                                                                                |
| Parameters       | The values in the model that change what task it can do, and are updated through model training                                                        |
| Fit              | Update the parameters of the model such that the predictions of the model using the input data match the target labels                                 |
| Train            | A synonym for _fit_                                                                                                                                    |
| Pretrained model | A model that has already been trained, generally using a large dataset, and will be fine-tuned                                                         |
| Fine-tune        | Update a pretrained model for a different task                                                                                                         |
| Epoch            | One complete pass through the input data                                                                                                               |
| Loss             | A measure of how good the model is, chosen to drive training via SGD                                                                                   |
| Metric           | A measurement of how good the model is, using the validation set, chosen for human consumption                                                         |
| Validation set   | A set of data held out from training, used only for measuring how good the model is                                                                    |
| Training set     | The data used for fitting the model; does not include any data from the validation set                                                                 |
| Overfitting      | Training a model in such a way that it _remembers_ specific features of the input data, rather than generalizing well to data not seen during training |
| CNN              | Convolutional neural network; a type of neural network that works particularly well for computer vision tasks                                          |

Asking ChatGPT for a definition of _neural networks_:

> A neural network is a computational model inspired by the structure and functioning of the human brain. It consists of interconnected nodes, called neurons, which are organized into layers. The neurons in each layer receive input signals, perform computations, and produce output signals that are passed on to the next layer.
>
> Neural networks are trained using a process called machine learning, where the network learns to recognize patterns and make predictions by adjusting the strength of connections (weights) between neurons. This training is typically achieved through a learning algorithm, such as backpropagation, which iteratively adjusts the weights based on the network's performance on a set of training examples.
>
> The strength of neural networks lies in their ability to learn and extract meaningful representations from complex and unstructured data, such as images, audio, and text. They have been successfully applied to various tasks, including image and speech recognition, natural language processing, anomaly detection, and even playing games.
>
> Neural networks have different architectures, such as feedforward neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), and more. Each architecture is designed to address specific types of problems and data structures, but they all share the fundamental concept of interconnected neurons and the ability to learn from data.

> When we train a model, a key concern is to ensure that our model _generalizes_—that is, that it learns general lessons from our data which also apply to new items it will encounter, so that it can make good predictions on those items.

Asking ChatGPT _If I need to avoid the model overfitting, is splitting data between training data and validation data a way to avoid overfitting_?. Some of the more salient points from the response:

> Yes, splitting the data between training and validation sets is a commonly used technique to help avoid overfitting in machine learning models, including neural networks.
>
> Overfitting occurs when the model becomes too specialized in capturing the patterns and noise present in the training data
>
> After each training iteration, the model's performance is evaluated on the validation set, providing an estimate of how well the model is generalizing to unseen data.
>
> To get a more reliable estimate of the model's performance, it's common to have a separate test set that is completely unseen by the model until the final evaluation. The test set provides an unbiased estimate of how well the model is expected to perform in the real world.

### Deep Learning Is Not Just for Image Classification

> Creating a model that can recognize the content of every individual pixel in an image is called _segmentation_

Explains how this is used for autonomous vehicles. We visualize how well segmentation is happening by color coding the pixels. The abstract image shows how well the segmentation worked.

> Let's move on to something much less sexy, but perhaps significantly more widely commercially useful: building models from plain tabular data.
>
> jargon: Tabular: Data that is in the form of a table, such as from a spreadsheet, database, or CSV file. A tabular model is a model that tries to predict one column of a table based on information in other columns of the table.
>
> It turns out that looks very similar too. Here is the code necessary to train a model that will predict whether a person is a high-income earner, based on their socioeconomic background:

```python
from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(3)
learn.show_results()
```

> As you see, we had to tell fastai which columns are categorical (that is, contain values that are one of a discrete set of choices, such as occupation) and which are continuous (that is, contain a number that represents a quantity, such as age).
>
> There is no pretrained model available for this task (in general, pretrained models are not widely available for any tabular modeling tasks, although some organizations have created them for internal use), so we don't use `fine_tune` in this case. Instead we use `fit_one_cycle`, the most commonly used method for training fastai models from scratch (i.e. without transfer learning)

`fine_tune` can still sometimes work with non-pretrained models. It will be explained in a future chapter. Experiment with both.

> We can use the same `show_results` call we saw earlier to view a few examples of user and movie IDs, actual ratings, and predictions

> fast.ai has spent a lot of time creating cut-down versions of popular datasets that are specially designed to support rapid prototyping and experimentation, and to be easier to learn with. In this book we will often start by using one of the cut-down versions and later scale up to the full-size version (just as we're doing in this chapter!). In fact, this is how the world’s top practitioners do their modeling in practice; they do most of their experimentation and prototyping with subsets of their data, and only use the full dataset when they have a good understanding of what they have to do.

### clean

If you hit a "CUDA out of memory error" after running this cell, click on the menu Kernel, then restart. Instead of executing the cell above, copy and paste the following code in it:

```python
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', bs=32)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)
```

This reduces the batch size to 32 (we will explain this later). If you keep hitting the same error, change 32 to 16.

### Sidebar: The Order Matters

> The outputs themselves can be deceiving, because they include the results of the last time the cell was executed; if you change the code inside a cell without executing it, the old (misleading) results will remain.

> In general, when experimenting, you will find yourself executing cells in any order to go fast (which is a super neat feature of Jupyter Notebook), but once you have explored and arrived at the final version of your code, make sure you can run the cells of your notebooks in order (your future self won't necessarily remember the convoluted path you took otherwise!).

--

> Tabular: Data that is in the form of a table, such as from a spreadsheet, database, or CSV file. A tabular model is a model that tries to predict one column of a table based on information in other columns of the table.

```python
from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
		# tells which cols are categorical
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
	 	# tells which cols are continues (ie have a number representing a quantity)
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)

# no pretrained model is available (generally pretrained models are not widely available for tabular modeling tasks), so we use `fit_one_cycle()`. 3 for 3 epochs
learn.fit_one_cycle(3)
```

```python
from fastai.collab import *
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
learn = collab_learner(dls, y_range=(0.5,5.5))
# per Fast.ai, this doesn't use a pretrained model, but we can still use `fine_tune()` - "it's best to experiment" with fine_tune() and fits_one_cycle() and see which one works best.
learn.fine_tune(10)
```

### Datasets: food for models

> Fast.ai has spent a lot of time creating cut-down versions of popular datasets that are specially designed to support rapid prototyping and experimentation, and to be easier to learn with. In this book we will often start by using one of the cut-down versions and later scale up to the full-size version (just as we're doing in this chapter!). In fact, this is how the world’s top practitioners do their modeling in practice; they do most of their experimentation and prototyping with subsets of their data, and only use the full dataset when they have a good understanding of what they have to do.

### Validation Sets and Test Sets

> One way to understand this situation is that, in a sense, we don't want our model to get good results by "cheating." If it makes an accurate prediction for a data item, that should be because it has learned characteristics of that kind of item, and not because the model has been shaped by actually having seen that particular item.

> Splitting off our validation data means our model never sees it in training and so is completely untainted by it, and is not cheating in any way. Right?
>
> In fact, not necessarily. The situation is more subtle. This is because in realistic scenarios we rarely build a model just by training its weight parameters once. Instead, we are likely to explore many versions of a model through various modeling choices regarding network architecture, learning rates, data augmentation strategies, and other factors we will discuss in upcoming chapters. Many of these choices can be described as choices of _hyperparameters_. The word reflects that they are parameters about parameters, since they are the higher-level choices that govern the meaning of the weight parameters.

> The problem is that even though the ordinary training process is only looking at predictions on the training data when it learns values for the weight parameters, the same is not true of us. We, as modelers, are evaluating the model by looking at predictions on the validation data when we decide to explore new hyperparameter values! So subsequent versions of the model are, indirectly, shaped by us having seen the validation data. Just as the automatic training process is in danger of overfitting the training data, we are in danger of overfitting the validation data through human trial and error and exploration.
>
> The solution to this conundrum is to introduce another level of even more highly reserved data, the _test set_. Just as we hold back the validation data from the training process, we **must hold back the test set data even from ourselves**. It cannot be used to improve the model; it can only be used to evaluate the model at the very end of our efforts.

Think about it like this:

> if you're considering bringing in an external vendor or service, make sure that you hold out some test data that the vendor never gets to see. Then you check their model on your test data, using a metric that you choose based on what actually matters to you in practice, and you decide what level of performance is adequate.

> For example, Kaggle had a competition to predict the sales in a chain of Ecuadorian grocery stores. Kaggle's training data ran from Jan 1 2013 to Aug 15 2017, and the test data spanned Aug 16 2017 to Aug 31 2017. That way, the competition organizer ensured that entrants were making predictions for a time period that was in the future, from the perspective of their model. This is similar to the way quant hedge fund traders do back-testing to check whether their models are predictive of future periods, based on past data.

It might be helpful to go back to the chapter 1 questions that you don't remember answers to and write them out here at some point.

> What is the difference between classification and regression?

Classification is a supervised learning task where the goal is to assign input instances to predefined categories or classes. The output is a discrete label representing the predicted class. (ie adding discrete labels to things)

Regression is a supervised learning task that aims to predict a continuous numerical value or a real-valued output. The output is a continuous value within a range. (ie quantitative prediction or continuous values)

## Lesson 2 Lecture

- `?verify_images` will give a breif summary
- `??verify_images` will provide source code
- I don't see autocomplete in paperspace...
- Unrelated to the documentation but still very useful: to get help at any point if you get an error, type %debug in the next cell and execute to open the Python debugger, which will let you inspect the content of every variable.

## Chapter 2 - Production

-[Paperspace Chapter 2](https://console.paperspace.com/evanemolo/notebook/rjnw81udus20vir?file=%2F02_production.ipynb)

### Starting Your Project

> So where should you start your deep learning journey? The most important thing is to ensure that you have some project to work on—it is only through working on your own projects that you will get real experience building and using models. When selecting a project, the most important consideration is data availability.

> The goal is not to find the "perfect" dataset or project, but just to get started and iterate from there.

> As you work through this book, we suggest that you complete lots of small experiments, by running and adjusting the notebooks we provide, at the same time that you gradually develop your own projects.

> Especially when you are just starting out with deep learning, it's not a good idea to branch out into very different areas, to places that deep learning has not been applied to before. That's because if your model does not work at first, you will not know whether it is because you have made a mistake, or if the very problem you are trying to solve is simply not solvable with deep learning.

### The State of Deep Learning

#### Computer Vision

> There is no general way to check what types of images are missing in your training set, but we will show in this chapter some ways to try to recognize when unexpected image types arise in the data when the model is being used in production (this is **known as checking for out-of-domain data**).

> One approach that is particularly helpful is to synthetically generate variations of input images, such as by rotating them or changing their brightness and contrast; this is called **data augmentation**

Jeremy showed in the lecture how we random cropping of images can be an effective way for data augmentation.

> Another point to consider is that although your problem might not look like a computer vision problem, it might be possible with a little imagination to turn it into one. For instance, if what you are trying to classify are sounds, you might try converting the sounds into images of their acoustic waveforms and then training a model on those images.

#### Text (natural language processing)

#### Combining text and images

> For example, a deep learning model can be trained on input images with output captions written in English, and can learn to generate surprisingly appropriate captions automatically for new images!

#### Tabular data

> Recommendation systems are really just a special type of tabular data.

#### Recommendation systems

#### Other data types

### The Drivetrain Approach

> To ensure that your modeling work is useful in practice, you need to consider how your work will be used.

Defined Objective (what is the outcome I am trying to achieve?) -> Levers (inputs we control) -> Data (what data do we need to collect?) -> Model (how the levers influence the objective)

### Cleaning

> To remove all the failed images, you can use unlink on each of them. Note that, like most fastai functions that return a collection, verify_images returns an object of type L, which includes the map method. This calls the passed function on each element of the collection:

```python
failed = verify_images(fns)
failed.map(Path.unlink);
```

### From Data to DataLoaders

> `DataLoaders` is a thin class that just stores whatever DataLoader objects you pass to it, and makes them available as `train` and `valid`. Although it's a very simple class, it's very important in fastai: it provides the data for your model.

> With this API you can fully customize every stage of the creation of your DataLoaders. Here is what we need to create a DataLoaders for the dataset that we just downloaded:

```python
bears = DataBlock(
		# tuple has the _independent variable_ first, _dependent variable_ second
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
```

> The _independent variable_ is the thing we are using to make predictions from, and the _dependent variable_ is our target.

> The `get_image_files` function takes a path, and returns a list of all of the images in that path

> In this case, however, we simply want to split our training and validation sets randomly. However, we would like to have the same training/validation split each time we run this notebook, so we fix the random seed **(computers don't really know how to create random numbers at all, but simply create lists of numbers that look random; if you provide the same starting point for that list each time—called the seed—then you will get the exact same list each time)**:

```python
splitter=RandomSplitter(valid_pct=0.2, seed=42)
```

> The independent variable is often referred to as `x` and the dependent variable is often referred to as `y`. Here, we are telling fastai what function to call to create the labels in our dataset:

```python
get_y=parent_label
```

> `parent_label` is a function provided by fastai that simply gets the name of the folder a file is in.

The types of bears are seperated by folder.

> Our images are all different sizes, and this is a problem for deep learning: we don't feed the model one image at a time but several of them (what we call a _mini-batc)h_). To group them in a big array (usually called a _tensor_)

> Item transforms are pieces of code that run on each individual item, whether it be an image, category, or so forth. fastai includes many predefined _transforms_; we use the `Resize` transform here:

```python
item_tfms=Resize(128)
```

> This command has given us a `DataBlock` object. This is like a template for creating a DataLoaders. We still need to tell fastai the actual source of our data—in this case, the path where the images can be found:

```python
# dataloaders...
dls = bears.dataloaders(path)
```

> what we normally do in practice is to randomly select part of the image, and crop to just that part. On each epoch (which is one complete pass through all of our images in the dataset) we randomly select a different part of each image. This means that our model can learn to focus on, and recognize, different features in our images. It also reflects how images work in the real world: different photos of the same thing may be framed in slightly different ways.

Jeremy covers ☝🏼 in the lecture.

```python
bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)
```

It looks like `.new()` is used to apply a new transformer and will return a new `DataBlock` with the updated transforms.

### Data Augmentation

> _Data augmentation_ refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data. Examples of common data augmentation techniques for images are rotation, flipping, perspective warping, brightness changes and contrast changes.

```python
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```

> or natural photo images such as the ones we are using here, a standard set of augmentations that we have found work pretty well are provided with the `aug_transforms` function.

### Training Your Model, and Using It to Clean Your Data

```python
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)
```

Now create a learner

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```

Explanation of a _confusion matrix_: basically a matrix showing actual on one axis and predicted on another, and display how well predictions aligned with the actual data categories.

### [Lesson 3](https://course.fast.ai/Lessons/lesson3.html#links-from-the-lesson)

[Lesson 0](https://www.youtube.com/watch?v=gGxe2mN3kAg) - worth watching

Suggestion on learning the lessons:

1. watch lecture
2. run notebook & experiment
3. reproduce results
   - look at `clean` folder; text is removed. you can run the code sections.
4. repeat with different dataset

Kaggle: "which image models are best?" notebook

    - By Jeremy
    - has a graph showing slow vs accurate

train -> export to pkl file -> predict

```python
catagories = learn.dls.vocab
```

then does a `zip()`, to map the categories to the accuracy

Inspecting the model output

    - something like a tree "deep learning"

**Check out all the resources for lesson 3!**

```python
# inside a notebook
from ipywidgets import  interact
@interact(a, b, b)
def plot(a, b, c):
	pass
```

Loss function: `mse` (mean square error)

    - calculates the loss...
    - adjusting the params he is attempting to get a smaller mse

Derivative: does the increase of input how much does the output increase or decrease

Slope is the gradient

When first prototyping stick with the faster, less accurate models

Matrix multiplication is the foundation of deep learning
matrixmultiplication.xyz
GPUs are excellence for it (tensor cores)

Kaggle titanic competition - data from titanic - ongoing comp

Binary categorical values - turning non number, or numbers that don't make sense to referent, to binary values in new cols

Normalizing the data - in the titanic case, dividing by the max to get a nuymber betwene 0 and 1

    - use a log for something like money (log_fare)
    - `y = mx + b` b is the constant term. the trick for ML is a new col of just 1s!!!

Next lesson..."Getting started with NLP for absolute beginners" kaggle notebook. Read before next

## Chapter 3 - Data Ethics

> There is no such thing as a completely debiased dataset.

> Machine learning can create feedback loops:: Small amounts of bias can rapidly increase exponentially due to feedback loops.

> One proposed approach is to develop some form of digital signature, to implement it in a seamless way, and to create norms that we should only trust content that has been verified.

https://hbr.org/2019/03/how-will-we-prevent-ai-based-forgery

> Clean air and clean drinking water are public goods which are nearly impossible to protect through individual market decisions, but rather require coordinated regulatory action. Similarly, many of the harms resulting from unintended consequences of misuses of technology involve public goods, such as a polluted information environment or deteriorated ambient privacy. Too often privacy is framed as an individual right, yet there are societal impacts to widespread surveillance (which would still be the case even if it was possible for a few individuals to opt out).
