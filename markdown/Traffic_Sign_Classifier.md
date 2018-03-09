
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file='valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of validation examples = 4410
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

fig, ax = plt.subplots()
ax.hist(y_train, n_classes, normed=1)
fig.tight_layout()
plt.show()

plt.imshow(X_train[0])
```


![png](output_8_0.png)





    <matplotlib.image.AxesImage at 0x7fc0e01aa9b0>




![png](output_8_2.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import numpy as np

def preprocess(x):
    # x = np.dot(x[...,:3], [[0.299], [0.587], [0.114]])
    # x = (x.astype(np.float32) - 128) / 128
    # x = np.dot(x[...,:3], [[0.333], [0.333], [0.334]])
    x = (x.astype(np.float32) - 128) / (3 * 128)
    return x
X_train_gray = preprocess(X_train)
X_valid_gray = preprocess(X_valid)
X_test_gray = preprocess(X_test)

from sklearn.utils import shuffle
X_train_gray, y_train = shuffle(X_train_gray, y_train)
```

### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
input_x = tf.placeholder(tf.float32, (None, ) + X_train_gray[0].shape)
input_y = tf.placeholder(tf.uint8, (None))
y = tf.one_hot(input_y, n_classes)

keep_prob0 = tf.placeholder(tf.float32)
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables
    # for the weights and biases for each layer.
    mu = 0
    sigma = 0.075

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    weight = tf.Variable(tf.truncated_normal([5, 5, 3, 12], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(12))
    conv_layer = tf.nn.bias_add(tf.nn.conv2d(x, weight, strides=[1,1,1,1], padding='VALID'), bias)

    # Activation.
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.nn.dropout(conv_layer, keep_prob0)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv_layer = tf.nn.max_pool(conv_layer, [1,2,2,1], [1,2,2,1], 'SAME')

    # Layer 2: Convolutional. Output = 10x10x16.
    weight = tf.Variable(tf.truncated_normal([5, 5, 12, 24], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(24))
    conv_layer = tf.nn.bias_add(tf.nn.conv2d(conv_layer, weight, strides=[1,1,1,1], padding='VALID'), bias)

    # Activation.
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.nn.dropout(conv_layer, keep_prob1)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv_layer = tf.nn.max_pool(conv_layer, [1,2,2,1], [1,2,2,1], 'SAME')

    # Flatten. Input = 5x5x16. Output = 400.
    weight = tf.Variable(tf.truncated_normal([5*5*24, 600], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(600))
    conv_layer = tf.reshape(conv_layer, [-1, 5*5*24])
    conv_layer = tf.add(tf.matmul(conv_layer, weight), bias)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    weight = tf.Variable(tf.truncated_normal([600, 150], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(150))
    conv_layer = tf.add(tf.matmul(conv_layer, weight), bias)

    # Activation.
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.nn.dropout(conv_layer, keep_prob2)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    weight = tf.Variable(tf.truncated_normal([150, 84], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(84))
    conv_layer = tf.add(tf.matmul(conv_layer, weight), bias)

    # Activation.
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.nn.dropout(conv_layer, keep_prob2)

    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    weight = tf.Variable(tf.truncated_normal([84, n_classes], mean = mu, stddev = sigma))
    bias = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(conv_layer, weight), bias)

    return logits
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
rate = 0.0005

logits = LeNet(input_x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

EPOCHS = 25
BATCH_SIZE = 50

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={input_x: batch_x, input_y: batch_y,
            keep_prob0: 1.0,keep_prob1: 1.0,keep_prob2: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_gray)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_gray, y_train = shuffle(X_train_gray, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_gray[offset:end], y_train[offset:end]
            sess.run(training_operation,
                    feed_dict={input_x: batch_x, input_y: batch_y,
                        keep_prob0: 0.8,keep_prob1: 0.7,keep_prob2: 0.5})

        train_accuracy = evaluate(X_train_gray, y_train)
        validation_accuracy = evaluate(X_valid_gray, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Train Accuracy, Validation Accuracy = {:.3f}, {:.3f}" \
                .format(train_accuracy, validation_accuracy))
        print()

        if validation_accuracy > 0.97:
            break

    saver.save(sess, './lenet')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_gray, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Training...
    
    EPOCH 1 ...
    Train Accuracy, Validation Accuracy = 0.823, 0.789
    
    EPOCH 2 ...
    Train Accuracy, Validation Accuracy = 0.909, 0.865
    
    EPOCH 3 ...
    Train Accuracy, Validation Accuracy = 0.952, 0.910
    
    EPOCH 4 ...
    Train Accuracy, Validation Accuracy = 0.971, 0.917
    
    EPOCH 5 ...
    Train Accuracy, Validation Accuracy = 0.980, 0.942
    
    EPOCH 6 ...
    Train Accuracy, Validation Accuracy = 0.981, 0.933
    
    EPOCH 7 ...
    Train Accuracy, Validation Accuracy = 0.985, 0.951
    
    EPOCH 8 ...
    Train Accuracy, Validation Accuracy = 0.989, 0.950
    
    EPOCH 9 ...
    Train Accuracy, Validation Accuracy = 0.990, 0.955
    
    EPOCH 10 ...
    Train Accuracy, Validation Accuracy = 0.990, 0.955
    
    EPOCH 11 ...
    Train Accuracy, Validation Accuracy = 0.992, 0.963
    
    EPOCH 12 ...
    Train Accuracy, Validation Accuracy = 0.991, 0.949
    
    EPOCH 13 ...
    Train Accuracy, Validation Accuracy = 0.993, 0.952
    
    EPOCH 14 ...
    Train Accuracy, Validation Accuracy = 0.994, 0.965
    
    EPOCH 15 ...
    Train Accuracy, Validation Accuracy = 0.994, 0.970
    
    EPOCH 16 ...
    Train Accuracy, Validation Accuracy = 0.994, 0.950
    
    EPOCH 17 ...
    Train Accuracy, Validation Accuracy = 0.994, 0.954
    
    EPOCH 18 ...
    Train Accuracy, Validation Accuracy = 0.995, 0.966
    
    EPOCH 19 ...
    Train Accuracy, Validation Accuracy = 0.996, 0.963
    
    EPOCH 20 ...
    Train Accuracy, Validation Accuracy = 0.993, 0.956
    
    EPOCH 21 ...
    Train Accuracy, Validation Accuracy = 0.996, 0.967
    
    EPOCH 22 ...
    Train Accuracy, Validation Accuracy = 0.996, 0.965
    
    EPOCH 23 ...
    Train Accuracy, Validation Accuracy = 0.995, 0.967
    
    EPOCH 24 ...
    Train Accuracy, Validation Accuracy = 0.997, 0.974
    
    Model saved
    Test Accuracy = 0.957


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
from PIL import Image
images = []
for i in range(0, 10):
    im = Image.open("GTSRB/Final_Test/Images/0000" + str(i) + ".ppm")
    im = im.resize((32, 32), Image.ANTIALIAS)
    images.append(np.array(im))
    plt.imshow(images[i])
    plt.show()
traffic_images = np.stack(images)
```


![png](output_20_0.png)



![png](output_20_1.png)



![png](output_20_2.png)



![png](output_20_3.png)



![png](output_20_4.png)



![png](output_20_5.png)



![png](output_20_6.png)



![png](output_20_7.png)



![png](output_20_8.png)



![png](output_20_9.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
traffic_image_inputs = preprocess(traffic_images)
fake_y = np.zeros(len(images), dtype = int)
results = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    output = sess.run(logits, feed_dict={input_x: traffic_image_inputs, input_y: fake_y,
        keep_prob0: 1.0,keep_prob1: 1.0,keep_prob2: 1.0})
    for i in range(0, 10):
        results.append(np.argmax(output[i]))
```

### Analyze Performance


```python
### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
import csv
expected = []
with open('GT-final_test.csv') as csvfile:
    rd = csv.reader(csvfile, delimiter=';')
    i = 0
    for row in rd:
        expected.append(row[-1])
        i = i + 1
        if i == 11:
            break
expected = expected[1:]
correct_count = 0
for a, b in zip(results, expected):
    if a == int(b):
        correct_count = correct_count + 1
print("Result with real traffic images: ", correct_count / 10.0)
```

    Result with real traffic images:  1.0


### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    probability = tf.nn.softmax(logits=logits)
    output = sess.run(probability, feed_dict={input_x: traffic_image_inputs, input_y: fake_y,
        keep_prob0: 1.0,keep_prob1: 1.0,keep_prob2: 1.0})
    for i in range(0, 10):
        mapping = [(x, j) for x, j in zip(output[i], range(0, len(output[i])))]
        print(sorted(mapping, reverse=True)[:5])
```

    [(1.0, 16), (7.1080641e-10, 9), (1.0109183e-10, 7), (4.6894749e-11, 3), (1.4506763e-12, 42)]
    [(0.99999928, 1), (7.4919762e-07, 4), (2.9745665e-08, 2), (8.3901499e-09, 0), (1.3125528e-11, 5)]
    [(1.0, 38), (1.3764925e-29, 40), (3.8931053e-30, 37), (2.4156224e-32, 36), (6.5857299e-33, 20)]
    [(1.0, 33), (3.8748436e-08, 34), (5.0187237e-09, 35), (2.6838642e-10, 15), (1.6287434e-10, 39)]
    [(0.99999893, 11), (7.6879104e-07, 21), (3.4780922e-07, 30), (1.8348071e-09, 27), (9.99246e-12, 28)]
    [(1.0, 38), (3.1638578e-27, 40), (1.3609089e-27, 37), (9.718068e-30, 36), (1.0023526e-30, 34)]
    [(1.0, 18), (6.9491121e-15, 26), (6.792804e-21, 27), (5.7132492e-27, 37), (9.4237553e-28, 11)]
    [(1.0, 12), (9.1664014e-13, 13), (2.310025e-13, 38), (1.6563254e-13, 42), (2.7238157e-14, 1)]
    [(1.0, 25), (6.8202365e-17, 20), (2.7364549e-17, 26), (9.0528781e-23, 29), (4.1635018e-24, 36)]
    [(1.0, 35), (3.172369e-10, 13), (1.1044466e-11, 36), (1.0469583e-11, 34), (2.9811967e-12, 3)]


### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```
