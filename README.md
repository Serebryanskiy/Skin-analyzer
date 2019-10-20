# Skin analyzer : Android application for lesion classification

Have you ever wondered whether this thing in your house is a cat or not? Well, now you can know for sure!

An elegant android application with which a user can take a photo and figure out whether there is a cat in it.

## Dataset

The Google Open Image V5 dataset has been chosen for this application. The whole dataset covers 6000 categories and ~9 million images with total size of 18TB. We have chosen a Subset with Bounding Boxes (600 classes with total size of 561GB) as the  specific images can be downloaded directly in the subset. https://storage.googleapis.com/openimages/web/download.html

There are two reasons taken into account for choice of OIV5. The first one is that the dimensions of images are high enough to train a network with a relativity large input shape. (The Images in a dataset have 1024x600 dimensions on average). The second reason is to have non-cat class images that more likely can represent an average cell phone photo image.  Subset with Bounding Boxes has 600 classes with majority in such classes as Person, Land vehicle, Furniture, Food, and Building.

Steps to construct training dataset from Subset with Bounding Boxes:
* [Download train image’s indexes csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-annotations-bbox.csv)
* [Download test image’s indexes csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test-annotations-bbox.csv)
* Download every image corresponding to [Cats class](./scripts/dl_cats.py) (14025 images)
* Download [test data set](https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/test.zip)
* [Remove](./scripts/rm.py) all files corresponding to IsDepiction attribute (e.g., a cartoon or drawing of the object, not a real physical instance) and move cats images from test dir.
* Choose randomly 14025 images from test 
* Make train, validation, test subsets as (70%/15%/15%)

Or you can simply explore and [download](https://drive.google.com/drive/folders/1bKuF3p7DAhR7fvZwLdivT2ZFUCNjJzAK) constructed dataset from Google Drive. 

## Mode
There is always a trade-off between costs spent for a model training and the quality of a model. A practical approach is to use transfer learning — transferring the network weights trained on a previous task like ImageNet to a new task — to adapt a pre-trained deep classifier to our own requirements.
In our app we are going to use transfer learning using Keras. One thing we need to take into account when making mobile application is the size of a model and it’s efficiency. Let’s look at the following table with models specs.

<hr />
<table>
<thead>
<tr>
<th>Model</th>
<th align="right">Size</th>
<th align="right">Top-1 Accuracy</th>
<th align="right">Top-5 Accuracy</th>
<th align="right">Parameters</th>
<th align="right">Depth</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="#xception">Xception</a></td>
<td align="right">88 MB</td>
<td align="right">0.790</td>
<td align="right">0.945</td>
<td align="right">22,910,480</td>
<td align="right">126</td>
</tr>
<tr>
<td><a href="#vgg16">VGG16</a></td>
<td align="right">528 MB</td>
<td align="right">0.713</td>
<td align="right">0.901</td>
<td align="right">138,357,544</td>
<td align="right">23</td>
</tr>
<tr>
<td><a href="#vgg19">VGG19</a></td>
<td align="right">549 MB</td>
<td align="right">0.713</td>
<td align="right">0.900</td>
<td align="right">143,667,240</td>
<td align="right">26</td>
</tr>
<tr>
<td><a href="#resnet">ResNet50</a></td>
<td align="right">98 MB</td>
<td align="right">0.749</td>
<td align="right">0.921</td>
<td align="right">25,636,712</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#resnet">ResNet101</a></td>
<td align="right">171 MB</td>
<td align="right">0.764</td>
<td align="right">0.928</td>
<td align="right">44,707,176</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#resnet">ResNet152</a></td>
<td align="right">232 MB</td>
<td align="right">0.766</td>
<td align="right">0.931</td>
<td align="right">60,419,944</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#resnet">ResNet50V2</a></td>
<td align="right">98 MB</td>
<td align="right">0.760</td>
<td align="right">0.930</td>
<td align="right">25,613,800</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#resnet">ResNet101V2</a></td>
<td align="right">171 MB</td>
<td align="right">0.772</td>
<td align="right">0.938</td>
<td align="right">44,675,560</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#resnet">ResNet152V2</a></td>
<td align="right">232 MB</td>
<td align="right">0.780</td>
<td align="right">0.942</td>
<td align="right">60,380,648</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#resnet">ResNeXt50</a></td>
<td align="right">96 MB</td>
<td align="right">0.777</td>
<td align="right">0.938</td>
<td align="right">25,097,128</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#resnet">ResNeXt101</a></td>
<td align="right">170 MB</td>
<td align="right">0.787</td>
<td align="right">0.943</td>
<td align="right">44,315,560</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#inceptionv3">InceptionV3</a></td>
<td align="right">92 MB</td>
<td align="right">0.779</td>
<td align="right">0.937</td>
<td align="right">23,851,784</td>
<td align="right">159</td>
</tr>
<tr>
<td><a href="#inceptionresnetv2">InceptionResNetV2</a></td>
<td align="right">215 MB</td>
<td align="right">0.803</td>
<td align="right">0.953</td>
<td align="right">55,873,736</td>
<td align="right">572</td>
</tr>
<tr>
<td><a href="#mobilenet">MobileNet</a></td>
<td align="right">16 MB</td>
<td align="right">0.704</td>
<td align="right">0.895</td>
<td align="right">4,253,864</td>
<td align="right">88</td>
</tr>
<tr>
<td><a href="#mobilenetv2">MobileNetV2</a></td>
<td align="right">14 MB</td>
<td align="right">0.713</td>
<td align="right">0.901</td>
<td align="right">3,538,984</td>
<td align="right">88</td>
</tr>
<tr>
<td><a href="#densenet">DenseNet121</a></td>
<td align="right">33 MB</td>
<td align="right">0.750</td>
<td align="right">0.923</td>
<td align="right">8,062,504</td>
<td align="right">121</td>
</tr>
<tr>
<td><a href="#densenet">DenseNet169</a></td>
<td align="right">57 MB</td>
<td align="right">0.762</td>
<td align="right">0.932</td>
<td align="right">14,307,880</td>
<td align="right">169</td>
</tr>
<tr>
<td><a href="#densenet">DenseNet201</a></td>
<td align="right">80 MB</td>
<td align="right">0.773</td>
<td align="right">0.936</td>
<td align="right">20,242,984</td>
<td align="right">201</td>
</tr>
<tr>
<td><a href="#nasnet">NASNetMobile</a></td>
<td align="right">23 MB</td>
<td align="right">0.744</td>
<td align="right">0.919</td>
<td align="right">5,326,716</td>
<td align="right">-</td>
</tr>
<tr>
<td><a href="#nasnet">NASNetLarge</a></td>
<td align="right">343 MB</td>
<td align="right">0.825</td>
<td align="right">0.960</td>
<td align="right">88,949,818</td>
<td align="right">-</td>
</tr>
</tbody>
</table>
<p>The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.</p>
<p>Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.</p>
<hr />

If we would carefully examine the table we can find out that Xception model size is only 88 Mb while it has one of the best performances. Taking that into account we choose Xception model as our base model.

The model training is implemented in jupyter [notebook](./training_model.ipynb). Short description of model training steps:
* Set input size and batch size.
* Preprocess images using Keras ImageDataGenerator (actually preprocessing is done only at the beginning of training)
* Load Imagenet pretrained Xception model
* Freeze first 110 layers
* Set ModelCheckpoint callback
* Train the model (change params, repeat)
* Select the best model and check it on test set
* Convert selected model to TFLite Flatbuffe for mobile application.

We managed to get 98.5% accuracy on validation set and 98.1% accuracy on test set. There are ways to improve accuracy even further, but it’s good enough for that type of application

## Android application 

We’ll use TensorFlowLite to plug in our ML model to Android. The whole process can be described in four steps: 

* Get camera permission and take a photo
* Preprocess bitmap to meet model’s input requirements
* Feed preprocess bitmap to TensorFlow Lite
* Get and display classification probabilities

<img src="images/example.jpg?raw=true" />

## Build and run

### Requirements

*   Android Studio 3.2 (installed on a Linux, Mac or Windows machine)

*   Android device in
    [developer mode](https://developer.android.com/studio/debug/dev-options)
    with USB debugging enabled or virtual device (AVD)

*   USB cable (to connect Android device to your computer)


### Step 1. Clone the Catificator source code

Clone the Catificator GitHub repository to your computer to get the
application.

```
git clone https://github.com/Serebryanskiy/Catificator-2.0.git
```

Open the Catificator source code in Android Studio. To do this, open Android
Studio and select `Open an existing project`, setting the folder to
`examples/lite/examples/image_classification/android`

<img src="images/classifydemo_img1.png?raw=true" />

### Step 2. Build the Android Studio project

Select `Build -> Make Project` and check that the project builds successfully.
You will need Android SDK configured in the settings. You'll need at least SDK
version 23. The `build.gradle` file will prompt you to download any missing
libraries.

The file `download.gradle` directs gradle to download the two models used in the
example, placing them into `assets`.

<img src="images/classifydemo_img4.png?raw=true" style="width: 40%" />

<img src="images/classifydemo_img2.png?raw=true" style="width: 60%" />

<aside class="note"><b>Note:</b><p>`build.gradle` is configured to use
TensorFlow Lite's nightly build.</p><p>If you see a build error related to
compatibility with Tensorflow Lite's Java API (for example, `method X is
undefined for type Interpreter`), there has likely been a backwards compatible
change to the API. You will need to run `git pull` in the examples repo to
obtain a version that is compatible with the nightly build.</p></aside>

### Step 3. Install and run the app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

<img src="images/classifydemo_img5.png?raw=true" style="width: 60%" />

<img src="images/classifydemo_img6.png?raw=true" style="width: 70%" />

<img src="images/classifydemo_img7.png?raw=true" style="width: 40%" />

<img src="images/classifydemo_img8.png?raw=true" style="width: 80%" />

To test the app, open the app called `Catificator 2.0` on your device. When you run
the app the first time, the app will request permission to access the camera.
Re-installing the app may require you to uninstall the previous installations.

## Assets folder
_Do not delete the assets folder content_. If you explicitly deleted the
files, choose `Build -> Rebuild` to re-download the deleted model files into the
assets folder.

