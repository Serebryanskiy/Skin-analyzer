# Skin analyzer : Android application for skin lesion classification

Skin lesions are conditions that appear on a patientdue  to  many  different  reasons.  One  of  these  can  be  because of  an  abnormal  growth  in  skin  tissue,  defined  as  cancer.  This disease  plagues  more  than  14.1  million  patients  and  had  beenthe cause of more than 8.2 million deaths, worldwide

We tried to make an android application with which a user can take a photo and figure out whether the mole is maligant or benigin.

## Dataset

The HAM10000 dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

<img src="images/lesion-types.png?raw=true" />

More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by the lesion_id-column within the HAM10000_metadata file.

## Model
There is always a trade-off between costs spent for a model training and the quality of a model. A practical approach is to use transfer learning — transferring the network weights trained on a previous task like ImageNet to a new task — to adapt a pre-trained deep classifier to our own requirements.
In our app we are going to use transfer learning using Keras. One thing we need to take into account when making mobile application is the size of a model and it’s efficiency. We picked EfficentNet for the purpose of our application.

## About EfficientNet Models

EfficientNets rely on AutoML and compound scaling to achieve superior performance without compromising resource efficiency. The [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) has helped develop a mobile-size baseline network, **EfficientNet-B0**, which is then improved by the compound scaling method  to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:

* In high-accuracy regime, EfficientNet-B7 achieves the state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS. At the same time, the model is 8.4x smaller and 6.1x faster on CPU inference than the former leader, [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared to the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraints.


The model training is implemented in jupyter [notebook](./model.ipynb). Short description of model training steps:
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

