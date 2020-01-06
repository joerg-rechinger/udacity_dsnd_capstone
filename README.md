# Udacity Data Science Nanodegree Capstone Project
## Dog Breed Classifier

This project is part of Udacity's Data Scientist Nanodegree program. I have decided to chose this project as the final project of the Nanodegree also called the Capstone Project.

### Libraries
Python 3.7+
Keras==2.0.9
OpenCV
Matplotlib
NumPy
glob
tqdm
Scikit-Learn
Flask
Tensorflow

### Project motivation
The goal of this project is to classify images of dogs according to their breed. When the image of a human is provided, it should recommend the best resembling dog breed. I decided to opt for this project as I found the topic of Neural Networks to be very fascinating and wanted to dive deeper into this with some practical work.

### Description of repository
The repository consists of the Jupyter Notebook files from the Udacity classroom, in both formats: dog_app.html and dog_app.ipynb. All credits for code examples here go to Udacity.
Moreover the folder web_app contains all code necessary for running the dog breed classifier app on your local machine.

### Running the code
For running the web app on your local machine, following these instructions
1. Make sure you have all necessary packages installed (if version is specified, then please refer to the one mentioned above for running the code without errors on your machine).
2. Git clone this repository
3. Within command line, cd to the cloned repo, and within the repository to the folder "web_app".
4. Run the following command in the app's directory to run the web app.
    `python run.py`
5. Go to http://0.0.0.0:3001/ to view the web app and input new pictures of dogs or humans – the app will tell you the resembling dog breed.

### Project Definition
The task was to develop an algorithm that takes an image as an input, pre-processes and transforms the image so that it can be fed into a CNN for classifying the breed of the dog. If a human image is uploaded, it should still tell the user what dog breed the human resembles most.

### Analysis
I decided to use a pre-trained ResNet50 model as this has shown very good results with regard to accuracy for image classification. In the provided classroom environment, my tests showed an a test accuracy of 82.6555%. This was accomplished by 20 epochs which ran very quickly on the provided GPU.
Moreover I decided to change the optimizer to "adam" when compiling the model as this was giving me higher accuracy results for the given use case.

### Conclusion
I was surprised by the good results of the algorithm. Without doing too much fine-tuning, the algorithm was already providing high accuracy and the predictions were mostly correct. For human faces it seems easier if the face has distinct features that resemble a certain dog breed. Otherwise it starts to guess from some features, but the results vary. For higher accuracy, the parameters could be further optimized, maybe also including more layers into the model. Also by providing an even bigger training data set, the classification accuracy could be improved further.

### Author
Jörg Rechinger

### Acknowledgements
Udacity for providing templates, data inputs and code snippets that were used to complete this project.
Credits for the main code basis of the web app go to https://medium.com/@venkinarayanan/tutorial-image-classifier-using-resnet50-deep-learning-model-python-flask-in-azure-4c2b129af6d2
Other sources of code not written by me are included in the Jupyter Notebook file.
