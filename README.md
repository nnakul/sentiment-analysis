## About
This project trains a *recurrent neural network* to learn the sentiments delivered by sentences, focussed on human feelings and emotions, like *anger*, *joy* etc. Given a sentence, the trained model categorizes the sentiment behind the sentence into one of these *five* emotions -- ***neutral***, ***joy***, ***sad***, ***anger*** or ***fear***.

## Training
The model was trained for *31 epochs* on a training set of *6358* samples, at an initial learning rate of *0.01* that was discretely reduced to *0.00001*. Training took almost *130 minutes* to complete.

&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120900641-8b80ee80-c653-11eb-833d-8bab5e9dff6c.png" width = '400' height = '320'>
&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120900640-8a4fc180-c653-11eb-9bfe-6b88628f2826.png" width = '400' height = '320'>

## Evaluation
The trained model obtained an accuracy of *60.1%* on the test dataset (made of *2120* samples), against an accuracy of *98.2%* on the training dataset. This much difference in accuracies on the train and test datasets is a clear indication of *over-fitting*. The model can be improved by *regularization* techniques and by monitoring the loss on a validation set while training so that the hyper-parameters can be tweaked accordingly to adjust the fitting.

The confusion matrices of the model for the train and test datasets are shown below. 
    
    0 : NEUTRAL
    1 : SAD
    2 : JOY
    3 : ANGER
    4 : FEAR
    
&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120900949-6ab99880-c655-11eb-9779-77ed4677e7e0.png" width = '400' height = '350'>
&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120900952-6c835c00-c655-11eb-93c2-0f51bd18f234.png" width = '400' height = '350'>

Some examples of the model's performance in predicting sentiments are shown in *ANALYSIS.ipynb*.
