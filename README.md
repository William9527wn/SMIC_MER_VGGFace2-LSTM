# SMIC_MER_VGGFace2-LSTM
Using VGGFace2 and Long short-term memory network to do the Micro-expression Recognition Task by implementing SMIC datasets.


Step1:
 Using CNN visualization and GradCam HeatMap technique to visualize the pre-trained VGGFace2 model, which is basically a ResNet-50 CNN networks;
 
 
Step2:
  Pre-processing the data from SMIC micro-expression datastes. In directory: Pre-processing, 4 methods beed used for pre-processing. 
  
  @DataPreNoAug:Expending the number of sample videos by fixed-size slide window;
  
  @DataPreAug:Expending the number of sample videos by Random steps between frames within a slide window;
  
  @DataAugFlip:Variabl-size of slide window(if image length smaller than the seleframe, window size = image length); Mirror flipping
  
  @DataAugFlipPad:Fixed-size of slide window(if image length smaller than the seleframe, padding frames); Mirror flipping
  
  
Step3:
  Jointing scracthes from VGGFace2 Model to Bi-LSTM and replace the last classifier layer as SVM model.
  
  Step4:
  
 @Gradcam could be used to visualize the Grad-Cam heat map.
 
