# MedGANs

## About the Project

Medical Imaging is used by radiologists for diagnostic purposes and to check for abnormalities, and these imaging techniques involve radiation. Overexposure to radiation can have an adverse impact on the human body, and using less radiation gives us a noisy output. Hence, radiologists find it difficult as there is a trade-off between the amount of radiation that can be used and the quality of the image. Moreover, noise in medical images can occur due to fluctuation of photons, a reflection of radiations from the subject, or due to instrumental vibration or faults. The proposed approach is a pipeline which starts with denoising using GANs architecture, in which two models have been trained, one for handling all kinds of noise and the second one specifically for Poisson noise. Further, post-processing methods like single-shot HDR using Retinex Filtering and Edge Enhancement using unsharp masking have been done to get a structurally more similar and enhanced denoised image.

## File and Folder Structure

- <b>pipeline.py</b> -> This is the main file which imports all other functions for denoising using GANs, HDR and Edge Enhancement. 

- <b>models.py</b> -> This file contains the architecture and structure of Generator and Discriminator model.

- <b>denoiser_utils.py</b> -> The file contains the functions for performing denoising operations.

- <b>const.py</b> -> This file contains the constants used in this project like batch_size, image_size etc.

- <b>/ train_model</b> -> contains the .ipynb file with the main logic for training the GANs models.

- <b>/ ckpt</b> -> Folder contains the saved checkpoint for random noise denoiser and poisson denoiser in the form of .pth files

- <b>/ data</b> -> Folder contains the data used for training the GANs model

- <b>/ EdgeEnhance</b> -> Folder contains the edge enhancement module which is based on unsharp masking

- <b>/ HDR</b> -> Folder contains the whole logic for single image HDR using retinex filtering

- <b>/ output_images</b> -> These include the result images at each step of the pipeline

- <b>/ plot_metric</b> -> Contains .ipynb file for plotting the graphs and their outputs as well

- <b>/ test_data</b> -> Data being used for testing only

- <b>/ traditional_methods</b> -> Contains analysis done using Conventional Methods (Gaussian, NLM, Bilateral, Median Filters) and ADMM+TV, ADMM+DnCNN

