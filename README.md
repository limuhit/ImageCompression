# ImageCompression

Results of our model on Kodak dataset are available at https://pan.baidu.com/s/1p141-1Vp2YD0D9II5b7uWg with the extract code 1232.

This is the codes for paper "Learning Convolutional Networks for Content-weighted Image Compression".

This project is based on a modified version of caffe framework. Currently, we only offer the complied pycaffe python package for test. The pycaffe package is complied with the environment "Windows 10" "VS 2015" and "CUDA 8.0". The pycaffe package is available at "https://drive.google.com/file/d/0B-XAj3Bp3YhHbU5pdTNfc3l5OGs/view?usp=sharing&resourcekey=0-HWL7I6i6EKyaGQd7RYsGHg". After download it, just put it into the library of python and make sure the code "import caffe" works. Here, we recommend you to take use of Anaconda as the python environment and just move the "caffe" directory in the uncompressed folder into the path "AnacondaInstallPath/Lib/site-packages". After that, install python package "numpy","protobuf","lmdb" and "cv2" before runing the codes, otherwise you may meet with some errors. For installing these packages, you make take use of pip install or conda install. For example, just type "pip install numpy" in the command line to install the numpy package. 

Now, I will show you how to take use of the test codes for our image compression model. 

The file "test_imp.py" is used to test different models with different compression ratios. One model for one ratio. For each model, this file can generate the compressed image, calculate the PSNR metrics and put the compressed images under the directory "model/img". The importance map for each image is transformed as a black white picture and saved in the folder "model/img/imp".

The compression ratio of our model can be calculated in two parts. One part for the importance map, another for the binary codes. The file "create_lmdb_for binary_codes.py" is used to extract the context for each binary codes and put the context cubic into a lmdb database for further calculate the hit or miss possibility in the arithmetic coding. The database should be created for each model before testing the compression ratio.  The file "create_lmdb_for_imp_map.py" is used to prepare the data for calculating the ratio of the binary codes. After preparing the data, we can run the file "test_entropy_encoder.py" to calculate the compression ratio of the binary codes and the importance map. Finally, by adding the importance map ratio and the binary codes ratio, we can get the final ratio of our model.

If you have problem in testing our model, please contact me at "limuhit@gmail.com".

If you the codes, please cite the paper "Learning Convolutional Networks for Content-weighted Image Compression".


@inproceedings{li2018learning,

  title={Learning Convolutional Networks for Content-weighted Image Compression},
  
  author={Li, Mu and Zuo, Wangmeng and Gu, Shuhang and Zhao, Debin and Zhang, David},
  
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  
  pages={3214--3223},
  
  year={2018}
  
}
