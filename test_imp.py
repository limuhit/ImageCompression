import numpy as np
import caffe
import cv2
import math
if __name__ == '__main__':
    caffe.set_device(1)
    caffe.set_mode_gpu()
    net=caffe.Net('./model/cmp/compress_v1_128_imp_deploy.prototxt','./model/cmp/5.caffemodel',caffe.TEST)
    f=open('./test_images/name.txt','r')
    imp_data=[]
    flist=[]
    for pt in f.readlines():
        flist.append(pt[:-1])
    f.close()
    prsum=0
    yprsum=0
    mrate=0
    idx=0
    num_p=len(flist)
    #num_p=23
    for pimg in flist[0:num_p]:
       print pimg
       img=cv2.imread(pimg)
       if img.shape[0] % 8 >0:
           img=img[0:img.shape[0]-img.shape[0]%8,:]
       if img.shape[1] % 8 >0:
           img=img[:,0:img.shape[1]-img.shape[1]%8]  
       net.blobs['data'].reshape(1,3,img.shape[0],img.shape[1])
       data=(img.transpose(2,0,1)-127.5)/127.5
       net.blobs['data'].data[0]=data
       net.forward()
       net.backward()
       gimg=net.blobs['pdata'].data[0]*127.5+127.5
       gimg[gimg<0]=0
       gimg[gimg>255]=255
       mrate=mrate+net.params['imap'][0].data[0,0,0,0]
       print net.params['imap'][0].data[0,0,0,0]
       pimg=net.blobs['imp_conv2'].data[0,0]*255
       gimg=gimg.transpose(1,2,0).astype(np.uint8)
       psnr=lambda x,y:10*math.log10(255.0*255.0/(np.sum(np.square(y.astype(np.float)-x))/float(x.size)))
       trimg=img.transpose(2,0,1)
       trorg=gimg.transpose(2,0,1)
       ytrans=lambda x:0.299*x[2]+0.587*x[1]+0.114*x[0]
       yimg=ytrans(trimg)
       yorg=ytrans(trorg)
       #cv2.imshow('y',trimg[0])
       #cv2.waitKey(0)
       yprsum+=psnr(yimg,yorg)
       print "y_psnr: "+str(psnr(yimg,yorg))
       print psnr(img,gimg)
       prsum+=psnr(img,gimg)
       idx+=1
       rt=net.params['imap'][0].data[0,0,0,0]
       cv2.imwrite('./model/img/%ds.png'%idx,img)
       cv2.imwrite('./model/img/%dg.png'%(idx),gimg)
       cv2.imwrite('./model/img/imp/%dm.png'%idx,pimg.astype(np.uint8))
       shown=False
       #shown=True
       if shown:
            cv2.imshow('imap',pimg.astype(np.uint8))
            cv2.imshow('src',img)
            cv2.imshow('dst',gimg)
            cv2.waitKey(0)
    print mrate/num_p
    print prsum/num_p
    print yprsum/num_p