import numpy as np
import caffe
import lmdb
import cv2
def binary16(img):
    res=np.zeros((4,img.shape[0],img.shape[1]),dtype=np.uint8)
    idx=(img>=8)
    res[3,idx]=1
    img[idx]=img[idx]-8
    idx=(img>=4)
    res[2,idx]=1
    img[idx]=img[idx]-4
    idx=(img>=2)
    res[1,idx]=1
    img[idx]=img[idx]-2
    res[0,img>0]=1
    return res
def binary32(img):
    res=np.zeros((5,img.shape[0],img.shape[1]),dtype=np.uint8)
    idx=(img>=16)
    res[4,idx]=1
    img[idx]=img[idx]-16
    idx=(img>=8)
    res[3,idx]=1
    img[idx]=img[idx]-8
    idx=(img>=4)
    res[2,idx]=1
    img[idx]=img[idx]-4
    idx=(img>=2)
    res[1,idx]=1
    img[idx]=img[idx]-2
    res[0,img>0]=1
    return res
def get_data(net,name_list,channel_128=False,shuffle=True):
    f=open(name_list,'r')
    flist=[]
    for pt in f.readlines():
        flist.append(pt[:-1])
    f.close()
    if shuffle: np.random.shuffle(flist)
    if len(flist)>160:
        flist=flist[:160]
    res=[]
    for pimg in flist[:]:
       img=cv2.imread(pimg)
       if img is None:
           continue
       print pimg
       if img.shape[0] % 16 >0:
           img=img[0:img.shape[0]-img.shape[0]%16,:]
       if img.shape[1] % 16 >0:
           img=img[:,0:img.shape[1]-img.shape[1]%16]  
       net.blobs['data'].reshape(1,3,img.shape[0],img.shape[1])
       data=(img.transpose(2,0,1)-127.5)/127.5
       net.blobs['data'].data[0]=data
       net.forward()
       if channel_128:
           imap=(net.blobs['imp_conv2'].data[0,0]*32).astype(np.uint8)
           mdata=binary32(imap)
       else:
           imap=(net.blobs['imp_conv2'].data[0,0]*16).astype(np.uint8)
           mdata=binary16(imap)
       net.blobs['data2'].reshape(1,mdata.shape[0],mdata.shape[1],mdata.shape[2])
       net.blobs['data2'].data[0]=mdata
       net.forward()
       for i in range(net.blobs['epack'].data.shape[0]):
           if net.blobs['elabel'].data[i,0,0,0]<1:continue
           res.append([net.blobs['epack'].data[i].astype(np.uint8),net.blobs['elabel'].data[i,0,0,0]-1])
    return res
def generate_lmdb(net,data,data_set_name,shuffle=True):
    if shuffle:np.random.shuffle(data)
    X=np.zeros((1,4,5,5),dtype=np.uint8)
    map_size=X.nbytes * len(data)*1.4
    env = lmdb.open('f:/compress/%s'%data_set_name,map_size)
    idx = 0
    datum=caffe.proto.caffe_pb2.Datum()
    datum.channels=4
    datum.height=5
    datum.width=5
    with env.begin(write=True) as txn:
        for tmp in data:
             datum.data=tmp[0].astype(np.uint8).tobytes()
             datum.label=int(tmp[1])
             stri_id='{:08}'.format(idx)
             idx = idx+1
             txn.put(stri_id.encode('ascii'),datum.SerializePartialToString())
             if idx % 100 == 0:
                print idx
if __name__ == '__main__':
    caffe.set_device(1)
    caffe.set_mode_gpu()
    train_flag=False
    model_idx=2
    channel_128=(model_idx>4)
    net=caffe.Net('./model/entropy/extract_entropy_package_for_imp_map.prototxt','./model/cmp/%d.caffemodel'%model_idx,caffe.TEST)
    if train_flag:
        data=get_data(net,'train_image_name_list.txt',channel_128)
        generate_lmdb(net,data,'imp_%d_map_lmdb_train'%model_idx)
    else:
        data=get_data(net,'./test_images/name.txt',channel_128,False)
        generate_lmdb(net,data,'imp_%d_map_lmdb_test'%model_idx,False)