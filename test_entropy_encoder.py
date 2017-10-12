import caffe
import numpy as np
import lmdb
from binary_encoder import binary_encoder
def test_all():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    bcoder=binary_encoder()
    env=lmdb.open('f:/compress/imp_5_lmdb_test')
    #env=lmdb.open('f:/compress/imp_11_map_lmdb_test')
    num_bit=0
    with env.begin() as txn:
       cursor=txn.cursor()
       cursor.last()
       num_bit=int(cursor.key())
    print num_bit
    net=caffe.Net('./model/entropy/ent_v2_test.prototxt','./model/entropy/5_binary.caffemodel',caffe.TRAIN)
    #print model 
    l = 0
    for i in range(num_bit/1024):
        net.forward()
        #print net.blobs['loss'].data
        la = 0
        for j in range(1024):
            la += bcoder.coding_bit(net.blobs['label'].data[j],1-net.blobs['fc2'].data[j])
        #print la, 1024
        l+=la
        if i % 10 ==0:
            print l,1024*i+1024
    if num_bit%1024>0:
       net.forward()
       la = 0
       for j in range(num_bit%1024):
           la += bcoder.coding_bit(net.blobs['label'].data[j],1-net.blobs['fc2'].data[j])
       l+=la
    print "bits after arithmatic encoding: %d, original bits: %d"%(l,num_bit)
    print "%.3f bpp"%(l/8951808.0)
def modify_test_prototxt(binary=True,model_idx=5):
    f=open('./model/entropy/ent_v2_test.prototxt')
    lines=f.readlines()
    f.close()
    if not binary: lines[6]='\tsource: "f:/compress/imp_%d_map_lmdb_test"\n'%model_idx
    else: lines[6]='\tsource: "f:/compress/imp_%d_lmdb_test"\n'%model_idx
    f=open('./model/entropy/ent_v2_test.prototxt','w')
    f.writelines(lines)
    f.close()
def test_per_image(binary=True,channel_128=False,model_idx=4):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    bcoder=binary_encoder()
    modify_test_prototxt(binary,model_idx)
    if binary: prex='binary'
    else: prex='map'
    net=caffe.Net('./model/entropy/ent_v2_test.prototxt','./model/entropy/%d_%s.caffemodel'%(model_idx,prex),caffe.TRAIN)
    if binary:
        f=open('./model/entropy/binary_size/%d.txt'%model_idx)
        llist=[int(pt[:-1]) for pt in f.readlines()]
        f.close()
        f=open('./model/entropy/ratio/%d_binary.txt'%model_idx,'w')
    else:
        if channel_128:llist=[752*496/64*5*i for i in xrange(1,25)]
        else: llist=[752*496/64*4*i for i in xrange(1,25)]
        f=open('./model/entropy/ratio/%d_map.txt'%model_idx,'w')
    pidx=0 
    num_pic=24
    l = 0
    ratio=0
    for i in range((llist[-1]+1023)/1024):
        if pidx>=num_pic: break
        net.forward()
        for j in range(1024):
            if i*1024 +j == llist[pidx]:
                f.write('%.3f\n'%(l/(752.0*496.0)))
                pidx+=1
                ratio+=l/(752.0*496.0)
                print 'image: %d binary ratio: %.3f'%(pidx,l/(752.0*496.0))
                l=0
                if pidx>=num_pic: break
            l += bcoder.coding_bit(net.blobs['label'].data[j],1-net.blobs['fc2'].data[j])
    f.close()
    print ratio/24.0
def combine_binary_and_map():
    for idx in xrange(1,6):
        f=open('./model/entropy/ratio/%d_binary.txt'%idx)
        bn=[float(pt[:-1]) for pt in f.readlines()]
        f.close()
        f=open('./model/entropy/ratio/%d_map.txt'%idx)
        mp=[float(pt[:-1]) for pt in f.readlines()]
        f.close()
        res=[bn[i]+mp[i] for i in range(24)]
        print sum(bn)/24.0
        print sum(mp)/24.0
        print sum(res)/24.0
        f=open('./model/entropy/ratio/%d_final.txt'%idx,'w')
        for val in res:
            f.write('%.3f\n'%val)
        f.close()
if __name__ == '__main__':
    test_per_image(False,False,2)
    #combine_binary_and_map()