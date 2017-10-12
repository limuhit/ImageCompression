import os
import numpy as np
class binary_encoder:
    '''
    The encoder part is the copy from the PAQ coder which is a application of arithmetic coding.
    '''
    def __init__(self):
        self.x1=0
        self.x2=0xffffffff
    def restart(self):
        self.x1=0
        self.x2=0xffffffff
    def coding_bit(self,ch,mp):
        p = int(mp*65535.0)
        xdiff=self.x2-self.x1
        xmid=self.x1
        if xdiff>=0x10000000: xmid+=(xdiff>>16)*p
        elif xdiff>=0x1000000: xmid+=((xdiff>>12)*p)>>4
        elif xdiff>=0x100000: xmid+=((xdiff>>8)*p)>>8
        elif xdiff>=0x10000: xmid+=((xdiff>>4)*p)>>12
        else: xmid+=(xdiff*p)>>16
        if ch>0:self.x1=xmid+1
        else:self.x2=xmid
        l=0
        while ((self.x1^self.x2)&0xff000000)==0:
            l+=8
            self.x1=(self.x1<<8)&0xffffffff
            self.x2=((self.x2<<8)+255)&0xffffffff
        return l
