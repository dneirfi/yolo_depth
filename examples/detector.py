# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb

dn.set_gpu(0)
#net = dn.load_net("cfg/yolo-thor.cfg", "/home/pjreddie/backup/yolo-thor_final.weights", 0)
net = dn.load_net("/mnt/sdcard/darknet/darknet/cfg/yolov3-tiny.cfg", "/mnt/sdcard/darknet_ori/yolov3-tiny.weights", 0)
#net = dn.load_net("/mnt/sdcard/darknet/darknet/cfg/yolov3.cfg", "/mnt/sdcard/darknet_ori/yolov3.weights", 0)
#meta = dn.load_meta("cfg/thor.data")
meta = dn.load_meta("/mnt/sdcard/darknet/darknet/cfg/coco.data")
#r = dn.detect(net, meta, "/mnt/sdcard/darknet/darknet/data/bedroom.jpg")

print('')
r, sec_4_forwarding = dn.detect(net, meta, "/mnt/sdcard/darknet/darknet/data/dog.jpg")
print (r); print('dog.jpg  :  sec for forwarding : {}'.format(sec_4_forwarding)); print('')
r, sec_4_forwarding = dn.detect(net, meta, "/mnt/sdcard/darknet/darknet/data/scream.jpg")
print (r); print('scream.jpg  :  sec for forwarding : {}'.format(sec_4_forwarding)); print('')

r, sec_4_forwarding = dn.detect(net, meta, "/mnt/sdcard/darknet/darknet/data/kite.jpg")
print (r); print('kite.jpg  :  sec for forwarding : {}'.format(sec_4_forwarding)); print('')
# And then down here you could detect a lot more images like:
r, sec_4_forwarding = dn.detect(net, meta, "/mnt/sdcard/darknet/darknet/data/eagle.jpg")
print (r); print('eagle.jpg  :  sec for forwarding : {}'.format(sec_4_forwarding)); print('')
r, sec_4_forwarding = dn.detect(net, meta, "/mnt/sdcard/darknet/darknet/data/giraffe.jpg")
print (r); print('giraffe.jpg  :  sec for forwarding : {}'.format(sec_4_forwarding)); print('')
r, sec_4_forwarding = dn.detect(net, meta, "/mnt/sdcard/darknet/darknet/data/horses.jpg")
print (r); print('horses.jpg  :  sec for forwarding : {}'.format(sec_4_forwarding)); print('')
r, sec_4_forwarding = dn.detect(net, meta, "/mnt/sdcard/darknet/darknet/data/person.jpg")
print (r); print('person.jpg  :  sec for forwarding : {}'.format(sec_4_forwarding)); print('')

