# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import argparse

caffe_root = '/home/cogrob/Research/software/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

parser = argparse.ArgumentParser(description='Deep dream python helper script.')
group = parser.add_mutually_exclusive_group()
group.add_argument('-r','--random', action='store_true', help='Start from a random image')
group.add_argument('-i','--image', type=str, help='The image to dream from. You can also use the -r flag to start from a random image')

group2 = parser.add_mutually_exclusive_group()
group2.add_argument('-g','--guided', type=str, help='Use an image to guide an optimization')
group2.add_argument('-c','--cl', type=int, help='Optimize for a class')

parser.add_argument('-m','--model', nargs=2, type=str, default=['/home/cogrob/Research/software/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel', '/home/cogrob/Research/software/caffe/models/bvlc_googlenet/deploy.prototxt'], help='Model file followed by deploy file')
parser.add_argument('-l','--layer', type=str, default='inception_4c/output', help='layer to dream')

args = parser.parse_args()
print args

net_fn   = args.model[1]
param_fn = args.model[0]
#model_path = caffe_root+'models/bvlc_googlenet/' # substitute your path here
#net_fn   = model_path + 'deploy.prototxt'
#param_fn = model_path + 'bvlc_googlenet.caffemodel'
caffe.set_mode_gpu()

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

#net = caffe.Classifier('tmp.prototxt', param_fn,
#                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
#                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
net = caffe.Classifier('tmp.prototxt', param_fn)
print("net finished training")
net.transformer.kReshape = True
net.transformer.set_raw_scale('data',255)
net.transformer.set_channel_swap('data',(2,1,0))
#net.transformer.set_mean('data',np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
net.transformer.set_mean('data',np.float32([104.0,116.0,122.0]))
caffe.set_mode_gpu()

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])
def objective_L2(dst):
        dst.diff[:] = dst.data

def setup_guide(guide):
    global guide_features
    end = 'inception_4c/output'
    h, w = guide.shape[:2]
    src, dst = net.blobs['data'], net.blobs[end]
    src.reshape(1,3,h,w)
    src.data[0] = preprocess(net, guide)
    net.forward(end=end)
    guide_features = dst.data[0].copy()
def objective_guide(dst):
    global guide_features
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

def objective_class(dst):
    global cl
    # slightly inhibit other classes
    dst.diff[:] = -0.1 * dst.data

    # we need a big gradient to force stuff to appear
    dst.diff[0, cl] = np.clip(5000 * dst.data[0, cl], 1000, 50000)

def make_step(net, step_size=1.5, end='inception_3b/output', jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_3a/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

img = None

if args.random == False:
    img = np.float32(PIL.Image.open(args.image))
else:
    img = np.float32(PIL.Image.fromarray((np.random.random((720,1280,3))*255).astype('uint8')))
    PIL.Image.fromarray(np.uint8(img)).save("frames/orig.jpg")

#for guiding
if args.guided != None:
    guide = np.float32(np.resize(PIL.Image.open(args.guided),(224,224,3)))
    setup_guide(guide)


frame = img
frame_i = 0
h, w = frame.shape[:2]
s = 0.05 # scale coefficient
for i in xrange(100):
    print 'calculating frame'
    #Normal
    if args.cl != None:
        # which class/feature to enhance
        cl = args.cl
        frame = deepdream(net, frame, 10, 4, 1.4, args.layer, objective=objective_class)
    elif args.guided != None:
        #Optimize guide
        frame = deepdream(net, frame, 10, 4, 1.4, args.layer, objective=objective_guide)
    else:
        frame = deepdream(net, frame, 10, 4, 1.4, args.layer)
    
    PIL.Image.fromarray(np.uint8(frame)).save("frames/%04d.jpg"%frame_i)
    frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
    frame_i += 1
